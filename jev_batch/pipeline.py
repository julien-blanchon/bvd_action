"""Bounded, ordered CPU preparation and GPU batch execution."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Protocol

from .types import (BatchConfig, ChoiceQuestion, EncodedVideo, Item, Prediction,
                    PreparedVideo, ProbabilityGuard, Sampling)

PreparedInput = PreparedVideo | EncodedVideo
ScheduledItem = tuple[Item, Future[PreparedInput]]


class BatchEngine(Protocol):
    """Minimal interface needed by the dataset-independent batch runner."""

    def prepare(self, source: Any, sampling: Sampling) -> PreparedInput: ...

    def predict_prepared_batch(
        self, inputs: list[PreparedInput], question: ChoiceQuestion,
        sampling: Sampling, *, decision_batch_size: int,
        guard: ProbabilityGuard | None,
    ) -> list[Prediction]: ...


logger = logging.getLogger(__name__)


class BatchRunner:
    """Run arbitrary input items; a source adapter may override preparation."""

    def __init__(self, engine: BatchEngine, question: ChoiceQuestion, *,
                 sampling: Sampling = Sampling(), config: BatchConfig = BatchConfig(),
                 guard: ProbabilityGuard | None = None,
                 prepare: Callable[[Item, Sampling], PreparedInput] | None = None) -> None:
        self.engine = engine
        self.question = question
        self.sampling = sampling
        self.config = config
        self.guard = guard
        self.prepare_item = prepare or (
            lambda item, settings: self.engine.prepare(item.source, settings))

    def _scheduled(self, items: Iterable[Item],
                   executor: ThreadPoolExecutor) -> Iterator[ScheduledItem]:
        pending: deque[ScheduledItem] = deque()
        for item in items:
            pending.append((item, executor.submit(self.prepare_item, item, self.sampling)))
            if len(pending) >= self.config.prefetch:
                yield pending.popleft()
        yield from pending

    def _groups(self, scheduled: Iterable[ScheduledItem]) -> Iterator[list[ScheduledItem]]:
        group: list[ScheduledItem] = []
        tokens = 0
        for pair in scheduled:
            try:
                count = int(pair[1].result().record["tokens"])
            except Exception:
                count = 0  # The result path reports the preparation error.
            if group and (len(group) >= self.config.vision_batch_size
                          or tokens + count > self.config.visual_token_budget):
                yield group
                group, tokens = [], 0
            group.append(pair)
            tokens += count
            if len(group) >= self.config.vision_batch_size:
                yield group
                group, tokens = [], 0
        if group:
            yield group

    def run(self, items: Iterable[Item]) -> Iterator[tuple[Item, Prediction]]:
        """Yield results in input order, retrying a failed GPU batch singly."""
        workers = min(self.config.decode_workers, max(1, self.config.prefetch - 1))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for group in self._groups(self._scheduled(items, executor)):
                output: list[Prediction | None] = [None] * len(group)
                ready: list[tuple[int, PreparedInput]] = []
                for index, (item, future) in enumerate(group):
                    try:
                        ready.append((index, future.result()))
                    except Exception as exc:
                        output[index] = {"error": f"{type(exc).__name__}: {exc}"}
                if ready:
                    prepared = [value for _, value in ready]
                    try:
                        predictions = self.engine.predict_prepared_batch(
                            prepared, self.question, self.sampling,
                            decision_batch_size=self.config.decision_batch_size,
                            guard=self.guard,
                        )
                        if len(predictions) != len(ready):
                            raise ValueError("Model returned a different number of predictions")
                        for (index, _), prediction in zip(ready, predictions):
                            output[index] = prediction
                    except Exception as exc:
                        logger.warning("GPU batch failed (%s); retrying individually", exc)
                        for index, value in ready:
                            try:
                                output[index] = self.engine.predict_prepared_batch(
                                    [value], self.question, self.sampling,
                                    decision_batch_size=1, guard=self.guard,
                                )[0]
                            except Exception as exc:
                                output[index] = {"error": f"{type(exc).__name__}: {exc}"}
                for (item, _), prediction in zip(group, output):
                    if prediction is None:
                        raise RuntimeError("Missing prediction after batch execution")
                    yield item, prediction
