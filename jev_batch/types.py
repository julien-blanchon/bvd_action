"""Dataset-independent inputs and settings for choice inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any, TypedDict


class VideoRecord(TypedDict):
    """VideoJev timing and visual-token metadata shared by both model stages."""

    frames: int
    fps: float
    grid_thw: list[int]
    tokens: int
    score_times: list[float]
    score_bounds: list[list[float]]
    sampled_duration: float


@dataclass(frozen=True)
class ChoiceQuestion:
    query: str
    options: tuple[str, ...]
    id: str = "choice"

    def __post_init__(self) -> None:
        if not self.query.strip() or len(self.options) < 2 or any(not x.strip() for x in self.options):
            raise ValueError("A choice question needs text and at least two nonempty options")

    def native(self) -> dict[str, object]:
        return {"id": self.id, "task": "choice", "query": self.query,
                "options": list(self.options)}


@dataclass(frozen=True)
class Sampling:
    fps: float = 1.0
    max_frames: int = 16
    context: int = 4096

    def __post_init__(self) -> None:
        if not math.isfinite(self.fps) or self.fps <= 0 or self.max_frames < 2 or self.context < 512:
            raise ValueError("fps, max_frames and context must be positive")


@dataclass(frozen=True)
class ProbabilityGuard:
    """Re-score a chosen option near caller-supplied decision boundaries."""

    option_index: int
    thresholds: tuple[float, ...]
    margin: float = 0.05

    def __post_init__(self) -> None:
        if self.option_index < 0 or not self.thresholds or not 0 < self.margin < 0.5:
            raise ValueError("Invalid probability guard")
        if any(not 0 <= boundary <= 1 for boundary in self.thresholds):
            raise ValueError("Guard thresholds must be probabilities")

    def matches(self, probabilities: Sequence[float]) -> bool:
        value = float(probabilities[self.option_index])
        return min(abs(value - boundary) for boundary in self.thresholds) <= self.margin


@dataclass(frozen=True)
class VideoFrames:
    """Already-sampled RGB frames, [T,H,W,3] or [T,3,H,W].

    ``fps`` is the effective sample rate. ``times`` may supply the exact
    timestamp of each frame; otherwise a uniform grid is used. The frames
    are not sampled or decoded a second time.
    """

    frames: Any
    fps: float
    times: tuple[float, ...] | None = None
    duration: float | None = None


@dataclass(frozen=True)
class PreparedVideo:
    """VideoJev processor output, suitable for direct GPU vision encoding."""

    pixels: Any
    record: VideoRecord
    prepare_seconds: float = 0.0


@dataclass(frozen=True)
class EncodedVideo:
    """Vision features plus the matching native video record."""

    features: Any
    record: VideoRecord


VideoInput = str | Path | VideoFrames | PreparedVideo | EncodedVideo
Prediction = dict[str, Any]


@dataclass(frozen=True)
class Item:
    id: str
    source: Any


@dataclass(frozen=True)
class BatchConfig:
    vision_batch_size: int = 8
    decision_batch_size: int = 8
    visual_token_budget: int = 8192
    decode_workers: int = 2
    prefetch: int = 16

    def __post_init__(self) -> None:
        if (min(self.vision_batch_size, self.decision_batch_size,
                self.visual_token_budget, self.decode_workers, self.prefetch) < 1
                or self.decision_batch_size > self.vision_batch_size):
            raise ValueError("Invalid batch sizes, token budget or worker count")
