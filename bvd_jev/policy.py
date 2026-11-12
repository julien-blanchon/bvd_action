"""BVD action question, review band, and output formatting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from jev_batch import ChoiceQuestion, ProbabilityGuard
from jev_batch.types import Prediction


ACTION_QUESTION = ChoiceQuestion(
    query=(
        "Does this video show substantial, meaningful, goal-directed physical action "
        "in the real world? Judge visible actions, not narration or stated intent. "
        "Count hands-on cooking, repairing, assembling, crafting, cleaning, building, "
        "gardening, or operating equipment. Do not count incidental gestures, talking, "
        "slides, screens, gaming, static scenes, or passive product display. "
        "Action may occupy a substantial part of a mixed video."
    ),
    options=(
        "ACTION: A person visibly performs a meaningful physical task with objects, tools, materials, or the environment.",
        "NON_ACTION: No substantial goal-directed physical task is visible.",
    ),
    id="action",
)


@dataclass(frozen=True)
class ActionPolicy:
    low: float = 0.35
    high: float = 0.65
    guard_margin: float = 0.05

    def __post_init__(self) -> None:
        if not 0 <= self.low < self.high <= 1:
            raise ValueError("Expected 0 <= low < high <= 1")

    @property
    def guard(self) -> ProbabilityGuard:
        return ProbabilityGuard(0, (self.low, self.high), self.guard_margin)

    def label(self, p_action: float) -> Literal["ACTION", "NON_ACTION", "UNCERTAIN"]:
        if p_action < self.low:
            return "NON_ACTION"
        if p_action > self.high:
            return "ACTION"
        return "UNCERTAIN"

    def format(self, row: Mapping[str, Any], prediction: Prediction) -> dict[str, Any]:
        from .sources import clip_key

        path = (row.get("path") or row.get("file") or
                f"{row.get('source_tar')}@{row.get('source_byte_offset')}")
        try:
            key = clip_key(row)
        except ValueError:
            key = None
        result = {
            "id": row.get("id", path),
            "path": str(path),
            "source": {key: row[key] for key in
                       ("source_tar", "source_member", "source_byte_offset", "bytes")
                       if key in row},
            "clip_key": key,
        }
        if "error" in prediction:
            result["error"] = prediction["error"]
            return result
        p_action, p_non_action = prediction["probabilities"]
        result.update({
            "label": self.label(p_action),
            "p_action": p_action,
            "p_non_action": p_non_action,
            "frames": prediction["frames"],
            "visual_tokens": prediction["visual_tokens"],
            "sampled_duration_s": prediction["sampled_duration_s"],
            "vision_batch_size": prediction["vision_batch_size"],
            "decision_batch_size": prediction["decision_batch_size"],
            "decision_mode": prediction["decision_mode"],
            "native_guard_fallback": prediction["native_guard_fallback"],
            "pre_guard_p_action": prediction["pre_guard_probabilities"][0],
            "seconds": prediction["seconds"],
        })
        return result
