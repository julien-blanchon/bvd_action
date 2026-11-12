"""GPU smoke check for the public video input and feature-reuse API."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from jev_batch import (BatchConfig, BatchRunner, ChoiceQuestion, Item, Sampling,
                       VideoFrames, VideoJevEngine)
from jev_batch.decode import decode_path
from bvd_jev.config import DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    engine = VideoJevEngine(args.model or DEFAULT_MODEL)
    sampling = Sampling()
    native_pixels, native_record = engine.prepare_video(
        engine.video_processor, args.video, fps=sampling.fps,
        max_frames=sampling.max_frames, context=sampling.context)
    decoded = decode_path(args.video, sampling.fps, sampling.max_frames)
    from_frames = engine.prepare(decoded, sampling)
    assert from_frames.record == native_record
    assert torch.equal(from_frames.pixels, native_pixels)

    tensor = torch.from_numpy(np.asarray(decoded.frames).copy())
    from_tensor = engine.prepare(VideoFrames(tensor, decoded.fps, decoded.times,
                                             decoded.duration), sampling)
    assert from_tensor.record == native_record
    assert torch.equal(from_tensor.pixels, native_pixels)
    if len(decoded.frames) == sampling.max_frames:
        direct = engine.prepare(tensor, sampling)
        assert direct.record["frames"] == sampling.max_frames

    question = ChoiceQuestion("What is most visible?", (
        "A person handling an object", "A person speaking", "A static scene"))
    encoded = engine.encode([from_frames])[0]
    first = engine.predict_prepared_batch([from_frames], question, sampling)[0]
    reused = engine.predict_prepared_batch([encoded], question, sampling)[0]
    assert len(first["probabilities"]) == 3
    assert max(abs(a - b) for a, b in zip(first["probabilities"],
                                          reused["probabilities"])) < 1e-6
    item, batched = next(BatchRunner(engine, question, sampling=sampling).run(
        [Item("frames", decoded)]))
    assert item.id == "frames" and len(batched["probabilities"]) == 3
    multi = list(BatchRunner(engine, question, sampling=sampling,
                             config=BatchConfig(vision_batch_size=2,
                                                decision_batch_size=2)).run(
        [Item("first", decoded), Item("second", decoded)]))
    assert len(multi) == 2
    assert all(len(result["probabilities"]) == 3 for _, result in multi)
    print("MP4, NumPy frames, Torch frames, prepared patches and reused features: OK")


if __name__ == "__main__":
    main()
