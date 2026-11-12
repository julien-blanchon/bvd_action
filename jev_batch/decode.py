"""Frame sampling for local videos, independent of the annotation dataset."""

from __future__ import annotations

import math
from pathlib import Path

import av
import numpy as np
import torch

from .types import VideoFrames


def _grid(duration: float, fps: float, max_frames: int):
    requested = np.arange(math.ceil(duration * fps), dtype=np.float64) / fps
    requested = requested[requested < duration]
    if len(requested) > max_frames:
        effective_fps = max_frames / duration
        return np.arange(max_frames, dtype=np.float64) / effective_fps, effective_fps
    return requested, fps


def _read_sequential(path: Path, duration: float, fps: float, max_frames: int,
                     codec_threads: int) -> VideoFrames:
    requested, effective_fps = _grid(duration, fps, max_frames)
    frames = []
    previous = None
    previous_time = 0.0
    origin = None
    index = 0
    end_time = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = codec_threads
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"Missing video PTS: {path}")
            stamp = float(frame.pts * frame.time_base)
            if origin is None:
                origin = stamp
            now = stamp - origin
            if previous is not None and now < previous_time:
                if previous_time - now <= 1e-4:
                    continue
                raise ValueError(f"Nonmonotonic PTS: {path}")
            while index < len(requested) and requested[index] <= now:
                chosen = (previous if previous is not None and
                          abs(previous_time - requested[index]) <= abs(now - requested[index])
                          else frame)
                frames.append(chosen.to_ndarray(format="rgb24"))
                index += 1
            previous, previous_time = frame, now
            frame_duration = (float(frame.duration * frame.time_base) if frame.duration
                              else 1 / float(stream.average_rate or 30))
            end_time = now + frame_duration
    if previous is None:
        raise ValueError(f"Empty video: {path}")
    sampled_duration = min(duration, end_time)
    times = requested[requested < sampled_duration]
    while index < len(times):
        frames.append(previous.to_ndarray(format="rgb24"))
        index += 1
    if not len(times) or len(frames) < len(times):
        raise ValueError("Video does not cover a requested sample")
    if len(times) != len(requested):
        if sampled_duration >= duration:
            raise ValueError("Sampling grid extends beyond media duration")
        return _read_sequential(path, sampled_duration, fps, max_frames, codec_threads)
    pixels = torch.stack([torch.from_numpy(frame) for frame in frames[:len(times)]]).numpy()
    return VideoFrames(pixels, effective_fps, tuple(times), sampled_duration)


def _read_sparse(path: Path, duration: float, fps: float, max_frames: int,
                 codec_threads: int) -> VideoFrames:
    requested, effective_fps = _grid(duration, fps, max_frames)
    if not len(requested):
        raise ValueError("Empty sampling grid")
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = codec_threads
        first = next(container.decode(stream))
        if first.pts is None:
            raise ValueError("Missing first-frame PTS")
        origin = float(first.pts * first.time_base)
        time_base = float(stream.time_base)

        def seek(seconds: float):
            offset = max(int(first.pts), int((origin + seconds) / time_base))
            container.seek(offset, stream=stream, backward=True, any_frame=False)

        seek(max(0.0, duration - 5.0))
        last = None
        for frame in container.decode(stream):
            last = frame
        if last is None or last.pts is None:
            raise ValueError("Unable to read video tail")
        end_time = (float(last.pts * last.time_base) - origin +
                    (float(last.duration * last.time_base) if last.duration
                     else 1 / float(stream.average_rate or 30)))
        sampled_duration = min(duration, end_time)
        if np.any(requested >= sampled_duration):
            raise ValueError("Sparse sampling exceeds actual media duration")
        for target in requested:
            seek(float(target))
            previous = None
            previous_time = 0.0
            chosen = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError("Missing sampled-frame PTS")
                now = float(frame.pts * frame.time_base) - origin
                if previous is not None and now < previous_time - 1e-4:
                    raise ValueError("Nonmonotonic sampled PTS")
                if now >= target:
                    chosen = (previous if previous is not None and
                              abs(previous_time - target) <= abs(now - target)
                              else frame)
                    break
                previous, previous_time = frame, now
            if chosen is None:
                if previous is None:
                    raise ValueError("Seek found no sampled frame")
                chosen = previous
            frames.append(chosen.to_ndarray(format="rgb24"))
    pixels = torch.stack([torch.from_numpy(frame) for frame in frames]).numpy()
    return VideoFrames(pixels, effective_fps, tuple(requested), sampled_duration)


def decode_path(path: str | Path, fps: float, max_frames: int, *,
                codec_threads: int = 2, sparse_seek: bool = True) -> VideoFrames:
    """Decode the exact sample grid; long-video seeking falls back safely."""
    path = Path(path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = (float(stream.duration * stream.time_base)
                    if stream.duration is not None else container.duration / 1e6)
    if sparse_seek and duration >= 80:
        try:
            return _read_sparse(path, duration, fps, max_frames, codec_threads)
        except (EOFError, ValueError, RuntimeError, StopIteration):
            pass
    return _read_sequential(path, duration, fps, max_frames, codec_threads)
