"""Prepare already-sampled RGB frames with VideoJev's native video geometry."""

from __future__ import annotations

import math

import numpy as np
import torch
from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
from transformers.video_utils import VideoMetadata

from .types import PreparedVideo, VideoFrames


def prepare_frames(processor, video: VideoFrames, context: int, pair_timeline) -> PreparedVideo:
    frames = video.frames
    if isinstance(frames, torch.Tensor):
        if frames.ndim != 4 or frames.dtype != torch.uint8:
            raise ValueError("VideoFrames tensor must be a 4D uint8 RGB tensor")
        if frames.shape[-1] != 3:
            if frames.shape[1] != 3:
                raise ValueError("VideoFrames tensor must have three RGB channels")
            frames = frames.permute(0, 2, 3, 1).contiguous()
        if frames.is_cuda:
            frames = frames.cpu()
    elif isinstance(frames, np.ndarray):
        if frames.ndim != 4 or frames.dtype != np.uint8:
            raise ValueError("VideoFrames array must be a 4D uint8 RGB array")
        if frames.shape[-1] != 3:
            if frames.shape[1] != 3:
                raise ValueError("VideoFrames array must have three RGB channels")
            frames = np.transpose(frames, (0, 2, 3, 1))
    else:
        raise TypeError("VideoFrames.frames must be a Torch tensor or NumPy array")
    n, height, width, _ = frames.shape
    if n < 1 or video.fps <= 0:
        raise ValueError("VideoFrames needs at least one frame and a positive sample rate")
    times = np.asarray(video.times if video.times is not None
                       else np.arange(n, dtype=np.float64) / video.fps, dtype=np.float64)
    if times.shape != (n,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Frame times must be strictly increasing timestamps")
    duration = float(video.duration if video.duration is not None else n / video.fps)
    if duration <= times[-1]:
        raise ValueError("Video duration must extend beyond the final frame timestamp")

    vp = processor.video_processor
    budget = context * 4 // 5

    def geometry(cap):
        h, w = smart_resize(
            n, height, width, temporal_factor=vp.temporal_patch_size,
            factor=vp.patch_size * vp.merge_size,
            min_pixels=vp.size["shortest_edge"], max_pixels=n * cap,
        )
        return h, w, (n + 1) // 2 * (h // 32) * (w // 32)

    low, high = max(1024, math.ceil(vp.size["shortest_edge"] / n)), 640 * 640
    if geometry(low)[2] > budget:
        raise ValueError("Complete video cannot fit the context at minimum native resolution")
    while low < high:
        middle = (low + high + 1) // 2
        if geometry(middle)[2] <= budget:
            low = middle
        else:
            high = middle - 1
    meta = VideoMetadata(total_num_frames=n, fps=video.fps, frames_indices=list(range(n)))
    native = vp(videos=[frames], video_metadata=[meta], do_sample_frames=False,
                size=dict(shortest_edge=vp.size["shortest_edge"], longest_edge=n * low),
                return_tensors="pt")
    grid = native["video_grid_thw"][0].tolist()
    score_times, bounds = pair_timeline(times, duration)
    record = dict(frames=n, fps=video.fps, grid_thw=grid,
                  tokens=int(np.prod(grid)) // 4,
                  score_times=score_times.tolist(), score_bounds=bounds.tolist(),
                  sampled_duration=duration)
    return PreparedVideo(native["pixel_values_videos"], record)
