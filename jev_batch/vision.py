"""Exact batched VideoJev vision encoding with cached CPU geometry."""

from __future__ import annotations

from functools import lru_cache

import torch

from .types import PreparedVideo


@lru_cache(maxsize=64)
def _geometry(grid, side, merge):
    from transformers.vision_utils import (
        get_vision_bilinear_indices_and_weights, get_vision_position_ids,
    )

    grid_cpu = torch.tensor([grid])
    indices, weights = get_vision_bilinear_indices_and_weights(grid_cpu, side, merge)
    positions = get_vision_position_ids(grid_cpu, merge)
    return indices, weights, positions


@torch.inference_mode()
def encode_many_videos(model, prepared: list[PreparedVideo], *, cached_metadata: bool):
    if not prepared:
        return []
    if len(prepared) == 1 or any(item.record["grid_thw"][0] > 16 for item in prepared):
        return [model.encode_video(
            item.pixels, torch.tensor([item.record["grid_thw"]]),
        ) for item in prepared]
    visual = model.backbone.visual
    device = next(visual.parameters()).device
    grids = [item.record["grid_thw"] for item in prepared]
    sizes = [int(t * h * w // 4) for t, h, w in grids]
    pixels = torch.cat([item.pixels for item in prepared], dim=0).to(device)
    grid_cpu = torch.tensor(grids)
    grid = grid_cpu.to(device)
    kwargs = {}
    if cached_metadata:
        from transformers.vision_utils import get_vision_cu_seqlens

        parts = [_geometry(tuple(row), visual.num_grid_per_side,
                           visual.config.spatial_merge_size) for row in grids]
        indices = torch.cat([part[0] for part in parts], dim=1)
        weights = torch.cat([part[1] for part in parts], dim=1)
        positions = torch.cat([part[2] for part in parts], dim=0)
        cu_seqlens = get_vision_cu_seqlens(grid_cpu)
        kwargs = {
            "bilinear_indices": indices.to(device),
            "bilinear_weights": weights.to(device),
            "position_ids": positions.to(device),
            # The SDPA vision implementation calls .tolist() every block.
            # CPU offsets avoid repeated device synchronizations exactly.
            "cu_seqlens": (cu_seqlens if visual.config._attn_implementation == "sdpa"
                            else cu_seqlens.to(device)),
        }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        features = visual(pixels, grid_thw=grid, return_dict=True,
                          **kwargs).pooler_output
    if len(features) != sum(sizes):
        raise ValueError("Batched vision output does not match video grids")
    return list(features.split(sizes))
