"""Batched VideoJev choice scoring over one caller-supplied question."""

from __future__ import annotations

import importlib
import time

import torch

from .types import ChoiceQuestion


@torch.inference_mode()
def score_batch(model, processor, records, visual_features, context: int,
                question: ChoiceQuestion):
    """Score one identical choice question over several videos in one text pass.

    Visual features are already encoded separately. Only valid choice tokens
    are read out; right padding cannot influence causal outputs before it.
    """
    if not records or len(records) != len(visual_features):
        raise ValueError("records and visual features must have equal nonzero lengths")
    if model.config.backbone_config.model_type not in ("qwen3_5", "qwen3_5_moe"):
        raise ValueError("batched decision scoring supports Qwen3.5 VideoJev backbones only")
    model_package = type(model).__module__.rsplit(".", 1)[0]
    prepare_groups = importlib.import_module(
        f"{model_package}.processing_videojev").prepare_query_groups
    native_question = question.native()
    groups = [next(prepare_groups(processor, record, [native_question], context))[0]
              for record in records]
    lengths = [int(group["input_ids"].shape[1]) for group in groups]
    width = (max(lengths) + 63) // 64 * 64
    device = model.get_input_embeddings().weight.device
    token_id = model.config.backbone_config.video_token_id
    pad_id = processor.tokenizer.pad_token_id or 0
    ids = torch.full((len(groups), width), pad_id, dtype=torch.long, device=device)
    types = torch.zeros_like(ids)
    mask = torch.zeros((len(groups), width), dtype=torch.long, device=device)
    for index, group in enumerate(groups):
        length = lengths[index]
        ids[index, :length] = group["input_ids"][0].to(device)
        types[index, :length] = group["mm_token_type_ids"][0].to(device)
        mask[index, :length] = 1

    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embeds = model.get_input_embeddings()(ids)
        positions = None
        for index, group in enumerate(groups):
            length = lengths[index]
            slots = ids[index, :length] == token_id
            features = visual_features[index]
            if int(slots.sum()) != len(features):
                raise ValueError("Video feature count does not match token slots")
            embeds[index, :length].masked_scatter_(
                slots[:, None].expand(length, embeds.shape[-1]),
                features.to(embeds.dtype),
            )
            grid = group["video_grid_thw"].to(device)
            if model.config.backbone_config.model_type == "qwen3_5_moe":
                sample_positions, _ = model.backbone.get_rope_index(
                    ids[index:index+1, :length],
                    types[index:index+1, :length], video_grid_thw=grid)
            else:
                sample_positions = model.backbone.compute_3d_position_ids(
                    input_ids=ids[index:index+1, :length],
                    inputs_embeds=embeds[index:index+1, :length],
                    video_grid_thw=grid,
                    attention_mask=None,
                    mm_token_type_ids=types[index:index+1, :length],
                )
            if positions is None:
                positions = torch.zeros((sample_positions.shape[0], len(groups), width),
                                        dtype=torch.long, device=device)
            positions[:, index:index+1, :length] = sample_positions
        hidden = model.backbone.language_model(
            inputs_embeds=embeds,
            position_ids=positions,
            attention_mask=mask,
            use_cache=False,
        ).last_hidden_state
    keys = torch.stack([hidden[index, group["pair_indices"][0].to(device)]
                        for index, group in enumerate(groups)])
    queries = torch.stack([hidden[index, group["query_indices"][0].to(device)]
                           for index, group in enumerate(groups)])
    with torch.autocast(device_type="cuda", enabled=False):
        logits = model.head(keys, queries)
        probabilities = logits.softmax(-1)
    torch.cuda.synchronize()
    return probabilities.float().cpu().tolist(), time.perf_counter() - started


@torch.inference_mode()
def score_packed(model, processor, records, visual_features, context: int,
                 question: ChoiceQuestion):
    """Pack dense 4B queries with variable-length FlashAttention."""
    if not records or len(records) != len(visual_features):
        raise ValueError("records and visual features must have equal nonzero lengths")
    if model.config.backbone_config.model_type != "qwen3_5":
        raise ValueError("packed Flash scoring is validated for dense VideoJev-4B only")
    started = time.perf_counter()
    model_package = type(model).__module__.rsplit(".", 1)[0]
    prepare_groups = importlib.import_module(
        f"{model_package}.processing_videojev").prepare_query_groups
    native_question = question.native()
    groups = [next(prepare_groups(processor, record, [native_question], context))[0]
              for record in records]
    lengths = [int(group["input_ids"].shape[1]) for group in groups]
    device = model.get_input_embeddings().weight.device
    ids = torch.cat([group["input_ids"] for group in groups], dim=1).to(device)
    token_id = model.config.backbone_config.video_token_id
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    total = offsets[-1]
    cu = torch.tensor(offsets, dtype=torch.int32, device=device)
    sequence_ids = torch.repeat_interleave(
        torch.arange(len(groups), dtype=torch.int32, device=device),
        torch.tensor(lengths, device=device),
    ).view(1, total)

    cpu_position_ids = torch.cat([
        model.backbone.get_rope_index(
            group["input_ids"], group["mm_token_type_ids"],
            video_grid_thw=group["video_grid_thw"])[0]
        for group in groups
    ], dim=-1)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        embeds = model.get_input_embeddings()(ids)
        for index, (group, features) in enumerate(zip(groups, visual_features)):
            left, right = offsets[index:index + 2]
            slots = ids[0, left:right] == token_id
            if int((group["input_ids"] == token_id).sum()) != len(features):
                raise ValueError("Video feature count does not match token slots")
            embeds[0, left:right].masked_scatter_(
                slots[:, None].expand(right - left, embeds.shape[-1]),
                features.to(embeds.dtype),
            )
        from .aten_flash import INTERFACE_NAME, ensure_registered
        ensure_registered()
        attention_config = model.backbone.language_model.config
        previous_attention = attention_config._attn_implementation
        attention_config._attn_implementation = INTERFACE_NAME
        try:
            hidden = model.backbone.language_model(
                inputs_embeds=embeds,
                position_ids=cpu_position_ids.to(device),
                attention_mask={"full_attention": None,
                                "linear_attention": None},
                use_cache=False,
                cu_seq_lens_q=cu,
                cu_seq_lens_k=cu,
                max_length_q=max(lengths),
                max_length_k=max(lengths),
                seq_idx=sequence_ids,
            ).last_hidden_state[0]
        finally:
            attention_config._attn_implementation = previous_attention
    keys = torch.stack([hidden[offsets[index] + group["pair_indices"][0].to(device)]
                        for index, group in enumerate(groups)])
    queries = torch.stack([hidden[offsets[index] + group["query_indices"][0].to(device)]
                           for index, group in enumerate(groups)])
    with torch.autocast(device_type="cuda", enabled=False):
        probabilities = model.head(keys, queries).softmax(-1)
    torch.cuda.synchronize()
    return probabilities.float().cpu().tolist(), time.perf_counter() - started
