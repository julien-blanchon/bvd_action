"""Packed causal FlashAttention using PyTorch's built-in variable-length op.

This path uses the actual per-video cumulative sequence lengths. It requires
no flash-attn extension and avoids a quadratic block-causal attention mask.
"""

from __future__ import annotations

import torch


INTERFACE_NAME = "jev_aten_flash_varlen"


def packed_flash_attention(module, query, key, value, attention_mask=None,
                           dropout=0.0, scaling=None, **kwargs):
    if attention_mask is not None:
        raise ValueError("packed FlashAttention requires no explicit mask")
    if dropout != 0.0:
        raise ValueError("packed FlashAttention supports inference only")
    cu_q = kwargs.get("cu_seq_lens_q")
    cu_k = kwargs.get("cu_seq_lens_k")
    max_q = kwargs.get("max_length_q")
    max_k = kwargs.get("max_length_k")
    if (cu_q is None or cu_k is None or max_q is None or max_k is None
            or query.ndim != 4 or query.shape[0] != 1
            or key.ndim != 4 or key.shape[0] != 1
            or value.ndim != 4 or value.shape[0] != 1
            or not query.is_cuda or cu_q.dtype != torch.int32 or cu_k.dtype != torch.int32):
        raise ValueError("packed FlashAttention needs a single packed CUDA batch and int32 offsets")
    if kwargs.get("is_causal", True) is False:
        raise ValueError("packed FlashAttention is causal")
    q = query[0].transpose(0, 1).contiguous()
    k = key[0].transpose(0, 1).contiguous()
    v = value[0].transpose(0, 1).contiguous()
    output = torch.ops.aten._flash_attention_forward(
        q, k, v, cu_q, cu_k, int(max_q), int(max_k), 0.0, True, False,
        scale=scaling,
    )[0]
    return output.unsqueeze(0), None


def ensure_registered():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if INTERFACE_NAME not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register(INTERFACE_NAME, packed_flash_attention)
