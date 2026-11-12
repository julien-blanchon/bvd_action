"""Fused, inference-only Qwen3.5 vision Q/K rotary embedding on CUDA."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_installed = False


@triton.jit
def _rope_qk_kernel(Q, K, COS, SIN, OQ, OK,
                    SQ0: tl.constexpr, SQ1: tl.constexpr, SQ2: tl.constexpr,
                    SK0: tl.constexpr, SK1: tl.constexpr, SK2: tl.constexpr,
                    H: tl.constexpr, D: tl.constexpr, TOTAL,
                    BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < TOTAL
    d = index % D
    head = (index // D) % H
    seq = index // (D * H)
    partner = tl.where(d < D // 2, d + D // 2, d - D // 2)
    sign = tl.where(d < D // 2, -1.0, 1.0)
    q = tl.load(Q + seq * SQ0 + head * SQ1 + d * SQ2, valid, other=0).to(tl.float32)
    k = tl.load(K + seq * SK0 + head * SK1 + d * SK2, valid, other=0).to(tl.float32)
    q_rot = tl.load(Q + seq * SQ0 + head * SQ1 + partner * SQ2, valid, other=0).to(tl.float32)
    k_rot = tl.load(K + seq * SK0 + head * SK1 + partner * SK2, valid, other=0).to(tl.float32)
    cos = tl.load(COS + seq * D + d, valid, other=0).to(tl.float32)
    sin = tl.load(SIN + seq * D + d, valid, other=0).to(tl.float32)
    tl.store(OQ + index, q * cos + sign * q_rot * sin, valid)
    tl.store(OK + index, k * cos + sign * k_rot * sin, valid)


def _fused_rope(q, k, cos, sin, original):
    if (not q.is_cuda or not k.is_cuda or not cos.is_cuda or not sin.is_cuda
            or q.device != k.device or q.device != cos.device or q.device != sin.device
            or q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16
            or q.shape != k.shape or q.ndim != 3 or q.shape[-1] != 64
            or cos.shape != (q.shape[0], 64) or sin.shape != cos.shape
            or cos.stride() != (64, 1) or sin.stride() != (64, 1)):
        return original(q, k, cos, sin)
    seq, heads, dim = q.shape
    out_q = torch.empty((seq, heads, dim), dtype=q.dtype, device=q.device)
    out_k = torch.empty((seq, heads, dim), dtype=k.dtype, device=k.device)
    total = seq * heads * dim
    _rope_qk_kernel[(triton.cdiv(total, 256),)](
        q, k, cos, sin, out_q, out_k,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        heads, dim, total, BLOCK=256, num_warps=4, enable_fp_fusion=False,
    )
    return out_q, out_k


def install_fused_vision_rope() -> None:
    """Install once for a single-model VideoJev-4B inference process."""
    global _installed
    if _installed:
        return
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

    original = qwen.apply_rotary_pos_emb_vision

    def fused(q, k, cos, sin):
        return _fused_rope(q, k, cos, sin, original)

    qwen.apply_rotary_pos_emb_vision = fused
    _installed = True
