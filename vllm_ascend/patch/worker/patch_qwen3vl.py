import sys

import torch
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.models.qwen3 import Qwen3Attention
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention

from vllm_ascend import envs
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.utils import enable_custom_op, is_310p

if not is_310p():
    from vllm.model_executor.models.qwen3_vl import (
        Qwen3_VisionTransformer,
        Qwen3VLForConditionalGeneration,
        pos_embed_interpolate_native,
    )

logger = init_logger(__name__)
ASCENDC_MROPE_REQUESTED = envs.VLLM_ASCEND_ENABLE_ASCENDC_MROPE


def log_runtime_qwen3_attention(model: torch.nn.Module) -> None:
    logger.warning(
        "Inspecting loaded model attention: model=%s.%s, AscendC MRoPE requested=%s.",
        type(model).__module__,
        type(model).__name__,
        ASCENDC_MROPE_REQUESTED,
    )
    candidates = []
    for name, module in model.named_modules():
        if not name.endswith("self_attn"):
            continue
        candidates.append(f"{name}={type(module).__module__}.{type(module).__name__}")
        if "language_model.model.layers" not in name or not all(
            hasattr(module, attr) for attr in ("qkv_proj", "q_norm", "k_norm", "rotary_emb")
        ):
            continue
        forward = module.forward
        logger.warning(
            "Qwen3-ASR runtime attention: name=%s, type=%s.%s, forward=%s.%s.",
            name,
            type(module).__module__,
            type(module).__name__,
            forward.__module__,
            forward.__name__,
        )
        return
    logger.warning(
        "Qwen3 runtime attention was not found; self_attn candidates=%s.",
        candidates[:8],
    )


def tensor_parallel_wrap(func):
    def wrap(*args, **kwargs):
        deepstack_input_embeds = func(*args, **kwargs)
        if deepstack_input_embeds is None:
            return deepstack_input_embeds
        try:
            flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled
        except (AssertionError, AttributeError, KeyError):
            flash_comm_v1_enabled = False
        if flash_comm_v1_enabled:
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
            deepstack_input_embeds.tensors = {
                k: v.chunk(tp_size)[tp_rank] for k, v in deepstack_input_embeds.tensors.items()
            }
        return deepstack_input_embeds

    return wrap


def forward_with_split_qkv_rmsnorm_mrope(self, positions: torch.Tensor, hidden_states: torch.Tensor):
    qkv, _ = self.qkv_proj(hidden_states)
    ascendc_requested = ASCENDC_MROPE_REQUESTED
    debug_layer = getattr(self, "_ascendc_mrope_layer_index", -1) == 0
    if not torch.compiler.is_compiling():
        if debug_layer and not getattr(self, "_ascendc_mrope_eager_dispatch_printed", False):
            print(
                "[ASCENDC_MROPE_DEBUG] eager patched attention forward reached",
                file=sys.stderr,
                flush=True,
            )
            self._ascendc_mrope_eager_dispatch_printed = True
        logger.warning_once(
            "Patched Qwen3 attention forward reached; AscendC MRoPE requested=%s.",
            ascendc_requested,
        )
    is_mrope = isinstance(self.rotary_emb, MRotaryEmbedding) or all(
        hasattr(self.rotary_emb, attr) for attr in ("mrope_section", "mrope_interleaved", "cos_sin_cache")
    )
    if is_mrope:
        cache = self.rotary_emb.cos_sin_cache
        dtype_supported = qkv.dtype == torch.float16
        custom_op_enabled = enable_custom_op() if ascendc_requested else False
        op_registered = custom_op_enabled and hasattr(torch.ops._C_ascend, "npu_split_qkv_rmsnorm_mrope")
        use_ascendc = (
            ascendc_requested
            and dtype_supported
            and positions.dtype == torch.int64
            and positions.ndim == 2
            and positions.shape[0] == 3
            and all(t.is_contiguous() for t in (qkv, self.q_norm.weight, self.k_norm.weight, cache))
            and positions.stride(1) == 1
            and positions.stride(0) >= positions.shape[1]
            and cache.device == qkv.device
            and cache.dtype == qkv.dtype
            and op_registered
        )
        if (
            debug_layer
            and not torch.compiler.is_compiling()
            and not getattr(self, "_ascendc_mrope_eager_decision_printed", False)
        ):
            print(
                "[ASCENDC_MROPE_DEBUG] "
                f"use_ascendc={use_ascendc}, requested={ascendc_requested}, "
                f"qkv={qkv.dtype}/{qkv.device}/contiguous={qkv.is_contiguous()}, "
                f"positions={positions.dtype}/{tuple(positions.shape)}/contiguous={positions.is_contiguous()}, "
                f"q_weight_contiguous={self.q_norm.weight.is_contiguous()}, "
                f"k_weight_contiguous={self.k_norm.weight.is_contiguous()}, "
                f"cache={cache.dtype}/{cache.device}/contiguous={cache.is_contiguous()}, "
                f"custom_op_enabled={custom_op_enabled}, op_registered={op_registered}",
                file=sys.stderr,
                flush=True,
            )
            self._ascendc_mrope_eager_decision_printed = True
        if use_ascendc:
            q, k, v = torch.ops._C_ascend.npu_split_qkv_rmsnorm_mrope(
                qkv, self.q_norm.weight, self.k_norm.weight, cache, positions,
                self.num_heads, self.num_kv_heads, self.head_dim,
                self.q_norm.variance_epsilon, self.rotary_emb.mrope_section,
                self.rotary_emb.mrope_interleaved, self.rotary_emb.rotary_dim,
            )
            if not torch.compiler.is_compiling():
                if debug_layer and not getattr(self, "_ascendc_mrope_eager_kernel_printed", False):
                    print(
                        "[ASCENDC_MROPE_DEBUG] eager AscendC kernel returned successfully",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._ascendc_mrope_eager_kernel_printed = True
                logger.warning_once("Executed experimental AscendC split QKV + RMSNorm + MRoPE kernel.")
        elif is_310p():
            if ascendc_requested and not torch.compiler.is_compiling():
                reasons = []
                if not dtype_supported:
                    reasons.append(f"qkv dtype is {qkv.dtype}, expected torch.float16")
                if positions.dtype != torch.int64 or positions.ndim != 2 or positions.shape[0] != 3:
                    reasons.append(
                        f"positions is dtype={positions.dtype}, shape={tuple(positions.shape)}, expected int64 [3, T]"
                    )
                if not all(t.is_contiguous() for t in (qkv, self.q_norm.weight, self.k_norm.weight, cache)):
                    reasons.append("qkv, weights, or cache is not contiguous")
                if positions.stride(1) != 1 or positions.stride(0) < positions.shape[1]:
                    reasons.append("positions token dimension is not contiguous or rows overlap")
                if cache.device != qkv.device or cache.dtype != qkv.dtype:
                    reasons.append(
                        f"cache is {cache.device}/{cache.dtype}, qkv is {qkv.device}/{qkv.dtype}"
                    )
                if not custom_op_enabled:
                    reasons.append("custom op extension is not enabled")
                elif not op_registered:
                    reasons.append("npu_split_qkv_rmsnorm_mrope is not registered")
                logger.warning_once("AscendC MRoPE fallback reason: %s", "; ".join(reasons) or "unknown")
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q = self.q_norm(q.view(*q.shape[:-1], self.num_heads, self.head_dim)).view(q.shape)
            k = self.k_norm(k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)).view(k.shape)
            q, k = self.rotary_emb(positions, q, k)
        else:
            cos_sin = cache[positions].to(device=qkv.device, dtype=qkv.dtype)
            q, k, v, _ = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
                qkv=qkv, q_weight=self.q_norm.weight, k_weight=self.k_norm.weight,
                cos_sin=cos_sin, num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads, head_size=self.head_dim,
                eps=self.q_norm.variance_epsilon, mrope_section=self.rotary_emb.mrope_section,
                is_interleaved=self.rotary_emb.mrope_interleaved, rope_dim=self.rotary_emb.rotary_dim,
            )
    else:
        if ascendc_requested and not torch.compiler.is_compiling():
            logger.warning_once(
                "AscendC MRoPE fallback reason: rotary embedding type is %s, expected MRotaryEmbedding",
                type(self.rotary_emb).__name__,
            )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


def patch_runtime_qwen3_attention(model: torch.nn.Module) -> None:
    patched = 0
    for name, module in model.named_modules():
        if "language_model.model.layers" not in name or not name.endswith("self_attn"):
            continue
        if not all(
            hasattr(module, attr) for attr in ("qkv_proj", "q_norm", "k_norm", "rotary_emb")
        ):
            continue
        cache = module.rotary_emb.cos_sin_cache
        weight = module.q_norm.weight
        if cache.device != weight.device or cache.dtype != weight.dtype:
            module.rotary_emb.cos_sin_cache = cache.to(
                device=weight.device, dtype=weight.dtype
            )
        module.forward = forward_with_split_qkv_rmsnorm_mrope.__get__(
            module, type(module)
        )
        module._ascendc_mrope_layer_index = patched
        patched += 1
    logger.warning("Bound experimental AscendC MRoPE forward to %d runtime attention instances.", patched)


Qwen3Attention.forward = forward_with_split_qkv_rmsnorm_mrope
Qwen3MoeAttention.forward = forward_with_split_qkv_rmsnorm_mrope

if is_310p() or ASCENDC_MROPE_REQUESTED:
    logger.warning_once(
        "Installed Qwen3 attention patch; is_310p=%s, AscendC MRoPE requested=%s.",
        is_310p(),
        ASCENDC_MROPE_REQUESTED,
    )
if not is_310p():
    Qwen3VLForConditionalGeneration._get_deepstack_input_embeds = tensor_parallel_wrap(
        Qwen3VLForConditionalGeneration._get_deepstack_input_embeds
    )


def _fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
    outputs = []
    for t, h, w in grid_thw:
        outputs.append(
            pos_embed_interpolate_native(
                self.pos_embed.weight,
                t,
                h,
                w,
                self.num_grid_per_side,
                self.spatial_merge_size,
                self.dtype,
            )
        )
    return torch.cat(outputs, dim=0)


if not is_310p():
    Qwen3_VisionTransformer.fast_pos_embed_interpolate = _fast_pos_embed_interpolate


def patch_qwen3_vl_moe_pp_layer_range():
    try:
        from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM
    except Exception:
        return

    if not hasattr(Qwen3MoeLLMForCausalLM, "start_layer"):
        Qwen3MoeLLMForCausalLM.start_layer = property(lambda self: self.model.start_layer)

    if not hasattr(Qwen3MoeLLMForCausalLM, "end_layer"):
        Qwen3MoeLLMForCausalLM.end_layer = property(lambda self: self.model.end_layer)


if not is_310p():
    patch_qwen3_vl_moe_pp_layer_range()
