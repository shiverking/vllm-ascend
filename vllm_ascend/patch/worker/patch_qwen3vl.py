import torch
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3 import Qwen3Attention
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLForConditionalGeneration,
)

from vllm_ascend import envs
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding
from vllm_ascend.utils import enable_custom_op, vllm_version_is

logger = init_logger(__name__)


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
    if isinstance(self.rotary_emb, AscendMRotaryEmbedding):
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        if (
            envs.VLLM_ASCEND_ENABLE_ASCENDC_MROPE
            and qkv.dtype == torch.bfloat16
            and (cos_sin_cache.device != qkv.device or cos_sin_cache.dtype != qkv.dtype)
        ):
            # Materialize the full cache once so the AscendC kernel can gather
            # positions internally without a per-forward cache conversion.
            self.rotary_emb.cos_sin_cache = cos_sin_cache.to(
                device=qkv.device, dtype=qkv.dtype
            )
            cos_sin_cache = self.rotary_emb.cos_sin_cache
        use_ascendc = (
            envs.VLLM_ASCEND_ENABLE_ASCENDC_MROPE
            and qkv.dtype == torch.bfloat16
            and positions.dtype == torch.int64
            and positions.ndim == 2
            and positions.shape[0] == 3
            and qkv.is_contiguous()
            and self.q_norm.weight.is_contiguous()
            and self.k_norm.weight.is_contiguous()
            and cos_sin_cache.is_contiguous()
            and positions.is_contiguous()
            and cos_sin_cache.device == qkv.device
            and cos_sin_cache.dtype == qkv.dtype
            and enable_custom_op()
            and hasattr(torch.ops._C_ascend, "npu_split_qkv_rmsnorm_mrope")
        )
        if use_ascendc:
            logger.info_once(
                "Using experimental AscendC split QKV + Q/K RMSNorm + MRoPE kernel."
            )
            q, k, v = torch.ops._C_ascend.npu_split_qkv_rmsnorm_mrope(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                cos_sin_cache,
                positions,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.q_norm.variance_epsilon,
                self.rotary_emb.mrope_section,
                self.rotary_emb.mrope_interleaved,
                self.rotary_emb.rotary_dim,
            )
        else:
            if envs.VLLM_ASCEND_ENABLE_ASCENDC_MROPE:
                logger.warning_once(
                    "Experimental AscendC MRoPE kernel is unavailable or the inputs "
                    "are unsupported; falling back to Triton."
                )
            cos_sin = cos_sin_cache[positions]
            if cos_sin.device != qkv.device:
                cos_sin = cos_sin.to(qkv.device)
            if cos_sin.dtype != qkv.dtype:
                cos_sin = cos_sin.to(qkv.dtype)
            q, k, v, _ = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
                qkv=qkv,
                q_weight=self.q_norm.weight,
                k_weight=self.k_norm.weight,
                cos_sin=cos_sin,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                eps=self.q_norm.variance_epsilon,
                mrope_section=self.rotary_emb.mrope_section,
                is_interleaved=self.rotary_emb.mrope_interleaved,
                rope_dim=self.rotary_emb.rotary_dim,
            )
    else:
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


Qwen3Attention.forward = forward_with_split_qkv_rmsnorm_mrope
Qwen3MoeAttention.forward = forward_with_split_qkv_rmsnorm_mrope
Qwen3VLForConditionalGeneration._get_deepstack_input_embeds = tensor_parallel_wrap(
    Qwen3VLForConditionalGeneration._get_deepstack_input_embeds
)

if not vllm_version_is("0.19.1"):
    # Only patch for latest main
    from vllm.model_executor.models.qwen3_vl import pos_embed_interpolate_native

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

    Qwen3_VisionTransformer.fast_pos_embed_interpolate = _fast_pos_embed_interpolate
