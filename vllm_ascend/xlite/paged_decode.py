"""Fail-closed capability check for the optional native 310P backend."""


def validate_paged_decode_build(backend, build_info, runtime_type):
    if backend == "direct_atb":
        expected = {"soc": "Ascend310P3", "kernel_set": "llm_fp16", "abi": 1,
                    "cache_layout": "BSHD", "direct_decode_attention": True,
                    "native_decode_cache_layout": "NZ_5D",
                    "direct_atb_task_queue_independent": True,
                    "direct_atb_runtime_version": 5,
                    "direct_atb_operation_scope": "per_layer_batch",
                    "direct_atb_setup_cache": True,
                    "direct_atb_fused_rope_staging": True,
                    "direct_atb_mixed_batch_decode": True,
                    "direct_atb_batched_compact_scatter": True,
                    "direct_atb_plan_cache": "layer_batch",
                    "direct_atb_metadata_upload": "once_per_forward"}
        supported = tuple(build_info.get("decode_attention_backends", ()))
        if (any(build_info.get(key) != value for key, value in expected.items())
                or backend not in supported
                or not hasattr(runtime_type, "set_decode_attention_backend")):
            raise RuntimeError(
                f"Xlite build does not support direct_atb: {build_info}")
        return
    if backend == "native_atb":
        expected = {"soc": "Ascend310P3", "kernel_set": "llm_fp16", "abi": 1,
                    "cache_layout": "BSHD", "native_decode_attention": True,
                    "native_decode_cache_layout": "NZ_5D"}
        supported = tuple(build_info.get("decode_attention_backends", ()))
        if (any(build_info.get(key) != value for key, value in expected.items())
                or backend not in supported
                or not hasattr(runtime_type, "set_decode_attention_backend")):
            raise RuntimeError(
                f"Xlite build does not support native_atb: {build_info}")
        return
    if backend == "batched_aclnn":
        expected = {"soc": "Ascend310P3", "kernel_set": "llm_fp16", "abi": 1,
                    "cache_layout": "BSHD", "batched_decode_attention": True}
        supported = tuple(build_info.get("decode_attention_backends", ()))
        if (any(build_info.get(key) != value for key, value in expected.items())
                or backend not in supported
                or not hasattr(runtime_type, "set_decode_attention_backend")):
            raise RuntimeError(
                f"Xlite build does not support batched_aclnn: {build_info}")
        return
    if backend != "paged_310p":
        return
    expected = {"soc": "Ascend310P3", "kernel_set": "llm_fp16", "abi": 1,
                "cache_layout": "BSHD", "paged_decode_310p": True}
    if any(build_info.get(key) != value for key, value in expected.items()) or not hasattr(
        runtime_type, "set_decode_attention_backend"
    ):
        raise RuntimeError(f"Xlite build does not support paged_310p: {build_info}")
