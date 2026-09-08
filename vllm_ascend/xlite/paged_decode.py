"""Fail-closed capability check for the optional native 310P backend."""


def validate_paged_decode_build(backend, build_info, runtime_type):
    if backend != "paged_310p":
        return
    expected = {"soc": "Ascend310P3", "kernel_set": "llm_fp16", "abi": 1,
                "cache_layout": "BSHD", "paged_decode_310p": True}
    if any(build_info.get(key) != value for key, value in expected.items()) or not hasattr(
        runtime_type, "set_decode_attention_backend"
    ):
        raise RuntimeError(f"Xlite build does not support paged_310p: {build_info}")
