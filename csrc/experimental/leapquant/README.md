# Ascend AIV LeapQuant 核心实现

显式启用的 Qwen3.5-9B 单卡实现。默认插件构建不编译此目录，默认推理不加载本库。
支持 BF16 模型、TP/PP/PCP/DCP=1，不叠加 MTP、权重量化、全注意力 KV 量化或缓存传输。

保留 W=16、R=4：GDN 状态由 INT8 残差、FP32 缩放、FP16 低秩因子和更新记录组成。
每 16 个 decode token 折叠一次；保留 FP16 溢出保护、FP32 回退、安全 Newton 除法、
FP32 取整及 UB 工作区隔离。包格式为 LQ04，内容 27744 字节，每头 serving 步长 32768。
权重及全注意力缓存保持原精度。预填充仍使用原有 GDN 计算，只在导入/导出状态时压缩。

```bash
cmake -S csrc/experimental/leapquant -B /path/to/leapquant-build \
  -DSOC_VERSION=Ascend910B3 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0
cmake --build /path/to/leapquant-build -j8
```

两个仓库切到配套的 `cloud-drive-core` 分支后，可按实际安装路径调整以下命令：

```bash
export PYTHONPATH=/path/to/vllm-core:/path/to/vllm-ascend-core:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=7
vllm serve /home/models/Qwen3.5-9B \
  --served-model-name Qwen3.5-9B --host 0.0.0.0 --port 1025 \
  --tensor-parallel-size 1 --max-model-len 16384 \
  --max-num-seqs 512 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.99 --enable-prefix-caching --enable-chunked-prefill \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,96,128,160,176,192,208,224,256,272,288,296,304,320,336,352,368,384,416,448,480,512]}' \
  --additional-config '{"fixed_prefix_cache":{"budget_fraction":0.02},"leapquant_config":{"enabled":true,"library":"/path/to/leapquant-build/lib/libleapquant_native.so","cores":40},"data_movement_optimizations":{"prefill_state_layout":true,"batched_prefix_copy":true,"greedy_no_fp32_cast":true}}'
```

CANN 环境、驱动和插件自定义算子包需要按其正常安装说明配置。
删除 `leapquant_config` 可关闭状态压缩。删除 `fixed_prefix_cache` 可关闭固定前缀接口。
三项搬运优化默认关闭，可分别启用：小批 prefill 状态布局融合、设备端批量前缀复制、
满足条件的 greedy 采样跳过 FP32 转换；不满足适用条件时回退原实现。

使用原生调度策略，无自定义吞吐调度、老化、队列扫描或 admission 参数。
固定前缀路径仅保留非对齐检查点和跳过文本计算后的视觉内存预算保护。
移除调度后不能直接引用旧实验的吞吐值作为本分支性能承诺。

固定前缀接口：`POST /v1/fixed-prefixes` 注册一个普通单图聊天请求并预热，
`GET /v1/fixed-prefixes/{prefix_id}` 查状态，`DELETE` 释放无活跃请求的前缀。
`POST /v1/fixed-prefix/completions` 传 `prefix_id`、`image_url` 和可选采样参数。
服务重启后必须重新注册。前缀块受到预算限制并被固定，后续可写尾块及 GDN 状态私有复制。

客户端见 [Direct Guided](../../../examples/online_serving/direct_guided/README.md)。
