# Qwen3-ASR-1.7B MTP

## Introduction

本页说明如何部署由 ParaASR 式串行 MTP 训练导出的
`Qwen3-ASR-1.7B-MTP3/5`。MTP 只参与转写文本的 decode；音频编码仍由
Qwen3-ASR target 模型完成。本文默认单卡 Ascend 310P、BF16、TP=1。

## Supported Features

| 功能 | 状态 |
|---|---|
| 音频转写 | 支持，使用 `/v1/audio/transcriptions` |
| MTP-3 / MTP-5 | 支持，运行时 K 不得超过训练深度 |
| ACLGraph | 代码路径支持，真实310P验证待完成 |
| Eager | 支持，用于问题隔离 |
| EP / flashcomm1 | N/A，Qwen3-ASR-1.7B是dense模型 |

## Environment Preparation

必须同时使用包含 `Qwen3ASRMTP` 的 vLLM `mtp-test` 分支和本仓
`mtp-test` 分支，不要升级镜像内的 Transformers。

:::::{tab-set}

::::{tab-item} 310P

将仓库分别安装到 `/vllm-workspace/vllm` 和
`/vllm-workspace/vllm-ascend`。本文的MTP-5示例模型目录为
`/models/Qwen3-ASR-1.7B-MTP5-stage2-3384`。

::::

::::{tab-item} A2/A3

代码可复用通用Ascend MTP路径，但本项目的商用验收范围是310P；A2/A3
需要按对应官方镜像另行验证。

::::

:::::

## Deployment

在 `/workspace` 直接启动：

```bash
vllm serve /models/Qwen3-ASR-1.7B-MTP5-stage2-3384 \
  --served-model-name qwen3-asr-mtp5 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --port 8000
```

Eager隔离命令是在上述命令末尾增加 `--enforce-eager`。同一个MTP-5
checkpoint应依次测试K=3、4、5，只需修改`num_speculative_tokens`；最终采用
真实端到端耗时最短的K，不默认K=5最快。

导出的`config.json`必须包含`mtp_num_hidden_layers=5`和
`mtp_branch_position_mode`。当前训练配置使用`base`：各串行MTP层复用target
最后位置的RoPE坐标，但KV槽位和序列长度仍逐步前进。缺少该字段的旧导出包
按`shifted`兼容模式处理，应重新导出后再进行接受率验收。

## Functional Verification

```bash
curl -sf http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@/workspace/test.wav \
  -F model=qwen3-asr-mtp5 \
  -F language=en \
  -F temperature=0
```

服务启动不等于通过；转写请求必须返回HTTP 200及非空文本。真实权重下，
greedy结果必须与禁用MTP的target逐token一致。

## Accuracy Evaluation

推测解码不改变target验证结果，目标WER与无MTP基线相同。当前仓库记录的
LibriSpeech test-clean 500条基线WER为0.035；真实训练checkpoint尚未完成
310P门禁，因此该数字不是MTP实测结论。

## Performance

依次测试K=3/4/5及batch=1/4/8/16，记录接受长度、decode TPS、P50/P90
端到端耗时和峰值显存。商用目标为batch=1 decode TPS提升至少20%，短音频
P50端到端耗时降低至少10%。配置理论上限为65,536；首轮实用验证为
8K/16K/32K，128K不适用。
