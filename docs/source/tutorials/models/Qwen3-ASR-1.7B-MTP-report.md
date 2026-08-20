# Qwen3-ASR-1.7B MTP 适配报告

## 架构与不兼容点

Qwen3-ASR target是音频encoder加28层dense Qwen3 decoder。训练新增的每个
MTP层由双RMSNorm、2H到H投影和一层Qwen3 decoder组成。原始vLLM没有
`qwen3_asr_mtp` draft配置和权重映射；Ascend proposer只对部分MTP模型共享
LM head，造成不必要的词表矩阵副本。

## 修改说明

- vLLM提供原生 `Qwen3ASRMTP`、稳定权重命名和文本RoPE draft配置。
- vLLM-Ascend复用现有310P proposer、verification、KV rollback和ACLGraph，
  并让所有标准MTP模型共享target LM head。
- 音频只进入target prefill，draft只接收最终hidden state和shifted token。

## 功能状态

| 功能 | 当前状态 | 说明 |
|---|---|---|
| 架构/配置/权重映射 | 已实现 | 合成与CPU测试用于门禁 |
| MTP-3/MTP-5 | 已实现 | K不得超过训练深度 |
| 音频多模态 | 待真实权重验证 | 必须完成HTTP音频请求 |
| ACLGraph | 待310P验证 | 必须保留capture/replay证据 |
| Eager | 预期支持 | 首个真实权重隔离路径 |
| EP / flashcomm1 | N/A | dense模型 |

## 验证边界

Dummy或合成权重只能证明注册、shape、算子和API路径，不能证明权重映射正确、
接受率、WER或加速比。真实MTP checkpoint仍是发布硬门禁。模型理论上下文为
65,536，不支持技能基线中的128K；商用短音频先验证8K，随后16K和32K。
