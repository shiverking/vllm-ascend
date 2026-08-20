# Qwen3-ASR-1.7B MTP 310P 验证手册

1. 确认运行时导入来自 `/vllm-workspace/vllm` 和
   `/vllm-workspace/vllm-ascend`，模型目录包含 `mtp_model.safetensors`。
2. 先用 `--enforce-eager --max-num-seqs 1` 启动并完成一个真实音频请求。
3. 删除 `--enforce-eager`，按K=1/2/3启动，保留ACLGraph capture/replay日志。
4. 分别运行无MTP和MTP服务，temperature=0逐token比较输出。
5. 从 `/metrics` 保存每位置接受token统计；压测batch=1/4/8/16。
6. 出现图模式问题时回退eager；出现多模态编译连续性错误时，仅作为诊断
   尝试 `TORCHDYNAMO_DISABLE=1`，不得把诊断结果冒充图模式通过。

真实checkpoint未通过HTTP音频请求前，只能标记为“框架适配完成”，不能标记
为“商用验证完成”。
