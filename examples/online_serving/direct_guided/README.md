# Direct Guided 核心客户端

独立处理一张图片或一个目录，无需 cloud-drive-search、图片数据集或测试工具。
需要配套的 vLLM `cloud-drive-core` 分支，Python 依赖为 httpx、Pillow、regex；
输入 HEIC/HEIF 时另需 pillow-heif。请在 vllm-ascend 仓库根目录运行。

```bash
python -m examples.online_serving.direct_guided /path/to/images \
  --base-url http://127.0.0.1:1025 --model Qwen3.5-9B \
  --concurrency 8 --recovery-concurrency 4 \
  --output direct-guided-output/results.jsonl \
  --state-dir direct-guided-output/state
```

客户端启动时注册并预热主提示词，后续主请求仅发送前缀 ID、图片和采样参数。
服务端复用固定文本的注意力 KV 和 GDN 递归/卷积状态，继续生成前私有复制可写部分。
图片仍参与每个请求的模型计算；图片之间不共享递归状态。
保留原有图片处理顺序：EXIF 校正、最长边 4096、JPEG92，之后按 factor=32、
4–2048 视觉 token 预算缩放并输出 JPEG95。未启用 token 裁剪。

六套有效纯文本提示词固定在 `prompts/`，主请求输出上限 4096，完整恢复 8192，
四个分组各 2048。请求没有 `response_format`、`structured_outputs` 或工具约束；
只在生成后解析 SUMMARY、TAGS、DETAILED，校验摘要、详细描述非空及 3–8 个标签。
JSONL 是客户端落盘格式，不是模型生成格式。

只接受 `finish_reason=stop`。失败依次进入一次完整恢复、四组独立恢复（每组至多两次），
每次都看原图且不拼接前次长输出。恢复走普通文本接口，并先检查上下文预算。
单任务最多 10 次已完成尝试；进程中断的在途调用可能在重启后重放。
任务锁防止同一任务并行写入，检查点采用 fsync 与原子替换。
重跑相同输入和 state-dir 会复用已完成步骤；预算耗尽标为 `needs_review`，不冒充成功。
输出文件采用追加方式，恢复运行可能再次追加已成功图片的一行。

服务启动、关闭配置见 [LeapQuant 核心说明](../../../csrc/experimental/leapquant/README.md)。
并发参数控制客户端请求数；默认 320 仅是容量配置，不代表已证明默认调度下的最优吞吐。
