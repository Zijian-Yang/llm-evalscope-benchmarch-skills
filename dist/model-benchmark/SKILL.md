---
name: model-benchmark
version: 2.4.0
updated: 2026-04-19
publisher: 淸水
description: >-
  通用模型 API 性能压测技能。使用 EvalScope perf 对 OpenAI-compatible / DashScope 等 HTTP API
  大模型服务进行冒烟、并发梯度、定速 rate、稳定性、输入/输出长度矩阵测试，并生成 Markdown 报告。
  <!-- negative triggers: 本地非 API 模型压测、仅查询模型信息、通用性能优化咨询、不需要执行 benchmark 的讨论不触发本 Skill。 -->
  <!-- insufficiency: 仅支持 HTTP API 形式的模型服务；结果受网络、限流和负载波动影响；不提供官方认证、目标值验收或 P2 趋势/图表能力。 -->
---

# Model Benchmark

## 版本信息

- Skill 版本：2.4.0
- 更新日期：2026-04-19
- 发布人：淸水

这是一个 Skill，不是完整压测平台。优先用 `SKILL.md` 指导信息收集、参数选择和报告完整性；`scripts/model_benchmark.py` 是薄 launcher，用来减少重复命令拼接。

## 平台与环境

- 支持 macOS 与 Ubuntu/Debian Linux，需要 Python 3.10+。
- 默认 venv 为 `.venv-model-benchmark`，依赖为 `evalscope[perf]`。
- API key 只从环境变量或 `.model_benchmark.env` 读取，YAML 只保存变量名。

## 安全红线

- 不上传 API key、Authorization token、生产 prompt、隐私数据或响应原文。
- 日志、报告、命令摘要和 usage 统计不得出现明文密钥。
- 高成本场景（大并发、长稳定性、长度矩阵、定速压测）必须先提示成本、时长和失败风险。
- 本 Skill 不做目标值验收 / SLA pass-fail；报告只陈述观测结果。

## MCP 工具清单

本 Skill 不依赖 MCP 工具；默认通过本地 CLI / Python / Shell 调用 `scripts/model_benchmark.py` 与 EvalScope perf。

## 场景识别与工具选择

- 触发：API 模型压测、benchmark、QPS、吞吐、TTFT、TPOT、延迟、并发上限、定速 rate、稳定性、Token 计数、性能报告。
- 不触发：本地模型压测、纯模型信息查询、泛性能优化建议、目标验收、长期趋势分析、图表/BI 看板。
- 环境未知先 `doctor`；缺依赖用 `bootstrap`；需要配置用 `menu`；正式压测前先跑 `smoke`。

## Core Workflow

```bash
python3 scripts/model_benchmark.py doctor --config configs/model_benchmark.example.yaml
python3 scripts/model_benchmark.py bootstrap --config configs/model_benchmark.example.yaml
python3 scripts/model_benchmark.py menu --output configs/model_benchmark.local.yaml
export DASHSCOPE_API_KEY=...
python3 scripts/model_benchmark.py run --config configs/model_benchmark.local.yaml --scenario smoke
python3 scripts/model_benchmark.py run --config configs/model_benchmark.local.yaml --scenario all
python3 scripts/model_benchmark.py report --config configs/model_benchmark.local.yaml --results-dir outputs/model_benchmark/<run_id> --output outputs/model_benchmark/<run_id>/model_benchmark_report.md
```

## 必问信息

- 模型：模型名、API 协议、完整 endpoint、API key 环境变量名。
- 数据集：`simulated`、`openqa`、`line_by_line`、`random`；非模拟数据必须确认路径和格式。
- 场景：`smoke`、`gradient`、`rate`、`stability`、`length_matrix`，并说明每个场景的成本和时长影响。
- 高级控制：headers、connect/read/total timeout、warmup 请求数、cooldown 秒数。

## Token 计量方案

必须让用户明确选择 token 计量模式，而不是隐式假设：

- `prefer_api_usage`：默认推荐。优先用 API 返回的 usage；缺失时按 `on_missing_usage` 处理。
- `api_usage_only`：只认 API usage；适合要求供应商返回真实 token 的场景。
- `tokenizer_only`：完全用 tokenizer 估算；适合 API 不返回 usage 的服务。
- `dual_compare`：同时记录 usage 与 tokenizer 对比；报告展示 delta，用于检查服务端 usage 可信度。
- `disabled`：不统计 token、TPOT、token 吞吐；报告必须标记这些指标不可计。

usage 缺失策略：

- `fallback_tokenizer`：有 tokenizer 时回退估算。
- `mark_unavailable`：不失败，但报告标记 token 指标不可计。
- `fail`：直接失败，适合强制要求 usage 的验收前检查。

tokenizer 来源：

- `modelscope`、`huggingface`、`local_path`、`disabled`。`random` 和 `length_matrix` 必须提供 tokenizer。

## P1 压测控制

- `rate`：定速压测，按 `rates[]` 多档执行，适合验证限流、稳定吞吐和队列堆积。
- `warmup`：主场景前可跑少量预热请求，避免冷启动污染正式指标。
- `cooldown`：场景之间可等待释放资源，避免连续档位相互影响。
- `headers`：支持自定义请求头；不得在报告中输出敏感 header 明文。
- `timeout`：支持 connect/read/total timeout；报告必须写明配置。
- `error classification`：报告按 timeout、rate_limit、auth、server_5xx、client_4xx、connect、empty 做最佳努力分类。
- `retry/backoff`：当前底层 EvalScope perf 未提供统一 retry/backoff 参数；不要假装支持，只能在报告或总结中说明由服务端/网关策略控制。

## Report Requirements

Markdown 报告必须包含：

- 测试信息：模型、API、endpoint、数据集、EvalScope 实际 dataset/dataset_path、token 模式、timeout、warmup/cooldown、结果目录和免责声明。
- Token 计量说明：模式、usage 缺失策略、tokenizer 来源、dual compare 样本数和 delta。
- 结论摘要：QPS 峰值、最高成功率、最低平均 E2E，并说明哪些指标是运行级、哪些有分位数。
- 指标表：数据量与成功率、QPS 与吞吐、TTFT、TPOT、E2E、ITL、Token 统计。
- 错误摘要：失败请求数、错误分类、原始路径。
- 原始结果路径：每个场景/档位的结果目录。

## 终止与总结

- 完成指定 `doctor`、`bootstrap`、`menu`、`run` 或 `report` 目标后停止，并返回配置、结果和报告路径。
- 凭证、网络、endpoint、依赖、tokenizer 或数据集格式阻塞时必须真实失败并说明下一步。
- 总结不得包含密钥、原始 prompt 或响应正文。

## Troubleshooting

- `DASHSCOPE_API_KEY missing`: 设置环境变量或 `.model_benchmark.env`。
- `evalscope not installed`: 运行 `bootstrap`。
- `tokenizer missing`: 选择 `mark_unavailable`，或补充 tokenizer path / ModelScope ID。
- `endpoint unreachable`: 检查 URL 是否为完整 `/v1/chat/completions`。
