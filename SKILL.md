---
name: model-benchmark
description: 通用模型 API 性能压测技能。使用 EvalScope perf 对 OpenAI-compatible / DashScope 等大模型服务进行 smoke test、并发梯度压测、SLA 自动调优、稳定性测试、输入/输出长度影响分析、usage/tokenizer 计量容错和详细 Markdown 报告生成。触发场景包括模型压测、benchmark、QPS、吞吐、TTFT、TPOT、延迟、并发上限、SLA、稳定性、性能报告、qwen3.6-plus 冒烟测试。
---

# Model Benchmark

使用这个 skill 时，把根目录的 `scripts/model_benchmark.py` 作为主入口。它负责环境检测、venv 安装、配置向导、EvalScope perf 执行、usage 缺失容错和报告生成；旧脚本 `benchmark.sh`、`stability_test.sh`、`sla_autotune.py`、`generate_report.py`、`convert_dataset.py` 只是兼容包装。

官方 EvalScope skill 位于 `evalscope/skills/evalscope`。遇到能力评测、benchmark 发现、结果可视化、EvalScope 参数细节时，优先读取并复用官方 skill，尤其是 `perf-reference.md`；不要在本 skill 中重复实现 `evalscope eval`、`benchmark-info` 或 `evalscope app`。

## Core Workflow

1. 先运行环境检测：
   ```bash
   python3 scripts/model_benchmark.py doctor --config configs/model_benchmark.example.yaml
   ```

2. 如果 venv 或 EvalScope 不存在，运行 bootstrap：
   ```bash
   python3 scripts/model_benchmark.py bootstrap --config configs/model_benchmark.example.yaml
   ```
   默认使用项目内 `.venv-model-benchmark`，国内 PyPI 镜像，优先从本仓库 `evalscope` 源码安装 `evalscope[perf]`。macOS 只使用现有 Python 或提示安装 Homebrew/Python；Ubuntu 可提示用户安装 `python3-venv`、`python3-pip`。

3. 生成或编辑配置：
   ```bash
   python3 scripts/model_benchmark.py menu --output configs/model_benchmark.local.yaml
   ```
   配置文件是最终可复现入口。菜单会引导第一次使用者填写模型名、端点 URL、API key、数据集、usage/tokenizer 策略、目标指标和测试场景。API key 不写入 YAML；如用户直接输入 key，会保存到 `environment.env_file` 指定的本地 env 文件，默认 `.model_benchmark.env`，权限设为 `0600`。

4. 先跑真实 API 冒烟测试：
   ```bash
   export DASHSCOPE_API_KEY=...
   python3 scripts/model_benchmark.py run --config configs/model_benchmark.local.yaml --scenario smoke
   ```
   默认模型是 `qwen3.6-plus`，默认 endpoint 是 DashScope OpenAI-compatible URL，默认从 `DASHSCOPE_API_KEY` 读取密钥。若密钥缺失，不要假装通过，明确说明真实验收被凭证阻塞。

5. 冒烟通过后运行正式压测：
   ```bash
   python3 scripts/model_benchmark.py run --config configs/model_benchmark.local.yaml --scenario all
   ```
   只会执行配置中 `enabled: true` 的场景。默认启用 `smoke` 和 `gradient`，`sla`、`stability`、`length_matrix` 默认关闭，避免误触长时间或高成本任务。

6. 需要只生成报告时：
   ```bash
   python3 scripts/model_benchmark.py report \
     --config configs/model_benchmark.local.yaml \
     --results-dir outputs/model_benchmark/<run_id> \
     --output outputs/model_benchmark/<run_id>/model_benchmark_report.md
   ```

## Configuration

从 `configs/model_benchmark.example.yaml` 复制本地配置。核心字段：

- `environment`: venv 路径、env 文件路径、安装源、pip 镜像、ModelScope/HF 镜像设置。国内网络默认使用清华 PyPI 镜像和 `https://hf-mirror.com`。
- `model`: `api`、`name`、`api_url`、`api_key_env`、`tokenizer_path`。默认端点是阿里云 DashScope OpenAI-compatible URL。API key 只从环境变量或本地 env 文件读取，日志和命令摘要必须脱敏。
- `dataset`: `simulated`、`openqa`、`line_by_line`、`random`。首次测试优先使用 `simulated`，它会生成本地 openqa JSONL，避免外部数据集下载。
- `token_accounting`: `mode=auto|api_usage|tokenizer`，`on_missing_usage=fail|fallback_tokenizer|skip_token_metrics`。默认 `auto + fallback_tokenizer`。
- `targets`: 成功率、QPS、输出吞吐、平均/P95/P99 TTFT、TPOT、E2E 延迟目标。报告必须展示目标、实测值、差距和达标状态。
- `scenarios`: `smoke`、`gradient`、`sla`、`stability`、`length_matrix` 的开关和参数。

## EvalScope Perf Rules

- 使用当前 EvalScope perf 参数语义。输入长度用 `--min-prompt-length` / `--max-prompt-length`；输出长度用 `--min-tokens` / `--max-tokens`。
- `random` 数据集必须有 `tokenizer_path`。如果没有 tokenizer，自动退回 `simulated/openqa`，并在报告或 stderr 中说明。
- 并发梯度中 `parallel` 和 `number` 必须一一对应；默认请求数为 `max(min_number, parallel * number_multiplier)`。
- SLA 成功率目标为 100% 且目标能映射为 EvalScope 指标时，使用官方 `--sla-auto-tune`；否则使用本 skill 的有界二分搜索。
- 对 OpenAI-compatible API 默认注册 `openai_optional_usage` 插件：优先读取 API `usage`；usage 缺失时按配置 fallback 到 tokenizer、失败退出或跳过 token 指标。
- 空响应不能算成功。自定义插件会把空 `generated_text` 标记失败；`patch_evalscope.py` 只保留为诊断/兼容工具，不作为默认路径修改 site-packages。

## Report Requirements

报告必须是 Markdown，并包含：

- 测试信息：时间、模型、API、数据集、token 计量策略、结果目录。
- 目标达标总览：每个目标的目标值、最佳实测、差距、是否达标、来源并发。
- 结论建议：QPS 峰值、最高安全并发、未达标原因。
- 基本性能表：场景、并发、请求数、成功率、QPS、吞吐。
- TTFT、TPOT、E2E、ITL：平均、P50、P90、P95、P99。
- Token 统计：平均输入/输出 tokens。
- 错误摘要和原始结果路径。

当 `on_missing_usage=skip_token_metrics`，或结果中 token 统计全为 0 时，报告中 token 吞吐、TPOT、平均 token 字段显示为不可计，并明确提醒这些指标不能作为结论。

## Troubleshooting

- `No Python >= 3.10`: 先安装 Python。macOS 推荐 `brew install python@3.12`；Ubuntu 推荐安装 `python3 python3-venv python3-pip`。
- `evalscope not installed`: 运行 `bootstrap`。国内网络下不要直接使用默认 PyPI。
- `DASHSCOPE_API_KEY missing`: 设置配置中的 `model.api_key_env` 对应环境变量。
- `random requires tokenizer_path`: 填 ModelScope tokenizer ID，例如 `Qwen/Qwen3-0.6B`，或把数据集改为 `simulated/openqa`。
- `usage missing`: 选择 `fallback_tokenizer` 并提供 tokenizer；若只关心请求级延迟和 QPS，可选择 `skip_token_metrics`，但报告必须标记 token 指标不可计。
- `endpoint unreachable`: 检查 URL 是否为完整 `/v1/chat/completions`，或运行 `doctor --check-endpoint`。

## Legacy Commands

旧命令仍可使用，但它们会委托给统一 runner：

```bash
bash scripts/benchmark.sh --api-url ... --api-key ... --model ...
bash scripts/stability_test.sh --api-url ... --api-key ... --model ...
python3 scripts/sla_autotune.py --api-url ... --api-key ... --model ... --target-ttft 100
python3 scripts/generate_report.py outputs/model_benchmark/<run_id> --output report.md
python3 scripts/convert_dataset.py input.jsonl output.jsonl --format auto
```
