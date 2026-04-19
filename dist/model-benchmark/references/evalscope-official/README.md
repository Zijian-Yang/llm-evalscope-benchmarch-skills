# EvalScope Official Reference Summary

- Use `evalscope perf` or the Python API `evalscope.perf.main.run_perf_benchmark` for HTTP API model benchmark tasks.
- Required fields are model name, endpoint URL, API protocol, dataset, concurrency and request count.
- OpenAI-compatible endpoints normally use `/v1/chat/completions`, `--api openai`, `--parallel`, `--number`, `--max-tokens`, and optional `--tokenizer-path`.
- Gradient tests pair `parallel` and `number` positionally, for example `(1,50)`, `(5,50)`, `(10,100)`.
- `random` datasets require a tokenizer; `openqa` JSONL requires one JSON object per line with a non-empty `question`.
- Discover exact installed parameters with `evalscope perf --help`; prefer installed docs over copied long references.
