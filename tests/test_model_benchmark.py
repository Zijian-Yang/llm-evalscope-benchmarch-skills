import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import model_benchmark as mb  # noqa: E402


class ModelBenchmarkTests(unittest.TestCase):

    def make_config(self, tmpdir: Path) -> dict:
        config = copy.deepcopy(mb.DEFAULT_CONFIG)
        config["run"]["outputs_dir"] = str(tmpdir / "outputs")
        config["dataset"]["path"] = str(tmpdir / "simulated.jsonl")
        config["model"]["api_key_env"] = "TEST_MODEL_BENCHMARK_KEY"
        os.environ["TEST_MODEL_BENCHMARK_KEY"] = "sk-test-secret"
        return config

    def test_minimal_yaml_loads_defaults_shape(self):
        data = mb.minimal_yaml_load(
            """
model:
  name: test-model
  api_url: https://example.test/v1/chat/completions
gradient:
  parallels: [1, 2, 5]
  enabled: true
target:
  value: null
"""
        )
        self.assertEqual(data["model"]["name"], "test-model")
        self.assertEqual(data["gradient"]["parallels"], [1, 2, 5])
        self.assertTrue(data["gradient"]["enabled"])
        self.assertIsNone(data["target"]["value"])

    def test_build_random_evalscope_args_use_prompt_length_controls(self):
        with tempfile.TemporaryDirectory() as td:
            config = self.make_config(Path(td))
            config["dataset"]["type"] = "random"
            config["dataset"]["random_prompt_tokens"] = 384
            config["model"]["tokenizer_path"] = "Qwen/Qwen3-0.6B"
            scenario = {
                "parallels": [1, 2],
                "number_multiplier": 10,
                "min_number": 20,
                "max_tokens": 64,
                "sleep_interval": 1,
            }
            eval_args, warnings = mb.build_evalscope_args(config, "gradient", scenario, Path(td) / "out")

            self.assertEqual(warnings, [])
            self.assertEqual(eval_args["dataset"], "random")
            self.assertEqual(eval_args["min_prompt_length"], 384)
            self.assertEqual(eval_args["max_prompt_length"], 384)
            self.assertEqual(eval_args["max_tokens"], 64)
            self.assertEqual(eval_args["parallel"], [1, 2])
            self.assertEqual(eval_args["number"], [20, 20])
            self.assertEqual(eval_args["api"], "openai_optional_usage")
            self.assertNotIn("api_key", eval_args)
            self.assertEqual(eval_args["api_key_env"], "TEST_MODEL_BENCHMARK_KEY")

    def test_random_without_tokenizer_falls_back_to_simulated_openqa(self):
        with tempfile.TemporaryDirectory() as td:
            config = self.make_config(Path(td))
            config["dataset"]["type"] = "random"
            config["model"]["tokenizer_path"] = ""
            eval_args, warnings = mb.build_evalscope_args(
                config,
                "smoke",
                {"parallel": 1, "number": 2, "max_tokens": 16},
                Path(td) / "out",
            )

            self.assertEqual(eval_args["dataset"], "openqa")
            self.assertTrue(eval_args["dataset_path"].endswith("simulated.jsonl"))
            self.assertTrue(any("requires tokenizer_path" in warning for warning in warnings))
            self.assertTrue(Path(eval_args["dataset_path"]).exists())

    def test_convert_to_openqa_supports_messages_text_and_txt(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            messages = tmp / "messages.jsonl"
            messages.write_text(
                json.dumps({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]})
                + "\n",
                encoding="utf-8",
            )
            out = tmp / "out.jsonl"
            total, success, failed = mb.convert_to_openqa(messages, out)
            self.assertEqual((total, success, failed), (1, 1, 0))
            self.assertIn("[System]: s", out.read_text(encoding="utf-8"))

            txt = tmp / "prompts.txt"
            txt.write_text("hello\nworld\n", encoding="utf-8")
            total, success, failed = mb.convert_to_openqa(txt, tmp / "txt_out.jsonl")
            self.assertEqual((total, success, failed), (2, 2, 0))

    def test_dataset_validation_reports_supported_formats(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            openqa = tmp / "openqa.jsonl"
            openqa.write_text(json.dumps({"question": "hello"}) + "\n", encoding="utf-8")
            ok, message = mb.dataset_validation_message("openqa", str(openqa))
            self.assertTrue(ok)
            self.assertIn("格式检查通过", message)

            messages = tmp / "messages.jsonl"
            messages.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n", encoding="utf-8")
            ok, message = mb.dataset_validation_message("openqa", str(messages))
            self.assertFalse(ok)
            self.assertIn("可转换", message)
            self.assertIn("openqa JSONL", message)

            bad = tmp / "bad.jsonl"
            bad.write_text(json.dumps({"foo": "bar"}) + "\n", encoding="utf-8")
            ok, message = mb.dataset_validation_message("openqa", str(bad))
            self.assertFalse(ok)
            self.assertIn("EvalScope perf 常用数据集格式", message)

            txt = tmp / "prompts.txt"
            txt.write_text("hello\n", encoding="utf-8")
            ok, message = mb.dataset_validation_message("line_by_line", str(txt))
            self.assertTrue(ok)
            self.assertIn("line_by_line TXT", message)

            no_suffix = tmp / "prompts.data"
            no_suffix.write_text("hello\nworld\n", encoding="utf-8")
            ok, message = mb.dataset_validation_message("line_by_line", str(no_suffix))
            self.assertTrue(ok)
            self.assertIn("line_by_line TXT", message)

    def test_build_evalscope_args_rejects_invalid_openqa_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            config = self.make_config(tmp)
            bad = tmp / "bad.jsonl"
            bad.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n", encoding="utf-8")
            config["dataset"]["type"] = "openqa"
            config["dataset"]["path"] = str(bad)

            with self.assertRaises(mb.ConfigError) as ctx:
                mb.build_evalscope_args(config, "smoke", {"parallel": 1, "number": 1, "max_tokens": 8}, tmp / "out")

            self.assertIn("可转换", str(ctx.exception))

    def test_report_parses_nested_and_flat_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            config = self.make_config(tmp)
            config["dataset"]["name"] = "unit_test_dataset"
            config["targets"]["qps"] = 2
            config["targets"]["avg_ttft_ms"] = 500

            nested = tmp / "gradient" / "20260101" / "model" / "parallel_1_number_2"
            flat = tmp / "parallel_2"
            for path, parallel, qps, ttft in [(nested, 1, 3.5, 0.2), (flat, 2, 1.0, 0.7)]:
                path.mkdir(parents=True)
                (path / "benchmark_summary.json").write_text(
                    json.dumps(
                        {
                            "Number of concurrency": parallel,
                            "Total requests": 2,
                            "Succeed requests": 2,
                            "Failed requests": 0,
                            "Request throughput (req/s)": qps,
                            "Output token throughput (tok/s)": 10,
                            "Total token throughput (tok/s)": 20,
                            "Average time to first token (s)": ttft,
                            "Average time per output token (s)": 0.01,
                            "Average latency (s)": 1.0,
                            "Average inter-token latency (s)": 0.01,
                            "Average input tokens per request": 12,
                            "Average output tokens per request": 8,
                        }
                    ),
                    encoding="utf-8",
                )
                (path / "benchmark_percentile.json").write_text(
                    json.dumps(
                        [
                            {"Percentiles": "50%", "TTFT (s)": ttft, "TPOT (s)": 0.01, "Latency (s)": 1.0, "ITL (s)": 0.01},
                            {
                                "Percentiles": "90%",
                                "TTFT (s)": ttft + 0.1,
                                "TPOT (s)": 0.02,
                                "Latency (s)": 1.2,
                                "ITL (s)": 0.02,
                                "Input tokens": 12,
                                "Output tokens": 8,
                                "Output (tok/s)": 6.5,
                                "Total (tok/s)": 18.5,
                            },
                            {
                                "Percentiles": "95%",
                                "TTFT (s)": ttft + 0.2,
                                "TPOT (s)": 0.03,
                                "Latency (s)": 1.3,
                                "ITL (s)": 0.03,
                                "Input tokens": 13,
                                "Output tokens": 9,
                                "Output (tok/s)": 7.5,
                                "Total (tok/s)": 19.5,
                            },
                            {
                                "Percentiles": "99%",
                                "TTFT (s)": ttft + 0.3,
                                "TPOT (s)": 0.04,
                                "Latency (s)": 1.4,
                                "ITL (s)": 0.04,
                                "Input tokens": 14,
                                "Output tokens": 10,
                                "Output (tok/s)": 8.5,
                                "Total (tok/s)": 20.5,
                            },
                        ]
                    ),
                    encoding="utf-8",
                )

            report_path = tmp / "report.md"
            report = mb.generate_report(config, tmp, report_path)
            self.assertIn("指标覆盖检查", report)
            self.assertIn("压测数据集名称", report)
            self.assertIn("unit_test_dataset", report)
            self.assertIn("EvalScope dataset", report)
            self.assertIn("数据量与成功率", report)
            self.assertIn("QPS 与吞吐", report)
            self.assertIn("Token 统计", report)
            self.assertNotIn("目标达标总览", report)
            self.assertIn("QPS 峰值", report)
            self.assertIn("parallel_2", report)
            self.assertTrue(report_path.exists())

    def test_extract_usage_tokens(self):
        responses = [
            {"choices": [{"delta": {"content": "hello"}, "index": 0}], "usage": None},
            {"choices": [{"delta": {}, "index": 0}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
        ]
        self.assertEqual(mb.extract_usage_tokens(responses), (7, 3))
        self.assertIsNone(mb.extract_usage_tokens([{"choices": []}]))

    def test_env_file_is_used_without_storing_secret_in_payload(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            config = self.make_config(tmp)
            os.environ.pop("TEST_MODEL_BENCHMARK_KEY", None)
            env_file = tmp / ".model_benchmark.env"
            config["environment"]["env_file"] = str(env_file)
            mb.write_env_secret(env_file, "TEST_MODEL_BENCHMARK_KEY", "sk-from-file")

            eval_args, _ = mb.build_evalscope_args(
                config,
                "smoke",
                {"parallel": 1, "number": 1, "max_tokens": 8},
                tmp / "out",
            )
            payload = {"evalscope_args": eval_args}
            dumped = json.dumps(payload)

            self.assertEqual(mb.get_api_key(config), "sk-from-file")
            self.assertIn("api_key_env", eval_args)
            self.assertNotIn("sk-from-file", dumped)

    def test_attach_safe_to_dict_masks_evalscope_args(self):
        class Args:
            def to_dict(self):
                return {
                    "api_key": "sk-secret-value",
                    "headers": {"Authorization": "Bearer sk-secret-value"},
                }

        args = Args()
        mb.attach_safe_to_dict(args)
        text = json.dumps(args.to_dict())

        self.assertNotIn("sk-secret-value", text)
        self.assertIn("Bearer sk-s", text)

    def test_build_parallel_values(self):
        self.assertEqual(mb.build_parallel_values("step", [1], 1, 10, step=3), [1, 4, 7, 10])
        self.assertEqual(mb.build_parallel_values("count", [1], 1, 10, count=4), [1, 4, 7, 10])
        self.assertEqual(mb.build_parallel_values("multiply", [1], 1, 10, multiplier=2), [1, 2, 4, 8, 10])
        self.assertEqual(mb.parse_int_values("1, 2, 5"), [1, 2, 5])

    def test_quick_start_clears_targets_and_uses_small_gradient(self):
        config = copy.deepcopy(mb.DEFAULT_CONFIG)
        original_yes_no = mb.prompt_yes_no
        try:
            mb.prompt_yes_no = lambda _label, default: default
            mb.configure_quick_start(config)
        finally:
            mb.prompt_yes_no = original_yes_no

        self.assertEqual(config["token_accounting"]["mode"], "auto")
        self.assertEqual(config["token_accounting"]["on_missing_usage"], "skip_token_metrics")
        self.assertEqual(config["dataset"]["type"], "simulated")
        self.assertEqual(config["scenarios"]["gradient"]["parallels"], [1, 2, 5])
        self.assertEqual(config["scenarios"]["gradient"]["numbers"], [10, 20, 50])
        self.assertIsNone(config["targets"]["qps"])
        self.assertIsNone(config["targets"]["success_rate_pct"])

    def test_main_without_args_defaults_to_menu(self):
        original = mb.run_menu
        calls = []

        def fake_menu(args):
            calls.append(args.output)
            return 0

        try:
            mb.run_menu = fake_menu
            rc = mb.main([])
        finally:
            mb.run_menu = original

        self.assertEqual(rc, 0)
        self.assertTrue(calls)
        self.assertTrue(calls[0].endswith("configs/model_benchmark.local.yaml"))


if __name__ == "__main__":
    unittest.main()
