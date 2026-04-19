#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "model_benchmark.example.yaml"
DISCLAIMER = "本报告仅供参考，不代表官方认证；生产性能会受网络、限流、负载和请求内容影响。"
TOKEN_MODES = ["prefer_api_usage", "api_usage_only", "tokenizer_only", "dual_compare", "disabled"]
MISSING_USAGE_POLICIES = ["fallback_tokenizer", "mark_unavailable", "fail"]
TOKENIZER_SOURCES = ["modelscope", "huggingface", "local_path", "disabled"]
DEFAULT = {
    "environment": {
        "venv_path": ".venv-model-benchmark",
        "env_file": ".model_benchmark.env",
        "evalscope_extras": "perf",
        "pip_index_url": "https://pypi.tuna.tsinghua.edu.cn/simple",
        "pip_trusted_host": "pypi.tuna.tsinghua.edu.cn",
        "modelscope_cache": ".cache/modelscope",
        "hf_endpoint": "https://hf-mirror.com",
    },
    "model": {
        "api": "openai",
        "name": "qwen3.6-plus",
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "dataset": {
        "type": "simulated",
        "name": "simulated_openqa",
        "path": "outputs/simulated_openqa.jsonl",
        "simulated_count": 12,
        "output_tokens": 128,
        "random_prompt_tokens": 512,
    },
    "token_accounting": {
        "mode": "prefer_api_usage",
        "on_missing_usage": "fallback_tokenizer",
        "tokenizer_source": "modelscope",
        "tokenizer_path": "Qwen/Qwen3-0.6B",
    },
    "scenarios": {
        "smoke": {"enabled": True, "parallel": 1, "number": 3, "max_tokens": 32},
        "gradient": {
            "enabled": True,
            "parallels": [1, 2, 5, 10, 20],
            "number_multiplier": 10,
            "min_number": 50,
            "max_tokens": 128,
            "sleep_interval": 5,
        },
        "rate": {
            "enabled": False,
            "parallel": 10,
            "rates": [1.0, 2.0, 5.0],
            "number_multiplier": 60,
            "min_number": 60,
            "max_tokens": 128,
            "sleep_interval": 5,
        },
        "stability": {
            "enabled": False,
            "parallel": 10,
            "duration_minutes": 30,
            "window_minutes": 5,
            "max_tokens": 128,
        },
        "length_matrix": {
            "enabled": False,
            "parallel": 5,
            "number": 100,
            "input_tokens": [100, 500, 1000],
            "output_tokens": [32, 128, 256],
        },
    },
    "run": {
        "outputs_dir": "outputs/model_benchmark",
        "stream": True,
        "headers": {},
        "connect_timeout": None,
        "read_timeout": None,
        "total_timeout": 21600,
        "warmup_requests": 0,
        "warmup_parallel": 1,
        "cooldown_seconds": 0,
    },
}


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(json.dumps(base, ensure_ascii=False))
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = deep_merge(data[key], value)
        else:
            data[key] = value
    return data


def parse_scalar(value: str) -> Any:
    value = value.strip().strip("\"'")
    if value in {"", "null", "None", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return [parse_scalar(item) for item in value[1:-1].split(",") if item.strip()]
    if value.startswith("{") and value.endswith("}"):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else value
        except json.JSONDecodeError:
            return value
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict) and isinstance(data.get("model", {}).get("tokenizer_path"), str):
            data.setdefault("token_accounting", {})
            data["token_accounting"].setdefault("tokenizer_path", data["model"].pop("tokenizer_path"))
        return deep_merge(DEFAULT, data if isinstance(data, dict) else {})
    except Exception:
        data: dict[str, Any] = {}
        stack: list[tuple[int, dict[str, Any]]] = [(-1, data)]
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip() or ":" not in line:
                continue
            indent = len(line) - len(line.lstrip())
            key, value = line.strip().split(":", 1)
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if value.strip():
                parent[key] = parse_scalar(value)
            else:
                parent[key] = {}
                stack.append((indent, parent[key]))
        return deep_merge(DEFAULT, data)


def yaml_value(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def dump_yaml(data: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                if value:
                    lines.append(f"{pad}{key}:")
                    lines.append(dump_yaml(value, indent + 1))
                else:
                    lines.append(f"{pad}{key}: {{}}")
            else:
                lines.append(f"{pad}{key}: {yaml_value(value)}")
        return "\n".join(lines)
    return f"{pad}{yaml_value(data)}"


def save_config(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(data).rstrip() + "\n", encoding="utf-8")


def get(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = data
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def put(data: dict[str, Any], dotted: str, value: Any) -> None:
    node = data
    parts = dotted.split(".")
    for key in parts[:-1]:
        node = node.setdefault(key, {})
    node[parts[-1]] = value


def num(value: Any) -> float | None:
    try:
        value = float(value)
        return None if value != value or value in {float("inf"), float("-inf")} else value
    except Exception:
        return None


def ms(value: Any) -> float | None:
    value = num(value)
    return None if value is None else value * 1000.0


def fmt(value: Any, digits: int = 2, unavailable: bool = False) -> str:
    value = None if unavailable else num(value)
    return "-" if value is None else str(int(round(value))) if digits == 0 else f"{value:.{digits}f}"


def prompt(label: str, default: Any = None, note: str = "", cast=None, allow_blank: bool = False) -> Any:
    text = f"{label}{f'（{note}）' if note else ''} [{'' if default is None else default}]: "
    try:
        raw = input(text).strip()
    except EOFError:
        raw = ""
    if raw == "":
        return None if allow_blank and default is None else default
    value = parse_scalar(raw)
    if cast:
        try:
            return cast(value)
        except Exception:
            print("输入无效，已使用默认值。")
            return default
    return value


def choose(label: str, options: list[tuple[str, str]], default: str) -> str:
    print(f"\n{label}")
    for idx, (_, note) in enumerate(options, 1):
        print(f"  {idx}. {note}")
    raw = prompt("请输入编号", next((idx for idx, item in enumerate(options, 1) if item[0] == default), 1))
    try:
        return options[max(1, min(len(options), int(raw))) - 1][0]
    except Exception:
        return default


def ask_list(label: str, default: list[int | float], cast=int) -> list[int | float]:
    raw = prompt(label, ",".join(str(item) for item in default))
    return [cast(parse_scalar(str(item).strip())) for item in str(raw).replace("[", "").replace("]", "").split(",") if str(item).strip()]


def ask_json_dict(label: str, default: dict[str, Any] | None, note: str = "") -> dict[str, Any]:
    default_text = json.dumps(default or {}, ensure_ascii=False)
    raw = prompt(label, default_text, note)
    if raw in {"", None, "{}"}:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        print("不是合法 JSON 对象，已使用空 headers。")
        return {}


def yes(label: str, default: bool) -> bool:
    return str(prompt(label, "y" if default else "n")).lower() in {"y", "yes", "1", "true", "是", "启用"}


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        key.strip(): value.strip().strip("\"'")
        for raw in path.read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.lstrip().startswith("#") and "=" in raw
        for key, value in [raw.split("=", 1)]
    }


def api_key(config: dict[str, Any]) -> tuple[str, str]:
    name = str(get(config, "model.api_key_env"))
    env_path = project_path(get(config, "environment.env_file"))
    return name, os.environ.get(name) or read_env_file(env_path).get(name, "")


def tokenizer_path(config: dict[str, Any]) -> str | None:
    if get(config, "token_accounting.tokenizer_source") == "disabled":
        return None
    return get(config, "token_accounting.tokenizer_path") or get(config, "model.tokenizer_path")


def venv_python(config: dict[str, Any]) -> Path:
    return project_path(get(config, "environment.venv_path")) / "bin" / "python"


def active_python(config: dict[str, Any]) -> str:
    py = venv_python(config)
    return str(py if py.exists() else Path(sys.executable))


def evalscope_available(config: dict[str, Any]) -> bool:
    return venv_python(config).exists() or shutil.which("evalscope") is not None


def check_endpoint(url: str) -> str:
    root = url.split("/v1/", 1)[0] if "/v1/" in url else url
    try:
        with urllib.request.urlopen(root, timeout=5) as response:
            return f"reachable ({response.status})"
    except urllib.error.URLError as exc:
        return f"unreachable ({exc.reason})"


def validate_dataset(config: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    kind = str(get(config, "dataset.type"))
    if kind == "simulated":
        return messages
    if kind in {"openqa", "line_by_line"}:
        path = project_path(get(config, "dataset.path", ""))
        if not path.exists():
            messages.append(f"数据集文件不存在：{path}")
            return messages
        try:
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except UnicodeDecodeError:
            messages.append(f"数据集不是 UTF-8 文本：{path}")
            return messages
        if not lines:
            messages.append(f"数据集没有有效行：{path}")
        elif kind == "line_by_line":
            messages.append(f"line_by_line 校验通过：{len(lines)} 条 prompt。")
        else:
            valid = 0
            invalid = 0
            for line in lines[:20]:
                try:
                    row = json.loads(line)
                    valid += int(isinstance(row, dict) and isinstance(row.get("question"), str) and bool(row["question"].strip()))
                    invalid += int(not (isinstance(row, dict) and isinstance(row.get("question"), str) and bool(row["question"].strip())))
                except Exception:
                    invalid += 1
            messages.append(f"openqa 抽样校验：有效 {valid}，无效 {invalid}。")
    if kind == "random" and not tokenizer_path(config):
        messages.append("random 数据集必须提供 tokenizer_path。")
    return messages


def doctor(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print(f"python_ok={sys.version_info >= (3, 10)}")
    print(f"config={Path(args.config).resolve()}")
    print(f"evalscope_available={evalscope_available(config)}")
    if args.check_endpoint:
        print(f"endpoint={check_endpoint(str(get(config, 'model.api_url', '')))}")
    for line in validate_dataset(config):
        print(f"dataset_check={line}")
    print("required_inputs=model.name, model.api_url, model.api_key_env, dataset.*, token_accounting.*, scenarios.*, run.*")
    return 0 if sys.version_info >= (3, 10) else 2


def bootstrap(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    venv_dir = project_path(get(config, "environment.venv_path"))
    if not venv_dir.exists():
        venv.EnvBuilder(with_pip=True).create(venv_dir)
    command = [str(venv_dir / "bin" / "pip"), "install", f"evalscope[{get(config, 'environment.evalscope_extras')}]"]
    if get(config, "environment.pip_index_url"):
        command += ["-i", str(get(config, "environment.pip_index_url"))]
    if get(config, "environment.pip_trusted_host"):
        command += ["--trusted-host", str(get(config, "environment.pip_trusted_host"))]
    return subprocess.call(command)


def menu(args: argparse.Namespace) -> int:
    output = Path(args.output)
    config = load_config(output if output.exists() else Path(args.config))
    print("将逐步收集模型、数据集、token 计量、场景与高级控制参数；直接回车使用默认值。")
    for key, note in [
        ("model.name", "报告中的模型名"),
        ("model.api", "通常 openai / dashscope"),
        ("model.api_url", "完整 /v1/chat/completions URL"),
        ("model.api_key_env", "只保存变量名，不保存密钥"),
    ]:
        put(config, key, prompt(key, get(config, key), note))
    kind = choose(
        "数据集类型",
        [("simulated", "内置模拟数据"), ("openqa", "OpenQA JSONL"), ("line_by_line", "TXT 每行一个 prompt"), ("random", "随机 token 数据")],
        get(config, "dataset.type"),
    )
    put(config, "dataset.type", kind)
    put(config, "dataset.name", prompt("dataset.name", get(config, "dataset.name"), "写入报告，便于追溯"))
    if kind == "simulated":
        put(config, "dataset.path", prompt("dataset.path", get(config, "dataset.path")))
        put(config, "dataset.simulated_count", prompt("dataset.simulated_count", get(config, "dataset.simulated_count"), "模拟样本数", int))
    elif kind == "random":
        put(config, "dataset.random_prompt_tokens", prompt("dataset.random_prompt_tokens", get(config, "dataset.random_prompt_tokens"), "固定输入长度 tokens", int))
    else:
        while True:
            path = prompt("dataset.path", get(config, "dataset.path"), "数据集路径")
            if path:
                put(config, "dataset.path", path)
                break
            print("该数据集类型必须提供文件路径。")
    put(config, "dataset.output_tokens", prompt("dataset.output_tokens", get(config, "dataset.output_tokens"), "默认输出 token 上限", int))
    put(
        config,
        "token_accounting.mode",
        choose(
            "Token 计量模式",
            [
                ("prefer_api_usage", "优先 API usage，缺失时按下面策略处理"),
                ("api_usage_only", "只认 API usage"),
                ("tokenizer_only", "完全使用 tokenizer 估算"),
                ("dual_compare", "同时记录 usage 与 tokenizer 对比；指标默认仍使用 usage"),
                ("disabled", "不统计 token / TPOT / token 吞吐"),
            ],
            get(config, "token_accounting.mode"),
        ),
    )
    put(
        config,
        "token_accounting.on_missing_usage",
        choose(
            "usage 缺失时",
            [
                ("fallback_tokenizer", "回退 tokenizer"),
                ("mark_unavailable", "标记 token 指标不可计"),
                ("fail", "失败并停止"),
            ],
            get(config, "token_accounting.on_missing_usage"),
        ),
    )
    put(
        config,
        "token_accounting.tokenizer_source",
        choose(
            "tokenizer 来源",
            [
                ("modelscope", "ModelScope ID"),
                ("huggingface", "Hugging Face ID"),
                ("local_path", "本地路径"),
                ("disabled", "不提供 tokenizer"),
            ],
            get(config, "token_accounting.tokenizer_source"),
        ),
    )
    if kind == "random" or get(config, "token_accounting.mode") in {"tokenizer_only", "dual_compare"} or get(config, "token_accounting.on_missing_usage") == "fallback_tokenizer":
        put(config, "token_accounting.tokenizer_path", prompt("token_accounting.tokenizer_path", tokenizer_path(config), "random / tokenizer 计量时必填"))
    smoke = yes("启用 smoke 冒烟", bool(get(config, "scenarios.smoke.enabled")))
    put(config, "scenarios.smoke.enabled", smoke)
    if smoke:
        for key, note in [("scenarios.smoke.parallel", "建议 1"), ("scenarios.smoke.number", "建议 2-5"), ("scenarios.smoke.max_tokens", "越大越慢、成本越高")]:
            put(config, key, prompt(key, get(config, key), note, int))
    gradient = yes("启用 gradient 并发梯度", bool(get(config, "scenarios.gradient.enabled")))
    put(config, "scenarios.gradient.enabled", gradient)
    if gradient:
        put(config, "scenarios.gradient.parallels", ask_list("scenarios.gradient.parallels", list(get(config, "scenarios.gradient.parallels")), int))
        if yes("每档请求数按公式生成", True):
            put(config, "scenarios.gradient.number_multiplier", prompt("scenarios.gradient.number_multiplier", get(config, "scenarios.gradient.number_multiplier"), "number=max(min_number, parallel*倍数)", int))
            put(config, "scenarios.gradient.min_number", prompt("scenarios.gradient.min_number", get(config, "scenarios.gradient.min_number"), "太小会导致统计不稳定", int))
            put(config, "scenarios.gradient.numbers", None)
        else:
            default_numbers = [max(int(get(config, "scenarios.gradient.min_number")), int(item) * int(get(config, "scenarios.gradient.number_multiplier"))) for item in get(config, "scenarios.gradient.parallels")]
            put(config, "scenarios.gradient.numbers", ask_list("scenarios.gradient.numbers", default_numbers, int))
        put(config, "scenarios.gradient.max_tokens", prompt("scenarios.gradient.max_tokens", get(config, "scenarios.gradient.max_tokens"), "越大越慢、成本越高", int))
        put(config, "scenarios.gradient.sleep_interval", prompt("scenarios.gradient.sleep_interval", get(config, "scenarios.gradient.sleep_interval"), "档位之间等待秒数", int))
    rate = yes("启用 rate 定速压测", bool(get(config, "scenarios.rate.enabled")))
    put(config, "scenarios.rate.enabled", rate)
    if rate:
        put(config, "scenarios.rate.parallel", prompt("scenarios.rate.parallel", get(config, "scenarios.rate.parallel"), "固定并发上限", int))
        put(config, "scenarios.rate.rates", ask_list("scenarios.rate.rates", list(get(config, "scenarios.rate.rates")), float))
        put(config, "scenarios.rate.number_multiplier", prompt("scenarios.rate.number_multiplier", get(config, "scenarios.rate.number_multiplier"), "number=max(min_number, rate*倍数)", int))
        put(config, "scenarios.rate.min_number", prompt("scenarios.rate.min_number", get(config, "scenarios.rate.min_number"), "太小会导致统计不稳定", int))
        put(config, "scenarios.rate.max_tokens", prompt("scenarios.rate.max_tokens", get(config, "scenarios.rate.max_tokens"), "越大越慢、成本越高", int))
        put(config, "scenarios.rate.sleep_interval", prompt("scenarios.rate.sleep_interval", get(config, "scenarios.rate.sleep_interval"), "档位之间等待秒数", int))
    stability = yes("启用 stability 稳定性", bool(get(config, "scenarios.stability.enabled")))
    put(config, "scenarios.stability.enabled", stability)
    if stability:
        for key, note, caster in [
            ("scenarios.stability.parallel", "固定并发", int),
            ("scenarios.stability.duration_minutes", "总时长分钟", float),
            ("scenarios.stability.window_minutes", "分窗分钟", float),
            ("scenarios.stability.max_tokens", "越大越慢", int),
        ]:
            put(config, key, prompt(key, get(config, key), note, caster))
    matrix = yes("启用 length_matrix 输入/输出长度矩阵", bool(get(config, "scenarios.length_matrix.enabled")))
    put(config, "scenarios.length_matrix.enabled", matrix)
    if matrix:
        put(config, "scenarios.length_matrix.parallel", prompt("scenarios.length_matrix.parallel", get(config, "scenarios.length_matrix.parallel"), "矩阵测试并发", int))
        put(config, "scenarios.length_matrix.number", prompt("scenarios.length_matrix.number", get(config, "scenarios.length_matrix.number"), "每组请求数", int))
        put(config, "scenarios.length_matrix.input_tokens", ask_list("scenarios.length_matrix.input_tokens", list(get(config, "scenarios.length_matrix.input_tokens")), int))
        put(config, "scenarios.length_matrix.output_tokens", ask_list("scenarios.length_matrix.output_tokens", list(get(config, "scenarios.length_matrix.output_tokens")), int))
        put(config, "token_accounting.tokenizer_path", prompt("token_accounting.tokenizer_path", tokenizer_path(config), "长度矩阵必填"))
    put(config, "run.connect_timeout", prompt("run.connect_timeout", get(config, "run.connect_timeout"), "连接超时秒；留空使用底层默认", int, True))
    put(config, "run.read_timeout", prompt("run.read_timeout", get(config, "run.read_timeout"), "读取超时秒；留空使用底层默认", int, True))
    put(config, "run.total_timeout", prompt("run.total_timeout", get(config, "run.total_timeout"), "总超时秒", int))
    put(config, "run.headers", ask_json_dict("run.headers", get(config, "run.headers"), "JSON 对象，例如 {\"x-trace-id\":\"demo\"}"))
    put(config, "run.warmup_requests", prompt("run.warmup_requests", get(config, "run.warmup_requests"), "每个主场景前的预热请求数；0 表示禁用", int))
    put(config, "run.warmup_parallel", prompt("run.warmup_parallel", get(config, "run.warmup_parallel"), "预热并发", int))
    put(config, "run.cooldown_seconds", prompt("run.cooldown_seconds", get(config, "run.cooldown_seconds"), "场景之间冷却秒数", int))
    save_config(output, config)
    env_path = project_path(get(config, "environment.env_file"))
    if not env_path.exists():
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(f"{get(config, 'model.api_key_env')}=\n", encoding="utf-8")
    print(f"Config written: {output}")
    return 0


def ensure_dataset(config: dict[str, Any], scenario_name: str, override: dict[str, Any] | None = None) -> dict[str, Any]:
    override = override or {}
    kind = override.get("dataset_type", get(config, "dataset.type"))
    if kind == "simulated":
        path = project_path(get(config, "dataset.path"))
        path.parent.mkdir(parents=True, exist_ok=True)
        count = int(get(config, "dataset.simulated_count"))
        with path.open("w", encoding="utf-8") as handle:
            for idx in range(count):
                handle.write(json.dumps({"question": f"请用简洁中文回答第 {idx + 1} 个 {scenario_name} 压测问题。"}, ensure_ascii=False) + "\n")
        return {"dataset": "openqa", "dataset_path": str(path)}
    if kind in {"openqa", "line_by_line"}:
        path = project_path(get(config, "dataset.path"))
        if not path.exists():
            raise SystemExit(f"Missing dataset file: {path}")
        return {"dataset": kind, "dataset_path": str(path)}
    if kind == "random":
        if not tokenizer_path(config):
            raise SystemExit("random / length_matrix requires token_accounting.tokenizer_path")
        size = int(override.get("input_tokens", get(config, "dataset.random_prompt_tokens")))
        return {"dataset": "random", "min_prompt_length": size, "max_prompt_length": size}
    raise SystemExit(f"Unsupported dataset.type={kind}")


def base_args(config: dict[str, Any], scenario_name: str, out_dir: Path, override: dict[str, Any] | None = None) -> dict[str, Any]:
    override = override or {}
    key_name, secret = api_key(config)
    if not secret:
        raise SystemExit(f"Missing API key. Set {key_name} or environment.env_file.")
    args_data: dict[str, Any] = {
        "model": get(config, "model.name"),
        "url": get(config, "model.api_url"),
        "api": get(config, "model.api"),
        "api_key": secret,
        "outputs_dir": str(out_dir),
        "name": scenario_name,
        "stream": bool(get(config, "run.stream")),
        "no_timestamp": True,
        "max_tokens": int(override.get("max_tokens", get(config, f"scenarios.{scenario_name}.max_tokens", get(config, "dataset.output_tokens")))),
    }
    args_data.update(ensure_dataset(config, scenario_name, override))
    if tokenizer_path(config):
        args_data["tokenizer_path"] = tokenizer_path(config)
    if isinstance(get(config, "run.headers"), dict) and get(config, "run.headers"):
        args_data["headers"] = get(config, "run.headers")
    for key, dotted in [("connect_timeout", "run.connect_timeout"), ("read_timeout", "run.read_timeout"), ("total_timeout", "run.total_timeout")]:
        if get(config, dotted) is not None:
            args_data[key] = get(config, dotted)
    for key in ["parallel", "number", "rate", "sleep_interval", "min_tokens", "min_prompt_length", "max_prompt_length"]:
        if key in override and override[key] is not None:
            args_data[key] = override[key]
    return args_data


def run_evalscope(config: dict[str, Any], args_data: dict[str, Any], dry_run: bool) -> int:
    out_dir = Path(args_data["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = {key: ("***" if key == "api_key" else value) for key, value in args_data.items()}
    (out_dir / "command_summary.json").write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
    if dry_run:
        print(json.dumps(safe, ensure_ascii=False, indent=2))
        return 0
    payload = {
        "kwargs": args_data,
        "token_accounting": get(config, "token_accounting"),
        "audit_path": str(out_dir / "token_audit.jsonl"),
    }
    payload_path = out_dir / "run_payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    env = os.environ.copy()
    if get(config, "environment.modelscope_cache"):
        env.setdefault("MODELSCOPE_CACHE", str(project_path(get(config, "environment.modelscope_cache"))))
    if get(config, "environment.hf_endpoint"):
        env.setdefault("HF_ENDPOINT", str(get(config, "environment.hf_endpoint")))
    return subprocess.call([active_python(config), str(Path(__file__).resolve()), "_perf-run", "--payload", str(payload_path)], env=env)


def run_warmup(config: dict[str, Any], scenario_name: str, run_id: str, dry_run: bool) -> int:
    requests = int(get(config, "run.warmup_requests") or 0)
    if requests <= 0:
        return 0
    warmup_dir = project_path(get(config, "run.outputs_dir")) / run_id / scenario_name / "_warmup"
    args_data = base_args(
        config,
        scenario_name,
        warmup_dir,
        {
            "parallel": int(get(config, "run.warmup_parallel") or 1),
            "number": requests,
            "max_tokens": min(32, int(get(config, f"scenarios.{scenario_name}.max_tokens", get(config, "dataset.output_tokens")))),
        },
    )
    return run_evalscope(config, args_data, dry_run)


def maybe_cooldown(config: dict[str, Any], dry_run: bool) -> None:
    seconds = int(get(config, "run.cooldown_seconds") or 0)
    if seconds > 0 and not dry_run:
        time.sleep(seconds)


def run_scenario(config: dict[str, Any], run_id: str, scenario_name: str, dry_run: bool) -> int:
    base_dir = project_path(get(config, "run.outputs_dir")) / run_id / scenario_name
    if scenario_name != "smoke":
        warmup_rc = run_warmup(config, scenario_name, run_id, dry_run)
        if warmup_rc:
            return warmup_rc
    if scenario_name == "smoke":
        rc = run_evalscope(
            config,
            base_args(
                config,
                scenario_name,
                base_dir,
                {"parallel": int(get(config, "scenarios.smoke.parallel")), "number": int(get(config, "scenarios.smoke.number"))},
            ),
            dry_run,
        )
        maybe_cooldown(config, dry_run)
        return rc
    if scenario_name == "gradient":
        parallels = [int(item) for item in get(config, "scenarios.gradient.parallels")]
        numbers = get(config, "scenarios.gradient.numbers") or [max(int(get(config, "scenarios.gradient.min_number")), int(item) * int(get(config, "scenarios.gradient.number_multiplier"))) for item in parallels]
        rc = run_evalscope(config, base_args(config, scenario_name, base_dir, {"parallel": parallels, "number": numbers, "sleep_interval": int(get(config, "scenarios.gradient.sleep_interval"))}), dry_run)
        maybe_cooldown(config, dry_run)
        return rc
    if scenario_name == "rate":
        rc = 0
        for rate in [float(item) for item in get(config, "scenarios.rate.rates")]:
            number = max(int(get(config, "scenarios.rate.min_number")), int(round(rate * int(get(config, "scenarios.rate.number_multiplier")))))
            out_dir = base_dir / f"rate_{str(rate).replace('.', '_')}"
            rc = run_evalscope(
                config,
                base_args(
                    config,
                    scenario_name,
                    out_dir,
                    {
                        "parallel": int(get(config, "scenarios.rate.parallel")),
                        "rate": rate,
                        "number": number,
                        "max_tokens": int(get(config, "scenarios.rate.max_tokens")),
                    },
                ),
                dry_run,
            )
            if rc:
                return rc
            sleep_interval = int(get(config, "scenarios.rate.sleep_interval") or 0)
            if sleep_interval > 0 and not dry_run:
                time.sleep(sleep_interval)
        maybe_cooldown(config, dry_run)
        return rc
    if scenario_name == "stability":
        parallel = int(get(config, "scenarios.stability.parallel"))
        duration = float(get(config, "scenarios.stability.duration_minutes"))
        window = float(get(config, "scenarios.stability.window_minutes"))
        windows = max(1, int((duration + window - 1) // window))
        rc = 0
        for idx in range(1, windows + 1):
            out_dir = base_dir / f"window_{idx}"
            rc = run_evalscope(
                config,
                base_args(
                    config,
                    scenario_name,
                    out_dir,
                    {
                        "parallel": parallel,
                        "number": max(1, int(parallel * 60 * window)),
                        "max_tokens": int(get(config, "scenarios.stability.max_tokens")),
                    },
                ),
                dry_run,
            )
            if rc:
                return rc
        maybe_cooldown(config, dry_run)
        return rc
    if scenario_name == "length_matrix":
        rc = 0
        for input_tokens in [int(item) for item in get(config, "scenarios.length_matrix.input_tokens")]:
            for output_tokens in [int(item) for item in get(config, "scenarios.length_matrix.output_tokens")]:
                out_dir = base_dir / f"in{input_tokens}_out{output_tokens}"
                rc = run_evalscope(
                    config,
                    base_args(
                        config,
                        scenario_name,
                        out_dir,
                        {
                            "dataset_type": "random",
                            "input_tokens": input_tokens,
                            "parallel": int(get(config, "scenarios.length_matrix.parallel")),
                            "number": int(get(config, "scenarios.length_matrix.number")),
                            "max_tokens": output_tokens,
                            "min_tokens": output_tokens,
                        },
                    ),
                    dry_run,
                )
                if rc:
                    return rc
        maybe_cooldown(config, dry_run)
        return rc
    raise SystemExit(f"Unknown scenario: {scenario_name}")


def update_stats(config: dict[str, Any], scenario: str, seconds: float) -> None:
    path = project_path(get(config, "run.outputs_dir")) / "usage_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"run_count": 0}
    data.update(
        {
            "run_count": int(data.get("run_count", 0)) + 1,
            "last_model": get(config, "model.name"),
            "last_scenario": scenario,
            "last_duration_sec": round(seconds, 3),
        }
    )
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.scenario == "all":
        names = [name for name, cfg in get(config, "scenarios").items() if isinstance(cfg, dict) and cfg.get("enabled")]
    else:
        names = [args.scenario]
    start = time.time()
    for name in names:
        if not get(config, f"scenarios.{name}.enabled", False) and args.scenario == "all":
            continue
        rc = run_scenario(config, run_id, name, args.dry_run)
        if rc:
            return rc
    update_stats(config, args.scenario, time.time() - start)
    return 0


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_percentiles(data: Any) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        labels = data.get("Percentiles", ["50%", "90%", "95%", "99%"])
        rows = [{"Percentiles": label, **{key: values[idx] for key, values in data.items() if isinstance(values, list)}} for idx, label in enumerate(labels)]
    else:
        rows = []
    for row in rows:
        label = str(row.get("Percentiles") or row.get("percentile") or "")
        for key, value in row.items():
            if key not in {"Percentiles", "percentile"}:
                result.setdefault(key, {})[label] = num(value)
    return result


def collect_runs(results: Path) -> list[dict[str, Any]]:
    runs = []
    for summary_path in sorted(results.rglob("benchmark_summary.json")):
        summary = load_json(summary_path)
        if not isinstance(summary, dict):
            continue
        pct = parse_percentiles(load_json(summary_path.parent / "benchmark_percentile.json"))
        args_data = load_json(summary_path.parent / "benchmark_args.json") or {}
        total = num(summary.get("Total requests")) or 0
        succeed = num(summary.get("Succeed requests")) or 0
        run = {
            "path": str(summary_path.parent),
            "scenario": summary_path.parent.relative_to(results).parts[0] if summary_path.parent != results else summary_path.parent.name,
            "parallel": num(summary.get("Number of concurrency")) or next((float(part) for part in summary_path.parent.name.split("_") if part.isdigit()), None),
            "total_requests": total,
            "succeed_requests": succeed,
            "failed_requests": num(summary.get("Failed requests")) or 0,
            "success_rate_pct": (succeed / total * 100.0) if total else None,
            "qps": num(summary.get("Request throughput (req/s)")),
            "output_tps": num(summary.get("Output token throughput (tok/s)")),
            "total_tps": num(summary.get("Total token throughput (tok/s)")),
            "avg_ttft_ms": ms(summary.get("Average time to first token (s)")),
            "avg_tpot_ms": ms(summary.get("Average time per output token (s)")),
            "avg_latency_ms": ms(summary.get("Average latency (s)")),
            "avg_itl_ms": ms(summary.get("Average inter-token latency (s)")),
            "avg_input_tokens": num(summary.get("Average input tokens per request")),
            "avg_output_tokens": num(summary.get("Average output tokens per request")),
            "args": args_data,
        }
        for label, suffix in [("50%", "50"), ("90%", "90"), ("95%", "95"), ("99%", "99")]:
            for src, dst, conv in [
                ("TTFT (s)", "ttft_ms", ms),
                ("TPOT (s)", "tpot_ms", ms),
                ("Latency (s)", "latency_ms", ms),
                ("ITL (s)", "itl_ms", ms),
                ("Input tokens", "input_tokens", num),
                ("Output tokens", "output_tokens", num),
                ("Output (tok/s)", "output_per_req_tps", num),
                ("Total (tok/s)", "total_per_req_tps", num),
            ]:
                run[f"p{suffix}_{dst}"] = conv(pct.get(src, {}).get(label))
        runs.append(run)
    return sorted(runs, key=lambda item: (item["scenario"], item.get("parallel") or 0, item["path"]))


def summarize_token_audit(results: Path) -> dict[str, Any]:
    summary = {"rows": 0, "usage_rows": 0, "tokenizer_rows": 0, "dual_rows": 0, "avg_prompt_delta": None, "avg_completion_delta": None}
    prompt_deltas: list[float] = []
    completion_deltas: list[float] = []
    for path in results.rglob("token_audit.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            summary["rows"] += 1
            if row.get("usage_prompt") is not None and row.get("usage_completion") is not None:
                summary["usage_rows"] += 1
            if row.get("tokenizer_prompt") is not None and row.get("tokenizer_completion") is not None:
                summary["tokenizer_rows"] += 1
            if row.get("prompt_delta") is not None or row.get("completion_delta") is not None:
                summary["dual_rows"] += 1
            if num(row.get("prompt_delta")) is not None:
                prompt_deltas.append(abs(float(row["prompt_delta"])))
            if num(row.get("completion_delta")) is not None:
                completion_deltas.append(abs(float(row["completion_delta"])))
    if prompt_deltas:
        summary["avg_prompt_delta"] = sum(prompt_deltas) / len(prompt_deltas)
    if completion_deltas:
        summary["avg_completion_delta"] = sum(completion_deltas) / len(completion_deltas)
    return summary


def classify_errors(run_path: Path) -> dict[str, int]:
    buckets = {
        "timeout": ["timeout", "timed out"],
        "rate_limit": ["429", "rate limit", "too many requests"],
        "auth": ["401", "403", "unauthorized", "forbidden", "api key"],
        "server_5xx": ["500", "502", "503", "504"],
        "client_4xx": ["400", "404", "422", "bad request"],
        "connect": ["connection refused", "connect", "dns", "resolve", "unreachable", "network"],
        "empty": ["empty response", "generated_text", "no data"],
    }
    counts = {key: 0 for key in buckets}
    for file_path in sorted(run_path.rglob("*"))[:30]:
        if not file_path.is_file() or file_path.name in {"benchmark_summary.json", "benchmark_percentile.json", "benchmark_args.json", "command_summary.json"}:
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")[:200000].lower()
        except Exception:
            continue
        for key, keywords in buckets.items():
            counts[key] += sum(text.count(keyword) for keyword in keywords)
    return {key: value for key, value in counts.items() if value > 0}


def report(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    results = Path(args.results_dir)
    output = Path(args.output)
    runs = collect_runs(results)
    unavailable = get(config, "token_accounting.mode") == "disabled" or get(config, "token_accounting.on_missing_usage") == "mark_unavailable"
    if not unavailable and runs:
        unavailable = all((run.get("avg_input_tokens") in {None, 0} and run.get("avg_output_tokens") in {None, 0}) for run in runs)
    token_audit = summarize_token_audit(results)
    eval_datasets = ", ".join(sorted({str(run.get("args", {}).get("dataset")) for run in runs if run.get("args", {}).get("dataset")})) or "-"
    eval_paths = ", ".join(sorted({str(run.get("args", {}).get("dataset_path")) for run in runs if run.get("args", {}).get("dataset_path")})) or "-"
    best = lambda key, reverse: next(iter(sorted([run for run in runs if run.get(key) is not None], key=lambda item: item[key], reverse=reverse)), None)
    lines = [
        "# Model Benchmark 压测报告",
        "",
        f"> {DISCLAIMER}",
        "",
        "## 1. 测试信息",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| 生成时间 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |",
        f"| 模型 | {get(config, 'model.name')} |",
        f"| API | {get(config, 'model.api')} |",
        f"| API URL | {get(config, 'model.api_url')} |",
        f"| 数据集名称 | {get(config, 'dataset.name')} |",
        f"| 数据集类型 | {get(config, 'dataset.type')} |",
        f"| 数据集路径 | {get(config, 'dataset.path', '-')} |",
        f"| EvalScope dataset | {eval_datasets} |",
        f"| EvalScope dataset_path | {eval_paths} |",
        f"| Token模式 | {get(config, 'token_accounting.mode')} |",
        f"| usage缺失策略 | {get(config, 'token_accounting.on_missing_usage')} |",
        f"| tokenizer来源 | {get(config, 'token_accounting.tokenizer_source')} |",
        f"| tokenizer路径 | {tokenizer_path(config) or '-'} |",
        f"| headers | {'已配置' if get(config, 'run.headers') else '未配置'} |",
        f"| connect/read/total timeout | {get(config, 'run.connect_timeout')}/{get(config, 'run.read_timeout')}/{get(config, 'run.total_timeout')} |",
        f"| warmup_requests | {get(config, 'run.warmup_requests')} |",
        f"| cooldown_seconds | {get(config, 'run.cooldown_seconds')} |",
        f"| 结果目录 | {results} |",
        "",
    ]
    if unavailable:
        lines += ["> Token 指标被禁用或被标记为不可计：TPOT、token 吞吐、输入/输出 token 统计不作为结论。", ""]
    lines += [
        "## 2. Token 计量说明",
        "",
        f"- 模式：`{get(config, 'token_accounting.mode')}`",
        f"- usage 缺失策略：`{get(config, 'token_accounting.on_missing_usage')}`",
        f"- dual compare 样本数：{token_audit['dual_rows']}",
        f"- usage 样本数：{token_audit['usage_rows']}",
        f"- tokenizer 样本数：{token_audit['tokenizer_rows']}",
        f"- 平均 prompt delta：{fmt(token_audit['avg_prompt_delta'])}",
        f"- 平均 completion delta：{fmt(token_audit['avg_completion_delta'])}",
        "",
        "## 3. 结论摘要",
        "",
    ]
    for text in [
        (f"QPS 峰值：`{best('qps', True)['scenario']}` 并发 {fmt(best('qps', True).get('parallel'), 0)}，{fmt(best('qps', True).get('qps'))} req/s。" if best("qps", True) else "未找到可用 QPS 结果。"),
        (f"最高成功率：`{best('success_rate_pct', True)['scenario']}` 并发 {fmt(best('success_rate_pct', True).get('parallel'), 0)}，{fmt(best('success_rate_pct', True).get('success_rate_pct'))}%。" if best("success_rate_pct", True) else "未找到可用成功率结果。"),
        (f"最低平均 E2E：`{best('avg_latency_ms', False)['scenario']}` 并发 {fmt(best('avg_latency_ms', False).get('parallel'), 0)}，{fmt(best('avg_latency_ms', False).get('avg_latency_ms'))} ms。" if best("avg_latency_ms", False) else "未找到可用延迟结果。"),
        "QPS/成功率/数据量是运行级指标；TTFT、TPOT、E2E、ITL、token 数和单请求吞吐提供分位数。",
    ]:
        lines.append(f"- {text}")
    lines += ["", "## 4. 数据量与成功率", "", "| 场景 | 并发 | 总请求 | 成功 | 失败 | 成功率(%) | 平均输入Tokens | 平均输出Tokens |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run_row in runs:
        lines.append(
            f"| {run_row['scenario']} | {fmt(run_row.get('parallel'), 0)} | {fmt(run_row.get('total_requests'), 0)} | {fmt(run_row.get('succeed_requests'), 0)} | {fmt(run_row.get('failed_requests'), 0)} | {fmt(run_row.get('success_rate_pct'))} | {fmt(run_row.get('avg_input_tokens'), unavailable=unavailable)} | {fmt(run_row.get('avg_output_tokens'), unavailable=unavailable)} |"
        )
    lines += ["", "## 5. QPS 与吞吐", "", "| 场景 | 并发 | QPS(req/s) | 输出吞吐(tok/s) | 总吞吐(tok/s) | 输出P90 | 输出P95 | 输出P99 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run_row in runs:
        lines.append(
            f"| {run_row['scenario']} | {fmt(run_row.get('parallel'), 0)} | {fmt(run_row.get('qps'))} | {fmt(run_row.get('output_tps'), unavailable=unavailable)} | {fmt(run_row.get('total_tps'), unavailable=unavailable)} | {fmt(run_row.get('p90_output_per_req_tps'), unavailable=unavailable)} | {fmt(run_row.get('p95_output_per_req_tps'), unavailable=unavailable)} | {fmt(run_row.get('p99_output_per_req_tps'), unavailable=unavailable)} |"
        )
    for title, prefix, blocked in [("TTFT", "ttft", False), ("TPOT", "tpot", unavailable), ("E2E", "latency", False), ("ITL", "itl", False)]:
        lines += ["", f"## 6.{['TTFT', 'TPOT', 'E2E', 'ITL'].index(title) + 1} {title}", "", "| 场景 | 并发 | 平均 | P50 | P90 | P95 | P99 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for run_row in runs:
            lines.append(
                f"| {run_row['scenario']} | {fmt(run_row.get('parallel'), 0)} | {fmt(run_row.get(f'avg_{prefix}_ms'), unavailable=blocked)} | {fmt(run_row.get(f'p50_{prefix}_ms'), unavailable=blocked)} | {fmt(run_row.get(f'p90_{prefix}_ms'), unavailable=blocked)} | {fmt(run_row.get(f'p95_{prefix}_ms'), unavailable=blocked)} | {fmt(run_row.get(f'p99_{prefix}_ms'), unavailable=blocked)} |"
            )
    lines += ["", "## 7. Token 统计", "", "| 场景 | 并发 | 输入平均 | 输入P50 | 输入P90 | 输出平均 | 输出P50 | 输出P90 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run_row in runs:
        lines.append(
            f"| {run_row['scenario']} | {fmt(run_row.get('parallel'), 0)} | {fmt(run_row.get('avg_input_tokens'), unavailable=unavailable)} | {fmt(run_row.get('p50_input_tokens'), unavailable=unavailable)} | {fmt(run_row.get('p90_input_tokens'), unavailable=unavailable)} | {fmt(run_row.get('avg_output_tokens'), unavailable=unavailable)} | {fmt(run_row.get('p50_output_tokens'), unavailable=unavailable)} | {fmt(run_row.get('p90_output_tokens'), unavailable=unavailable)} |"
        )
    lines += ["", "## 8. 错误摘要", "", "| 场景 | 并发 | 失败请求 | 错误分类 | 原始路径 |", "|---|---:|---:|---|---|"]
    failed_runs = [run_row for run_row in runs if (run_row.get("failed_requests") or 0) > 0]
    for run_row in failed_runs:
        categories = classify_errors(Path(run_row["path"]))
        category_text = ", ".join(f"{key}:{value}" for key, value in categories.items()) if categories else "未识别/需人工查看"
        lines.append(f"| {run_row['scenario']} | {fmt(run_row.get('parallel'), 0)} | {fmt(run_row.get('failed_requests'), 0)} | {category_text} | `{run_row['path']}` |")
    if not failed_runs:
        lines.append("| - | - | 0 | 未发现失败请求 | - |")
    lines += ["", "## 9. 原始结果路径", ""]
    lines += [f"- `{run_row['scenario']}` 并发 {fmt(run_row.get('parallel'), 0)}: `{run_row['path']}`" for run_row in runs] or ["- 未找到 `benchmark_summary.json`。"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written: {output}")
    return 0


def _perf_run(args: argparse.Namespace) -> int:
    payload = load_json(Path(args.payload))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid payload: {args.payload}")
    kwargs = payload["kwargs"]
    token_cfg = payload.get("token_accounting", {})
    audit_path = Path(payload.get("audit_path", ""))
    mode = str(token_cfg.get("mode", "prefer_api_usage"))
    missing = str(token_cfg.get("on_missing_usage", "fallback_tokenizer"))
    tokenizer_source = str(token_cfg.get("tokenizer_source", "modelscope"))
    local_evalscope = ROOT.parent.parent / "evalscope"
    if local_evalscope.exists() and str(local_evalscope) not in sys.path:
        sys.path.insert(0, str(local_evalscope))
    try:
        from evalscope.perf.arguments import Arguments
        from evalscope.perf.main import run_perf_benchmark
        from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
        from evalscope.perf.plugin.registry import register_api
    except Exception as exc:
        raise SystemExit(f"EvalScope is not available. Run bootstrap first. Details: {exc}") from exc

    def extract_usage(responses: list[dict[str, Any]]) -> tuple[int, int] | None:
        for response in reversed(responses):
            if not isinstance(response, dict):
                continue
            usage = response.get("usage")
            if isinstance(usage, dict) and usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None:
                return int(usage["prompt_tokens"]), int(usage["completion_tokens"])
        return None

    @register_api("openai_skill_token")
    class OpenAISkillTokenPlugin(OpenaiPlugin):  # type: ignore
        def __init__(self, param):  # noqa: ANN001
            original_path = getattr(param, "tokenizer_path", None)
            if tokenizer_source == "huggingface" and original_path:
                param.tokenizer_path = None
                super().__init__(param)
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(original_path, trust_remote_code=True)
                param.tokenizer_path = original_path
            else:
                super().__init__(param)

        def parse_responses(self, responses, request=None, **kwargs_):  # noqa: ANN001
            usage_counts = extract_usage(responses)
            tokenizer_counts = None
            tokenizer_error = None
            if self.tokenizer is not None:
                try:
                    tokenizer_counts = super()._count_input_tokens(request), 0
                    from collections import defaultdict

                    content = defaultdict(list)
                    for response in responses:
                        if not isinstance(response, dict):
                            continue
                        if response.get("object") == "chat.completion":
                            for choice in response.get("choices", []) or []:
                                content[choice.get("index", 0)].append((choice.get("message") or {}).get("content") or "")
                        elif response.get("object") == "chat.completion.chunk":
                            for choice in response.get("choices", []) or []:
                                content[choice.get("index", 0)].append(((choice.get("delta") or {}).get("content")) or "")
                        elif response.get("object") == "text_completion":
                            for choice in response.get("choices", []) or []:
                                content[choice.get("index", 0)].append(choice.get("text") or "")
                    tokenizer_counts = tokenizer_counts[0], sum(self._count_output_tokens("".join(parts)) for parts in content.values())
                except Exception as exc:
                    tokenizer_error = str(exc)
            if audit_path:
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                prompt_delta = None
                completion_delta = None
                if usage_counts and tokenizer_counts:
                    prompt_delta = usage_counts[0] - tokenizer_counts[0]
                    completion_delta = usage_counts[1] - tokenizer_counts[1]
                row = {
                    "mode": mode,
                    "usage_prompt": usage_counts[0] if usage_counts else None,
                    "usage_completion": usage_counts[1] if usage_counts else None,
                    "tokenizer_prompt": tokenizer_counts[0] if tokenizer_counts else None,
                    "tokenizer_completion": tokenizer_counts[1] if tokenizer_counts else None,
                    "prompt_delta": prompt_delta,
                    "completion_delta": completion_delta,
                    "tokenizer_error": tokenizer_error,
                }
                with audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if mode == "disabled":
                return 0, 0
            if mode == "tokenizer_only":
                if tokenizer_counts:
                    return tokenizer_counts
                if missing == "mark_unavailable":
                    return 0, 0
                raise ValueError(tokenizer_error or "tokenizer_only mode requires tokenizer_path.")
            if mode in {"prefer_api_usage", "api_usage_only", "dual_compare"} and usage_counts:
                return usage_counts
            if mode == "api_usage_only":
                if missing == "mark_unavailable":
                    return 0, 0
                if missing == "fallback_tokenizer" and tokenizer_counts:
                    return tokenizer_counts
                raise ValueError("API response does not contain usage.")
            if tokenizer_counts and missing == "fallback_tokenizer":
                return tokenizer_counts
            if missing == "mark_unavailable":
                return 0, 0
            raise ValueError(tokenizer_error or "Unable to determine token counts.")

        async def process_request(self, client_session, url, headers, body):  # noqa: ANN001
            output = await super().process_request(client_session, url, headers, body)
            if output.success and not (output.generated_text or "").strip():
                output.success = False
                output.error = "Empty response body or empty generated_text."
            return output

    if kwargs.get("api") == "openai":
        kwargs["api"] = "openai_skill_token"
    run_perf_benchmark(Arguments(**kwargs))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model_benchmark.py")
    sub = parser.add_subparsers(dest="command")
    for name in ("doctor", "bootstrap"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub.choices["doctor"].add_argument("--check-endpoint", action="store_true")
    cmd = sub.add_parser("menu")
    cmd.add_argument("--config", default=str(DEFAULT_CONFIG))
    cmd.add_argument("--output", default="configs/model_benchmark.local.yaml")
    cmd = sub.add_parser("run")
    cmd.add_argument("--config", default=str(DEFAULT_CONFIG))
    cmd.add_argument("--scenario", default="smoke")
    cmd.add_argument("--run-id")
    cmd.add_argument("--dry-run", action="store_true")
    cmd = sub.add_parser("report")
    cmd.add_argument("--config", default=str(DEFAULT_CONFIG))
    cmd.add_argument("--results-dir", required=True)
    cmd.add_argument("--output", required=True)
    cmd = sub.add_parser("_perf-run")
    cmd.add_argument("--payload", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        argv = ["menu"]
    args = build_parser().parse_args(argv)
    return globals()[args.command.replace("-", "_")](args)


if __name__ == "__main__":
    raise SystemExit(main())
