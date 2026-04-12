#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified model performance benchmark helper for the model-benchmark skill.

The public commands intentionally depend only on Python's standard library so
`doctor`, `bootstrap`, and `menu` can run on a nearly empty macOS or Ubuntu host.
EvalScope is imported only by the private `_evalscope-run` command that executes
inside the configured virtual environment.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_benchmark.example.yaml"
DEFAULT_DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
SECRET_KEYS = {"api_key", "api-key", "authorization", "token", "secret", "password"}


DEFAULT_CONFIG: Dict[str, Any] = {
    "environment": {
        "venv_path": ".venv-model-benchmark",
        "env_file": ".model_benchmark.env",
        "install_source": "pip",
        "evalscope_extras": "perf",
        "pip_index_url": "https://pypi.tuna.tsinghua.edu.cn/simple",
        "pip_trusted_host": "pypi.tuna.tsinghua.edu.cn",
        "modelscope_cache": ".cache/modelscope",
        "hf_endpoint": "https://hf-mirror.com",
    },
    "model": {
        "api": "openai",
        "name": "qwen3.6-plus",
        "api_url": DEFAULT_DASHSCOPE_URL,
        "api_key_env": "DASHSCOPE_API_KEY",
        "tokenizer_path": "Qwen/Qwen3-0.6B",
    },
    "dataset": {
        "type": "simulated",
        "path": "outputs/simulated_openqa.jsonl",
        "simulated_count": 12,
        "simulated_prompt_chars": 240,
        "random_prompt_tokens": 512,
        "output_tokens": 128,
        "prefix_length": 0,
    },
    "token_accounting": {
        "mode": "auto",
        "on_missing_usage": "fallback_tokenizer",
    },
    "targets": {
        "success_rate_pct": 99,
        "qps": None,
        "output_tps": None,
        "avg_ttft_ms": None,
        "p95_ttft_ms": None,
        "p99_ttft_ms": None,
        "avg_tpot_ms": None,
        "avg_latency_ms": None,
        "p95_latency_ms": None,
        "p99_latency_ms": None,
    },
    "scenarios": {
        "smoke": {"enabled": True, "parallel": 1, "number": 3, "max_tokens": 32},
        "gradient": {
            "enabled": True,
            "parallels": [1, 2, 5, 8, 10, 15, 20],
            "number_multiplier": 10,
            "min_number": 50,
            "max_tokens": 128,
            "sleep_interval": 5,
        },
        "sla": {
            "enabled": False,
            "variable": "parallel",
            "lower_bound": 1,
            "upper_bound": 100,
            "number_multiplier": 5,
            "num_runs": 3,
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
            "input_tokens": [100, 500, 1000, 2000, 4000],
            "output_tokens": [32, 128, 256, 512],
        },
    },
    "run": {
        "outputs_dir": "outputs/model_benchmark",
        "connect_timeout": 600,
        "read_timeout": 600,
        "total_timeout": 21600,
        "stream": True,
        "temperature": 0.0,
        "seed": 42,
        "log_every_n_query": 50,
        "enable_progress_tracker": True,
    },
}


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    out = []
    for char in line:
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\":
            out.append(char)
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            break
        out.append(char)
    return "".join(out).rstrip()


def split_top_level(value: str, delimiter: str = ",") -> List[str]:
    parts: List[str] = []
    depth = 0
    in_single = False
    in_double = False
    start = 0
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if char in "[{(":
                depth += 1
            elif char in "]})":
                depth -= 1
            elif char == delimiter and depth == 0:
                parts.append(value[start:index].strip())
                start = index + 1
    parts.append(value[start:].strip())
    return [part for part in parts if part != ""]


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return [parse_scalar(item) for item in split_top_level(inner)]
    if value.startswith("{") and value.endswith("}"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            inner = value[1:-1].strip()
            if not inner:
                return {}
            parsed: Dict[str, Any] = {}
            for item in split_top_level(inner):
                if ":" not in item:
                    continue
                key, val = item.split(":", 1)
                parsed[key.strip().strip("'\"")] = parse_scalar(val)
            return parsed
    try:
        if value.startswith("0") and value not in {"0", "0.0"} and not value.startswith("0."):
            return value
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def minimal_yaml_load(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        cleaned = strip_comment(raw_line)
        if not cleaned.strip():
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        line = cleaned.strip()
        if line.startswith("- "):
            raise ConfigError("Block-style YAML lists are not supported without PyYAML; use inline lists like [1, 2].")
        if ":" not in line:
            raise ConfigError(f"Invalid YAML line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)
    return root


def read_yaml_or_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ConfigError("Top-level config must be a mapping.")
        return data
    except ModuleNotFoundError:
        return minimal_yaml_load(text)


def format_yaml_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(format_yaml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    text = str(value)
    if text == "" or any(char in text for char in [":", "#", "[", "]", "{", "}", ","]) or text.lower() in {
        "true",
        "false",
        "null",
        "none",
    }:
        return json.dumps(text, ensure_ascii=False)
    return text


def dump_simple_yaml(data: Dict[str, Any], indent: int = 0) -> str:
    lines: List[str] = []
    prefix = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(dump_simple_yaml(value, indent + 2))
        else:
            lines.append(f"{prefix}{key}: {format_yaml_value(value)}")
    return "\n".join(line for line in lines if line != "") + ("\n" if indent == 0 else "")


def load_config(path: Path | str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    loaded = read_yaml_or_json(config_path)
    config = deep_merge(DEFAULT_CONFIG, loaded)
    validate_config(config)
    return config


def load_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def configured_env_file(config: Dict[str, Any]) -> Optional[Path]:
    env_file = config.get("environment", {}).get("env_file")
    return project_path(env_file) if env_file else None


def get_secret_from_env_or_file(config: Dict[str, Any], env_name: str) -> str:
    if os.environ.get(env_name):
        return os.environ[env_name]
    env_file = configured_env_file(config)
    if env_file:
        return load_env_file(env_file).get(env_name, "")
    return ""


def write_env_secret(env_file: Path, env_name: str, secret: str) -> None:
    env_file.parent.mkdir(parents=True, exist_ok=True)
    existing = load_env_file(env_file)
    existing[env_name] = secret
    lines = [f"{key}={value}" for key, value in sorted(existing.items())]
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        env_file.chmod(0o600)
    except OSError:
        pass


def ensure_env_placeholder(env_file: Path, env_name: str) -> None:
    if get_secret_from_env_or_file({"environment": {"env_file": str(env_file)}}, env_name):
        return
    env_file.parent.mkdir(parents=True, exist_ok=True)
    existing = load_env_file(env_file)
    if env_name not in existing:
        with env_file.open("a", encoding="utf-8") as handle:
            handle.write(f"{env_name}=\n")
    try:
        env_file.chmod(0o600)
    except OSError:
        pass


def validate_config(config: Dict[str, Any]) -> None:
    model = config.get("model", {})
    dataset = config.get("dataset", {})
    token_accounting = config.get("token_accounting", {})

    required_model_fields = ["api", "name", "api_url", "api_key_env"]
    missing = [field for field in required_model_fields if not model.get(field)]
    if missing:
        raise ConfigError(f"Missing model fields: {', '.join(missing)}")

    dataset_type = dataset.get("type")
    allowed_datasets = {"simulated", "openqa", "line_by_line", "random"}
    if dataset_type not in allowed_datasets:
        raise ConfigError(f"dataset.type must be one of {sorted(allowed_datasets)}, got {dataset_type!r}")

    mode = token_accounting.get("mode")
    if mode not in {"api_usage", "tokenizer", "auto"}:
        raise ConfigError("token_accounting.mode must be api_usage, tokenizer, or auto")

    on_missing = token_accounting.get("on_missing_usage")
    if on_missing not in {"fail", "fallback_tokenizer", "skip_token_metrics"}:
        raise ConfigError("token_accounting.on_missing_usage must be fail, fallback_tokenizer, or skip_token_metrics")


def python_version_ok(executable: str) -> bool:
    try:
        proc = subprocess.run(
            [executable, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    if proc.returncode != 0:
        return False
    major, minor = [int(part) for part in proc.stdout.strip().split(".")[:2]]
    return (major, minor) >= (3, 10)


def find_python() -> Optional[str]:
    candidates = ["python3.12", "python3.11", "python3.10", "python3"]
    for candidate in candidates:
        path = shutil.which(candidate)
        if path and python_version_ok(path):
            return path
    if python_version_ok(sys.executable):
        return sys.executable
    return None


def venv_python(config: Dict[str, Any]) -> Path:
    venv_path = project_path(config["environment"]["venv_path"])
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def active_python(config: Dict[str, Any]) -> str:
    venv_py = venv_python(config)
    if venv_py.exists():
        return str(venv_py)
    found = find_python()
    if not found:
        raise ConfigError("No Python >= 3.10 found. Run bootstrap or install Python first.")
    return found


def mask_secret(value: str, visible: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= visible * 2:
        return "*" * len(value)
    return value[:visible] + "*" * (len(value) - visible * 2) + value[-visible:]


def sanitize_obj(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key.lower().replace("_", "-") in SECRET_KEYS:
                sanitized[key] = mask_secret(str(item))
            else:
                sanitized[key] = sanitize_obj(item)
        return sanitized
    if isinstance(value, list):
        result = []
        skip_next = False
        for index, item in enumerate(value):
            if skip_next:
                result.append(mask_secret(str(item)))
                skip_next = False
                continue
            if isinstance(item, str) and item.lower() in {"--api-key", "api-key", "authorization"}:
                result.append(item)
                skip_next = True
            else:
                result.append(sanitize_obj(item))
        return result
    return value


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def endpoint_root(url: str) -> str:
    lowered = url.rstrip("/")
    for suffix in ["/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions"]:
        if lowered.endswith(suffix):
            return lowered[: -len(suffix)]
    return lowered


def check_endpoint(url: str, timeout: int = 5) -> Tuple[bool, str]:
    target = endpoint_root(url)
    request = urllib.request.Request(target, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return True, f"reachable ({response.status})"
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403, 404, 405}:
            return True, f"reachable (HTTP {exc.code})"
        return False, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - diagnostic command
        return False, str(exc)


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    py = find_python()
    venv_py = venv_python(config)
    api_key_env = config["model"]["api_key_env"]
    api_key_present = bool(get_secret_from_env_or_file(config, api_key_env))
    evalscope_check_python = str(venv_py) if venv_py.exists() else (py or sys.executable)

    evalscope_version = "not installed"
    if Path(evalscope_check_python).exists():
        proc = subprocess.run(
            [evalscope_check_python, "-m", "evalscope.cli.cli", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            evalscope_version = (proc.stdout or proc.stderr).strip() or "installed"
        else:
            proc = subprocess.run(
                [evalscope_check_python, "-c", "import evalscope; print(getattr(evalscope, '__version__', 'installed'))"],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                evalscope_version = proc.stdout.strip() or "installed"

    report = {
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "selected": py,
            "version_ok": bool(py),
            "venv_python": str(venv_py),
            "venv_exists": venv_py.exists(),
        },
        "package_tools": {
            "pip": bool(py),
            "apt": command_exists("apt-get"),
            "brew": command_exists("brew"),
        },
        "evalscope": evalscope_version,
        "network": {
            "pip_index_url": config["environment"]["pip_index_url"],
            "hf_endpoint": config["environment"]["hf_endpoint"],
        },
        "credentials": {
            "api_key_env": api_key_env,
            "env_file": str(configured_env_file(config)) if configured_env_file(config) else None,
            "present": api_key_present,
        },
    }

    if args.check_endpoint:
        ok, message = check_endpoint(config["model"]["api_url"], timeout=args.timeout)
        report["endpoint"] = {"url": config["model"]["api_url"], "reachable": ok, "message": message}

    print_json(report)
    if not py or (args.require_api_key and not api_key_present):
        return 1
    return 0


def run_subprocess(cmd: Sequence[str], dry_run: bool = False, env: Optional[Dict[str, str]] = None) -> int:
    safe_cmd = sanitize_obj(list(cmd))
    print("$ " + " ".join(str(part) for part in safe_cmd))
    if dry_run:
        return 0
    proc = subprocess.run(list(cmd), env=env, check=False)
    return proc.returncode


def run_bootstrap(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    py = find_python()
    if not py:
        system = platform.system()
        if system == "Darwin":
            print("未找到 Python >= 3.10。macOS 建议先安装 Homebrew 后执行: brew install python@3.12", file=sys.stderr)
        elif system == "Linux" and command_exists("apt-get"):
            print(
                "未找到 Python >= 3.10。Ubuntu 建议执行: sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip",
                file=sys.stderr,
            )
        else:
            print("未找到 Python >= 3.10，请先安装 Python。", file=sys.stderr)
        return 1

    venv_py = venv_python(config)
    venv_dir = venv_py.parents[1]
    if not venv_py.exists():
        rc = run_subprocess([py, "-m", "venv", str(venv_dir)], dry_run=args.dry_run)
        if rc != 0:
            return rc

    pip_cmd = [str(venv_py), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"]
    index_url = config["environment"].get("pip_index_url")
    trusted_host = config["environment"].get("pip_trusted_host")
    if index_url:
        pip_cmd.extend(["-i", index_url])
    if trusted_host:
        pip_cmd.extend(["--trusted-host", trusted_host])
    rc = run_subprocess(pip_cmd, dry_run=args.dry_run)
    if rc != 0:
        return rc

    install_source = config["environment"].get("install_source", "pip")
    extras = str(config["environment"].get("evalscope_extras") or "perf").strip()
    package_name = f"evalscope[{extras}]" if extras else "evalscope"
    if install_source == "local" and (PROJECT_ROOT / "evalscope" / "pyproject.toml").exists():
        package_spec = str(PROJECT_ROOT / "evalscope") + (f"[{extras}]" if extras else "")
        install_cmd = [str(venv_py), "-m", "pip", "install", "-e", package_spec]
    else:
        install_cmd = [str(venv_py), "-m", "pip", "install", "-U", package_name]
    if index_url:
        install_cmd.extend(["-i", index_url])
    if trusted_host:
        install_cmd.extend(["--trusted-host", trusted_host])
    return run_subprocess(install_cmd, dry_run=args.dry_run)


def prompt_value(label: str, default: Any) -> Any:
    default_text = "" if default is None else str(default)
    try:
        raw = input(f"{label} [{default_text}]: ").strip()
    except EOFError:
        return default
    if raw == "":
        return default
    return parse_scalar(raw)


def prompt_text(label: str, default: Optional[str] = None) -> str:
    value = prompt_value(label, default or "")
    return "" if value is None else str(value).strip()


def prompt_number(label: str, default: Any, cast=float, allow_null: bool = False) -> Any:
    while True:
        value = prompt_value(label, default)
        if allow_null and value in {"", None, "null", "none"}:
            return None
        try:
            return cast(value)
        except (TypeError, ValueError):
            print("请输入数字；直接回车使用默认值。")


def prompt_optional_number(label: str, default: Any = None) -> Optional[float]:
    value = prompt_value(label, default)
    if value in {"", None, "null", "none"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        print("输入不是数字，已按未配置处理。")
        return None


def prompt_yes_no(label: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    try:
        raw = input(f"{label} [{suffix}]: ").strip().lower()
    except EOFError:
        return default
    if raw == "":
        return default
    return raw in {"y", "yes", "1", "true", "是", "好", "启用"}


def choose_option(title: str, options: List[Tuple[str, str, str]], default_value: str) -> str:
    print(f"\n{title}")
    for idx, (_value, label, description) in enumerate(options, 1):
        default_mark = "（默认）" if _value == default_value else ""
        print(f"  {idx}. {label}{default_mark} - {description}")
    default_idx = next((idx for idx, item in enumerate(options, 1) if item[0] == default_value), 1)
    while True:
        try:
            raw = input(f"请选择 1-{len(options)} [{default_idx}]: ").strip()
        except EOFError:
            return default_value
        if raw == "":
            return default_value
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        print("无效选项，请输入编号。")


def parse_int_values(value: Any) -> List[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            parsed = parse_scalar(text)
            return parse_int_values(parsed)
        return [int(item.strip()) for item in text.split(",") if item.strip()]
    return [int(value)]


def unique_sorted(values: Iterable[int]) -> List[int]:
    return sorted({int(value) for value in values if int(value) > 0})


def build_parallel_values(mode: str, current: List[int], start: int, end: int, step: int = 1, count: int = 5, multiplier: float = 2.0) -> List[int]:
    if mode == "custom":
        return unique_sorted(current)
    if mode == "step":
        if step <= 0:
            step = 1
        return list(range(start, end + 1, step))
    if mode == "count":
        if count <= 1:
            return [start]
        if end < start:
            start, end = end, start
        span = end - start
        return unique_sorted(round(start + span * idx / (count - 1)) for idx in range(count))
    if mode == "multiply":
        values = []
        cur = max(1, start)
        multiplier = max(1.1, multiplier)
        while cur <= end:
            values.append(cur)
            nxt = int(math.ceil(cur * multiplier))
            if nxt <= cur:
                nxt = cur + 1
            cur = nxt
        if values[-1] != end:
            values.append(end)
        return unique_sorted(values)
    return unique_sorted(current)


def configure_dataset(base: Dict[str, Any]) -> None:
    dataset_options = [
        ("simulated", "内置模拟数据", "首次冒烟和流程验证推荐；自动生成 openqa JSONL，不需要外部下载。"),
        ("openqa", "自定义 OpenQA JSONL", "文件每行 JSON，包含 question 字段；最贴近真实业务问答。"),
        ("line_by_line", "逐行 TXT", "文件每行一个 prompt；适合快速拿业务问题清单压测。"),
        ("convert", "自动转换 JSONL/TXT", "支持 messages、question、text、prompt 或 TXT，转换成 openqa 后压测。"),
        ("random", "随机 token 数据", "用于固定输入长度压测；必须配置 tokenizer。"),
    ]
    selected = choose_option("数据集类型", dataset_options, base["dataset"].get("type", "simulated"))
    if selected == "simulated":
        base["dataset"]["type"] = "simulated"
        base["dataset"]["path"] = prompt_text("模拟数据输出路径", base["dataset"].get("path", "outputs/simulated_openqa.jsonl"))
        base["dataset"]["simulated_count"] = int(prompt_number("模拟样本条数", base["dataset"].get("simulated_count", 12), int))
        base["dataset"]["simulated_prompt_chars"] = int(prompt_number("每条模拟 prompt 约多少字符", base["dataset"].get("simulated_prompt_chars", 240), int))
        return

    if selected == "openqa":
        base["dataset"]["type"] = "openqa"
        base["dataset"]["path"] = prompt_text("OpenQA JSONL 路径（每行含 question 字段）", base["dataset"].get("path", ""))
        return

    if selected == "line_by_line":
        base["dataset"]["type"] = "line_by_line"
        while True:
            path = prompt_text("TXT 数据集路径（每行一个 prompt，必填）", base["dataset"].get("path", ""))
            if path:
                base["dataset"]["path"] = path
                return
            print("line_by_line 模式必须提供数据集路径。")

    if selected == "convert":
        source = prompt_text("源数据路径（JSONL 或 TXT）", "")
        fmt = choose_option(
            "源数据格式",
            [
                ("auto", "自动识别", "根据首行字段自动判断 messages/openqa/text/TXT。"),
                ("messages", "OpenAI messages", "每行包含 messages 数组。"),
                ("openqa", "OpenQA question", "每行包含 question 字段。"),
                ("text", "text/prompt", "每行包含 text 或 prompt 字段。"),
            ],
            "auto",
        )
        output = prompt_text("转换后 openqa 输出路径", "outputs/converted_openqa.jsonl")
        if source and Path(source).expanduser().exists():
            total, success, failed = convert_to_openqa(Path(source).expanduser(), project_path(output), fmt)
            print(f"已转换数据集：总计 {total}，成功 {success}，失败 {failed}。")
        else:
            print("源文件暂不存在，已仅写入目标配置；运行前请先转换或补齐文件。")
        base["dataset"]["type"] = "openqa"
        base["dataset"]["path"] = output
        return

    base["dataset"]["type"] = "random"
    if not base["model"].get("tokenizer_path"):
        base["model"]["tokenizer_path"] = prompt_text("random 数据集必须填写 tokenizer 路径或 ModelScope ID", "Qwen/Qwen3-0.6B")
    base["dataset"]["random_prompt_tokens"] = int(prompt_number("随机输入长度 tokens", base["dataset"].get("random_prompt_tokens", 512), int))
    base["dataset"]["prefix_length"] = int(prompt_number("固定 prefix 长度 tokens", base["dataset"].get("prefix_length", 0), int))


def configure_targets(base: Dict[str, Any]) -> None:
    print("\n目标指标（可直接回车跳过；报告会展示目标差距）")
    base["targets"]["success_rate_pct"] = float(prompt_number("目标成功率(%)", base["targets"].get("success_rate_pct", 99), float))
    base["targets"]["qps"] = prompt_optional_number("目标 QPS(req/s)", base["targets"].get("qps"))
    base["targets"]["output_tps"] = prompt_optional_number("目标输出吞吐(tok/s)", base["targets"].get("output_tps"))
    base["targets"]["avg_ttft_ms"] = prompt_optional_number("目标平均 TTFT(ms)", base["targets"].get("avg_ttft_ms"))
    base["targets"]["p95_ttft_ms"] = prompt_optional_number("目标 P95 TTFT(ms)", base["targets"].get("p95_ttft_ms"))
    base["targets"]["p99_ttft_ms"] = prompt_optional_number("目标 P99 TTFT(ms)", base["targets"].get("p99_ttft_ms"))
    base["targets"]["avg_tpot_ms"] = prompt_optional_number("目标平均 TPOT(ms)", base["targets"].get("avg_tpot_ms"))
    base["targets"]["avg_latency_ms"] = prompt_optional_number("目标平均端到端延迟(ms)", base["targets"].get("avg_latency_ms"))
    base["targets"]["p95_latency_ms"] = prompt_optional_number("目标 P95 端到端延迟(ms)", base["targets"].get("p95_latency_ms"))
    base["targets"]["p99_latency_ms"] = prompt_optional_number("目标 P99 端到端延迟(ms)", base["targets"].get("p99_latency_ms"))


def configure_gradient(base: Dict[str, Any]) -> None:
    gradient = base["scenarios"]["gradient"]
    gradient["enabled"] = prompt_yes_no("启用并发梯度压测", bool(gradient.get("enabled", True)))
    if not gradient["enabled"]:
        return

    current = numeric_list(gradient.get("parallels", [1, 2, 5, 8, 10, 15, 20]))
    mode = choose_option(
        "并发梯度生成方式",
        [
            ("default", "使用推荐列表", f"当前默认 {current}，适合首次探测。"),
            ("custom", "手动输入列表", "例如 1,2,5,10,20,50。"),
            ("step", "起止范围 + 固定步长", "例如 1 到 100，每 5 个并发测一次。"),
            ("count", "起止范围 + 测试档位数量", "例如 1 到 100，共 8 档，自动均匀取点。"),
            ("multiply", "起点 + 终点 + 倍增系数", "例如 1 开始，每次 x2，直到 128。"),
        ],
        "default",
    )
    if mode == "custom":
        current = parse_int_values(prompt_value("并发列表，逗号分隔", ",".join(str(item) for item in current)))
    elif mode in {"step", "count", "multiply"}:
        start = int(prompt_number("起始并发", current[0], int))
        end = int(prompt_number("结束并发", current[-1], int))
        if mode == "step":
            step = int(prompt_number("并发步长", 5, int))
            current = build_parallel_values("step", current, start, end, step=step)
        elif mode == "count":
            count = int(prompt_number("测试档位数量", len(current), int))
            current = build_parallel_values("count", current, start, end, count=count)
        else:
            multiplier = float(prompt_number("倍增系数", 2.0, float))
            current = build_parallel_values("multiply", current, start, end, multiplier=multiplier)
    gradient["parallels"] = current

    request_mode = choose_option(
        "每个并发档位请求数",
        [
            ("formula", "按公式计算", "number = max(最小请求数, 并发 * 倍数)。"),
            ("fixed", "每档固定请求数", "所有并发档位使用相同 number。"),
            ("custom", "手动输入配对列表", "请求数列表必须和并发列表长度一致。"),
        ],
        "formula",
    )
    gradient.pop("numbers", None)
    if request_mode == "formula":
        gradient["number_multiplier"] = int(prompt_number("请求数倍数", gradient.get("number_multiplier", 10), int))
        gradient["min_number"] = int(prompt_number("每档最小请求数", gradient.get("min_number", 50), int))
    elif request_mode == "fixed":
        fixed = int(prompt_number("每档固定请求数", gradient.get("min_number", 50), int))
        gradient["numbers"] = [fixed for _ in current]
        gradient["min_number"] = fixed
    else:
        while True:
            numbers = parse_int_values(prompt_value("请求数列表，逗号分隔", ",".join(str(max(50, p * 10)) for p in current)))
            if len(numbers) == len(current):
                gradient["numbers"] = numbers
                break
            print("请求数列表长度必须和并发列表一致。")

    gradient["max_tokens"] = int(prompt_number("每次请求 max_tokens", gradient.get("max_tokens", 128), int))
    gradient["sleep_interval"] = int(prompt_number("并发档位之间等待秒数", gradient.get("sleep_interval", 5), int))


def configure_optional_scenarios(base: Dict[str, Any]) -> None:
    smoke = base["scenarios"]["smoke"]
    print("\n连接验证 / 小样本试跑")
    print("说明：这里只发少量请求，用来确认 API Key、URL、模型名、返回格式和 token 计量策略正常，避免正式压测跑到一半才失败。")
    smoke["enabled"] = prompt_yes_no("启用连接验证/小样本试跑（推荐）", bool(smoke.get("enabled", True)))
    if smoke["enabled"]:
        smoke["parallel"] = int(prompt_number("冒烟并发", smoke.get("parallel", 1), int))
        smoke["number"] = int(prompt_number("冒烟请求数", smoke.get("number", 3), int))
        smoke["max_tokens"] = int(prompt_number("冒烟 max_tokens", smoke.get("max_tokens", 32), int))

    configure_gradient(base)

    sla = base["scenarios"]["sla"]
    sla["enabled"] = prompt_yes_no("启用 SLA 自动调优", bool(sla.get("enabled", False)))
    if sla["enabled"]:
        sla["variable"] = choose_option(
            "SLA 调优变量",
            [("parallel", "并发数", "寻找满足目标的最大并发。"), ("rate", "请求速率", "寻找满足目标的最大请求速率。")],
            sla.get("variable", "parallel"),
        )
        sla["lower_bound"] = int(prompt_number("SLA 搜索下界", sla.get("lower_bound", 1), int))
        sla["upper_bound"] = int(prompt_number("SLA 搜索上界", sla.get("upper_bound", 100), int))
        sla["number_multiplier"] = float(prompt_number("SLA 每档请求数倍数", sla.get("number_multiplier", 5), float))
        sla["num_runs"] = int(prompt_number("SLA 每档重复次数", sla.get("num_runs", 3), int))

    stability = base["scenarios"]["stability"]
    stability["enabled"] = prompt_yes_no("启用稳定性测试", bool(stability.get("enabled", False)))
    if stability["enabled"]:
        stability["parallel"] = int(prompt_number("稳定性测试并发", stability.get("parallel", 10), int))
        stability["duration_minutes"] = float(prompt_number("稳定性测试总时长(分钟)", stability.get("duration_minutes", 30), float))
        stability["window_minutes"] = float(prompt_number("采样窗口(分钟)", stability.get("window_minutes", 5), float))
        stability["max_tokens"] = int(prompt_number("稳定性 max_tokens", stability.get("max_tokens", 128), int))

    matrix = base["scenarios"]["length_matrix"]
    matrix["enabled"] = prompt_yes_no("启用输入/输出长度矩阵测试", bool(matrix.get("enabled", False)))
    if matrix["enabled"]:
        if not base["model"].get("tokenizer_path"):
            base["model"]["tokenizer_path"] = prompt_text("长度矩阵使用 random 数据集，必须填写 tokenizer 路径或 ModelScope ID", "Qwen/Qwen3-0.6B")
        matrix["parallel"] = int(prompt_number("长度矩阵并发", matrix.get("parallel", 5), int))
        matrix["number"] = int(prompt_number("每个长度组合请求数", matrix.get("number", 100), int))
        matrix["input_tokens"] = parse_int_values(prompt_value("输入长度 tokens 列表", matrix.get("input_tokens", [100, 500, 1000])))
        matrix["output_tokens"] = parse_int_values(prompt_value("输出 max_tokens 列表", matrix.get("output_tokens", [32, 128, 256])))


def run_menu(args: argparse.Namespace) -> int:
    base = copy.deepcopy(DEFAULT_CONFIG)
    if args.base_config and Path(args.base_config).exists():
        base = load_config(args.base_config)

    print("Model Benchmark 配置向导")
    print("说明：方括号中是默认值，直接回车会使用默认；固定选项请输入编号。")
    print("\n模型连接")
    base["model"]["name"] = prompt_text("模型名称", base["model"]["name"])
    base["model"]["api_url"] = prompt_text("OpenAI-compatible API URL", base["model"]["api_url"])
    base["environment"]["env_file"] = base["environment"].get("env_file") or ".model_benchmark.env"
    print(f"API Key 默认从本地 `{base['environment']['env_file']}` 的 `{base['model']['api_key_env']}` 读取，不会写入 YAML。")
    secret = ""

    base["token_accounting"]["mode"] = choose_option(
        "Token 计量模式",
        [
            ("auto", "自动", "优先使用 API usage；缺失时按下面策略处理。"),
            ("api_usage", "只用 API usage", "最准确，但服务不返回 usage 时可能失败或跳过 token 指标。"),
            ("tokenizer", "使用 tokenizer 估算", "适合 API 不返回 usage；需要 tokenizer 路径或 ModelScope ID。"),
        ],
        base["token_accounting"].get("mode", "auto"),
    )
    base["token_accounting"]["on_missing_usage"] = choose_option(
        "API usage 缺失时怎么办",
        [
            ("fallback_tokenizer", "回退 tokenizer", "推荐；需要 tokenizer_path，报告中 token 指标仍可用。"),
            ("skip_token_metrics", "跳过 token 指标", "只看 QPS/TTFT/E2E；报告会标注 token/TPOT 不作为结论。"),
            ("fail", "失败并停止", "适合强制要求 API 返回 usage 的验收。"),
        ],
        base["token_accounting"]["on_missing_usage"],
    )
    needs_tokenizer = (
        base["token_accounting"]["mode"] == "tokenizer"
        or base["token_accounting"]["on_missing_usage"] == "fallback_tokenizer"
    )
    if needs_tokenizer:
        base["model"]["tokenizer_path"] = prompt_text("Tokenizer 路径或 ModelScope ID（当前策略建议填写）", base["model"].get("tokenizer_path"))
    else:
        base["model"]["tokenizer_path"] = None
        print("当前 token 计量策略不需要 tokenizer，已跳过 tokenizer 配置。")

    configure_dataset(base)
    configure_targets(base)
    configure_optional_scenarios(base)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dump_simple_yaml(base), encoding="utf-8")
    if secret:
        env_file = configured_env_file(base)
        if env_file:
            write_env_secret(env_file, base["model"]["api_key_env"], secret)
            print(f"API Key 已保存到: {env_file} (0600)")
    else:
        env_file = configured_env_file(base)
        if env_file:
            ensure_env_placeholder(env_file, base["model"]["api_key_env"])
            print(f"本地 key 文件已准备: {env_file}。运行前请填写 {base['model']['api_key_env']}。")
    print(f"配置已写入: {output}")
    return 0


def ensure_simulated_dataset(config: Dict[str, Any], scenario_name: str = "default", prompt_chars: Optional[int] = None) -> Path:
    dataset = config["dataset"]
    path = project_path(dataset.get("path") or "outputs/simulated_openqa.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(dataset.get("simulated_count") or 12)
    prompt_len = int(prompt_chars or dataset.get("simulated_prompt_chars") or 240)
    topic = scenario_name.replace("_", " ")
    rows = []
    for index in range(count):
        repeated = (
            f"这是用于模型性能压测的模拟问题 {index + 1}，场景是 {topic}。"
            "请用清晰、简短、结构化的中文回答，并保持内容稳定。"
        )
        while len(repeated) < prompt_len:
            repeated += " 补充上下文用于控制输入长度。"
        rows.append({"question": repeated[:prompt_len]})
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def convert_to_openqa(input_path: Path, output_path: Path, input_format: str = "auto") -> Tuple[int, int, int]:
    total = success = failed = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                if input_path.suffix.lower() == ".txt":
                    data = {"question": line}
                else:
                    data = json.loads(line)
                    if input_format == "auto":
                        if "messages" in data:
                            fmt = "messages"
                        elif "question" in data:
                            fmt = "openqa"
                        elif "text" in data or "prompt" in data:
                            fmt = "text"
                        else:
                            fmt = "unknown"
                    else:
                        fmt = input_format
                    if fmt == "messages":
                        parts = []
                        for message in data.get("messages", []):
                            role = str(message.get("role", "user")).capitalize()
                            parts.append(f"[{role}]: {message.get('content', '')}")
                        data = {"question": "\n".join(parts)}
                    elif fmt == "openqa":
                        data = {"question": data["question"]}
                    elif fmt == "text":
                        data = {"question": data.get("text") or data.get("prompt") or ""}
                    else:
                        raise ValueError("unknown input format")
                if not data.get("question"):
                    raise ValueError("empty question")
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                success += 1
            except Exception as exc:  # noqa: BLE001 - converter reports per-line failures
                print(f"转换失败: {exc}: {line[:120]}", file=sys.stderr)
                failed += 1
    return total, success, failed


def scenario_enabled(config: Dict[str, Any], name: str) -> bool:
    return bool(config.get("scenarios", {}).get(name, {}).get("enabled"))


def get_api_key(config: Dict[str, Any]) -> str:
    env_name = config["model"].get("api_key_env")
    return get_secret_from_env_or_file(config, env_name)


def scenario_output_dir(config: Dict[str, Any], scenario_name: str, run_id: str) -> Path:
    base = project_path(config["run"].get("outputs_dir") or "outputs/model_benchmark")
    return base / run_id / scenario_name


def numeric_list(values: Any) -> List[int]:
    if isinstance(values, list):
        return [int(value) for value in values]
    if isinstance(values, str):
        return [int(item.strip()) for item in values.split(",") if item.strip()]
    return [int(values)]


def build_numbers(parallels: List[int], scenario_cfg: Dict[str, Any]) -> List[int]:
    if scenario_cfg.get("numbers"):
        numbers = numeric_list(scenario_cfg["numbers"])
        if len(numbers) != len(parallels):
            raise ConfigError("numbers and parallels must have the same length")
        return numbers
    multiplier = int(scenario_cfg.get("number_multiplier") or 10)
    min_number = int(scenario_cfg.get("min_number") or 1)
    return [max(min_number, parallel * multiplier) for parallel in parallels]


def needs_optional_usage_plugin(config: Dict[str, Any]) -> bool:
    token_cfg = config.get("token_accounting", {})
    return token_cfg.get("mode") in {"api_usage", "tokenizer", "auto"} or token_cfg.get("on_missing_usage") != "fail"


def build_evalscope_args(
    config: Dict[str, Any],
    scenario_name: str,
    scenario_cfg: Dict[str, Any],
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    overrides = overrides or {}
    warnings: List[str] = []
    dataset_cfg = copy.deepcopy(config["dataset"])
    dataset_cfg.update(overrides.get("dataset", {}))
    model_cfg = config["model"]
    run_cfg = config["run"]

    api_key_env = model_cfg["api_key_env"]
    if not get_api_key(config):
        raise ConfigError(f"Missing API key. Set environment variable {model_cfg['api_key_env']}.")

    dataset_type = dataset_cfg.get("type", "simulated")
    tokenizer_path = model_cfg.get("tokenizer_path")

    eval_args: Dict[str, Any] = {
        "model": model_cfg["name"],
        "url": model_cfg["api_url"],
        "api": "openai_optional_usage" if needs_optional_usage_plugin(config) else model_cfg.get("api", "openai"),
        "api_key_env": api_key_env,
        "env_file": str(configured_env_file(config)) if configured_env_file(config) else None,
        "outputs_dir": str(output_dir),
        "no_timestamp": True,
        "name": scenario_name,
        "connect_timeout": int(run_cfg.get("connect_timeout") or 600),
        "read_timeout": int(run_cfg.get("read_timeout") or 600),
        "total_timeout": int(run_cfg.get("total_timeout") or 21600),
        "stream": bool(run_cfg.get("stream", True)),
        "temperature": float(run_cfg.get("temperature") or 0.0),
        "log_every_n_query": int(run_cfg.get("log_every_n_query") or 50),
        "enable_progress_tracker": bool(run_cfg.get("enable_progress_tracker", True)),
    }
    if run_cfg.get("seed") is not None:
        eval_args["seed"] = int(run_cfg["seed"])

    if tokenizer_path:
        eval_args["tokenizer_path"] = tokenizer_path

    if dataset_type == "random":
        if not tokenizer_path:
            warnings.append("dataset.type=random requires tokenizer_path; falling back to simulated openqa dataset.")
            dataset_type = "simulated"
        else:
            prompt_tokens = int(overrides.get("input_tokens") or dataset_cfg.get("random_prompt_tokens") or 512)
            eval_args.update(
                {
                    "dataset": "random",
                    "min_prompt_length": prompt_tokens,
                    "max_prompt_length": prompt_tokens,
                    "prefix_length": int(dataset_cfg.get("prefix_length") or 0),
                }
            )

    if dataset_type == "simulated":
        prompt_chars = overrides.get("prompt_chars")
        dataset_path = ensure_simulated_dataset(config, scenario_name=scenario_name, prompt_chars=prompt_chars)
        eval_args.update({"dataset": "openqa", "dataset_path": str(dataset_path)})
    elif dataset_type == "openqa":
        eval_args["dataset"] = "openqa"
        if dataset_cfg.get("path"):
            eval_args["dataset_path"] = str(project_path(dataset_cfg["path"]))
    elif dataset_type == "line_by_line":
        if not dataset_cfg.get("path"):
            raise ConfigError("dataset.type=line_by_line requires dataset.path")
        eval_args.update({"dataset": "line_by_line", "dataset_path": str(project_path(dataset_cfg["path"]))})

    max_tokens = overrides.get("max_tokens", scenario_cfg.get("max_tokens", dataset_cfg.get("output_tokens", 128)))
    eval_args["max_tokens"] = int(max_tokens)
    if overrides.get("min_tokens") is not None:
        eval_args["min_tokens"] = int(overrides["min_tokens"])
    elif scenario_cfg.get("min_tokens") is not None:
        eval_args["min_tokens"] = int(scenario_cfg["min_tokens"])

    if scenario_name == "gradient":
        parallels = numeric_list(scenario_cfg.get("parallels", [1]))
        eval_args["parallel"] = parallels
        eval_args["number"] = build_numbers(parallels, scenario_cfg)
        eval_args["sleep_interval"] = int(scenario_cfg.get("sleep_interval") or 5)
    else:
        parallel = int(overrides.get("parallel", scenario_cfg.get("parallel", 1)))
        number = int(overrides.get("number", scenario_cfg.get("number", max(1, parallel * 10))))
        eval_args["parallel"] = parallel
        eval_args["number"] = number

    return eval_args, warnings


def write_run_payload(
    config: Dict[str, Any],
    eval_args: Dict[str, Any],
    scenario_name: str,
    output_dir: Path,
    warnings: Optional[List[str]] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": scenario_name,
        "evalscope_args": eval_args,
        "token_accounting": config.get("token_accounting", {}),
        "warnings": warnings or [],
    }
    payload_path = output_dir / f"{scenario_name}_run_payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = output_dir / f"{scenario_name}_command_summary.json"
    safe_summary = sanitize_obj(payload)
    summary_path.write_text(json.dumps(safe_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload_path


def self_runner_command(config: Dict[str, Any], payload_path: Path) -> List[str]:
    return [active_python(config), str(Path(__file__).resolve()), "_evalscope-run", "--payload", str(payload_path)]


def execute_evalscope_payload(config: Dict[str, Any], payload_path: Path, dry_run: bool = False) -> int:
    env = os.environ.copy()
    env.setdefault("PIP_INDEX_URL", config["environment"].get("pip_index_url", ""))
    if config["environment"].get("modelscope_cache"):
        env.setdefault("MODELSCOPE_CACHE", str(project_path(config["environment"]["modelscope_cache"])))
    if config["environment"].get("hf_endpoint"):
        env.setdefault("HF_ENDPOINT", config["environment"]["hf_endpoint"])
    cmd = self_runner_command(config, payload_path)
    return run_subprocess(cmd, dry_run=dry_run, env=env)


def run_evalscope_scenario(
    config: Dict[str, Any],
    scenario_name: str,
    scenario_cfg: Dict[str, Any],
    run_id: str,
    dry_run: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
) -> int:
    output_dir = scenario_output_dir(config, scenario_name, run_id)
    eval_args, warnings = build_evalscope_args(config, scenario_name, scenario_cfg, output_dir, overrides)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    payload_path = write_run_payload(config, eval_args, scenario_name, output_dir, warnings)
    return execute_evalscope_payload(config, payload_path, dry_run=dry_run)


def build_sla_params(config: Dict[str, Any]) -> List[Dict[str, str]]:
    targets = config.get("targets", {})
    metric_map = {
        "avg_latency_ms": "avg_latency",
        "p99_latency_ms": "p99_latency",
        "avg_ttft_ms": "avg_ttft",
        "p99_ttft_ms": "p99_ttft",
        "avg_tpot_ms": "avg_tpot",
        "p99_tpot_ms": "p99_tpot",
        "qps": "rps",
        "output_tps": "tps",
    }
    params: Dict[str, str] = {}
    for target_key, sla_key in metric_map.items():
        value = targets.get(target_key)
        if value is None:
            continue
        if target_key.endswith("_ms"):
            params[sla_key] = f"<={float(value) / 1000.0}"
        elif target_key in {"qps", "output_tps"}:
            params[sla_key] = f">={float(value)}"
    return [params] if params else []


def run_sla(config: Dict[str, Any], run_id: str, dry_run: bool = False) -> int:
    scenario_cfg = copy.deepcopy(config["scenarios"].get("sla", {}))
    targets = config.get("targets", {})
    success_target = float(targets.get("success_rate_pct") or 100)
    sla_params = build_sla_params(config)
    if success_target == 100 and sla_params:
        output_dir = scenario_output_dir(config, "sla", run_id)
        eval_args, warnings = build_evalscope_args(config, "sla", scenario_cfg, output_dir)
        eval_args.update(
            {
                "sla_auto_tune": True,
                "sla_variable": scenario_cfg.get("variable", "parallel"),
                "sla_params": sla_params,
                "sla_lower_bound": int(scenario_cfg.get("lower_bound") or 1),
                "sla_upper_bound": int(scenario_cfg.get("upper_bound") or 100),
                "sla_num_runs": int(scenario_cfg.get("num_runs") or 3),
                "sla_number_multiplier": float(scenario_cfg.get("number_multiplier") or 5),
            }
        )
        payload_path = write_run_payload(config, eval_args, "sla", output_dir, warnings)
        return execute_evalscope_payload(config, payload_path, dry_run=dry_run)

    print("SLA target success_rate_pct is not 100 or no mappable SLA metric exists; running bounded custom search.")
    lower = int(scenario_cfg.get("lower_bound") or 1)
    upper = int(scenario_cfg.get("upper_bound") or 100)
    best = None
    records: List[Dict[str, Any]] = []
    while lower <= upper:
        mid = (lower + upper) // 2
        overrides = {"parallel": mid, "number": max(1, int(mid * float(scenario_cfg.get("number_multiplier") or 5)))}
        name = f"sla_parallel_{mid}"
        rc = run_evalscope_scenario(config, name, scenario_cfg, run_id, dry_run=dry_run, overrides=overrides)
        records.append({"parallel": mid, "return_code": rc})
        if dry_run:
            lower = mid + 1
            best = mid
            continue
        result_dir = scenario_output_dir(config, name, run_id)
        runs = collect_runs(result_dir)
        passed = bool(runs) and all(run.get("success_rate_pct", 0) >= success_target for run in runs)
        if passed:
            best = mid
            lower = mid + 1
        else:
            upper = mid - 1
    out = scenario_output_dir(config, "sla", run_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "custom_sla_summary.json").write_text(
        json.dumps({"best_parallel": best, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0 if best is not None else 1


def run_stability(config: Dict[str, Any], run_id: str, dry_run: bool = False) -> int:
    scenario_cfg = config["scenarios"].get("stability", {})
    duration = float(scenario_cfg.get("duration_minutes") or 30)
    window = float(scenario_cfg.get("window_minutes") or 5)
    num_windows = max(1, int(math.ceil(duration / window)))
    parallel = int(scenario_cfg.get("parallel") or 1)
    rc = 0
    for index in range(1, num_windows + 1):
        number = max(1, int(parallel * 60 * window))
        name = f"stability_window_{index}"
        overrides = {"parallel": parallel, "number": number, "max_tokens": scenario_cfg.get("max_tokens")}
        rc = run_evalscope_scenario(config, name, scenario_cfg, run_id, dry_run=dry_run, overrides=overrides)
        if rc != 0:
            break
    return rc


def run_length_matrix(config: Dict[str, Any], run_id: str, dry_run: bool = False) -> int:
    scenario_cfg = config["scenarios"].get("length_matrix", {})
    rc = 0
    for input_tokens in numeric_list(scenario_cfg.get("input_tokens", [500])):
        for output_tokens in numeric_list(scenario_cfg.get("output_tokens", [128])):
            name = f"length_in{input_tokens}_out{output_tokens}"
            overrides = {
                "input_tokens": input_tokens,
                "max_tokens": output_tokens,
                "min_tokens": output_tokens,
                "parallel": int(scenario_cfg.get("parallel") or 1),
                "number": int(scenario_cfg.get("number") or 100),
                "dataset": {"type": "random"},
                "prompt_chars": max(80, input_tokens * 2),
            }
            rc = run_evalscope_scenario(config, name, scenario_cfg, run_id, dry_run=dry_run, overrides=overrides)
            if rc != 0:
                return rc
    return rc


def run_benchmark(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    scenarios = [args.scenario] if args.scenario != "all" else ["smoke", "gradient", "sla", "stability", "length_matrix"]
    rc = 0
    for scenario in scenarios:
        if scenario in {"smoke", "gradient"}:
            if args.scenario == "all" and not scenario_enabled(config, scenario):
                continue
            rc = run_evalscope_scenario(config, scenario, config["scenarios"][scenario], run_id, dry_run=args.dry_run)
        elif scenario == "sla":
            if args.scenario == "all" and not scenario_enabled(config, "sla"):
                continue
            rc = run_sla(config, run_id, dry_run=args.dry_run)
        elif scenario == "stability":
            if args.scenario == "all" and not scenario_enabled(config, "stability"):
                continue
            rc = run_stability(config, run_id, dry_run=args.dry_run)
        elif scenario == "length_matrix":
            if args.scenario == "all" and not scenario_enabled(config, "length_matrix"):
                continue
            rc = run_length_matrix(config, run_id, dry_run=args.dry_run)
        else:
            raise ConfigError(f"Unknown scenario: {scenario}")
        if rc != 0:
            return rc

    if not args.skip_report and not args.dry_run:
        base = project_path(config["run"].get("outputs_dir") or "outputs/model_benchmark") / run_id
        report_path = base / "model_benchmark_report.md"
        generate_report(config, base, report_path)
        print(f"报告已生成: {report_path}")
    return rc


PERCENTILE_KEYS = ["10%", "25%", "50%", "66%", "75%", "80%", "90%", "95%", "98%", "99%"]


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_percentiles(data: Any) -> Dict[str, Dict[str, Optional[float]]]:
    result: Dict[str, Dict[str, Optional[float]]] = {}
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            percentile = str(row.get("Percentiles") or row.get("percentile") or "")
            if not percentile:
                continue
            for metric, value in row.items():
                if metric in {"Percentiles", "percentile"}:
                    continue
                result.setdefault(metric, {})[percentile] = safe_float(value)
    elif isinstance(data, dict):
        labels = data.get("Percentiles") or data.get("percentiles") or PERCENTILE_KEYS
        for metric, values in data.items():
            if metric in {"Percentiles", "percentiles"} or not isinstance(values, list):
                continue
            for label, value in zip(labels, values):
                result.setdefault(metric, {})[str(label)] = safe_float(value)
    return result


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def fmt(value: Any, decimals: int = 2, unavailable: bool = False) -> str:
    if unavailable:
        return "-"
    number = safe_float(value)
    if number is None:
        return "-"
    if decimals == 0:
        return str(int(round(number)))
    return f"{number:.{decimals}f}"


def collect_runs(results_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for summary_path in sorted(results_dir.rglob("benchmark_summary.json")):
        summary = load_json(summary_path)
        if not isinstance(summary, dict):
            continue
        percentile = parse_percentiles(load_json(summary_path.parent / "benchmark_percentile.json"))
        args_data = load_json(summary_path.parent / "benchmark_args.json") or {}
        total = safe_float(summary.get("Total requests")) or 0
        succeed = safe_float(summary.get("Succeed requests")) or 0
        success_rate = (succeed / total * 100.0) if total else None
        path_parts = summary_path.parent.relative_to(results_dir).parts if summary_path.parent != results_dir else ()
        scenario = path_parts[0] if path_parts else summary_path.parent.name
        run = {
            "path": summary_path.parent,
            "scenario": scenario,
            "summary": summary,
            "percentile": percentile,
            "args": args_data,
            "parallel": safe_float(summary.get("Number of concurrency")) or extract_parallel(summary_path.parent.name),
            "total_requests": total,
            "succeed_requests": succeed,
            "failed_requests": safe_float(summary.get("Failed requests")) or 0,
            "success_rate_pct": success_rate,
            "qps": safe_float(summary.get("Request throughput (req/s)")),
            "output_tps": safe_float(summary.get("Output token throughput (tok/s)")),
            "total_tps": safe_float(summary.get("Total token throughput (tok/s)")),
            "avg_ttft_ms": seconds_to_ms(summary.get("Average time to first token (s)")),
            "avg_tpot_ms": seconds_to_ms(summary.get("Average time per output token (s)")),
            "avg_latency_ms": seconds_to_ms(summary.get("Average latency (s)")),
            "avg_itl_ms": seconds_to_ms(summary.get("Average inter-token latency (s)")),
            "avg_input_tokens": safe_float(summary.get("Average input tokens per request")),
            "avg_output_tokens": safe_float(summary.get("Average output tokens per request")),
        }
        for label in ["50%", "90%", "95%", "99%"]:
            suffix = label.replace("%", "")
            run[f"p{suffix}_ttft_ms"] = seconds_to_ms(percentile.get("TTFT (s)", {}).get(label))
            run[f"p{suffix}_tpot_ms"] = seconds_to_ms(percentile.get("TPOT (s)", {}).get(label))
            run[f"p{suffix}_latency_ms"] = seconds_to_ms(percentile.get("Latency (s)", {}).get(label))
            run[f"p{suffix}_itl_ms"] = seconds_to_ms(percentile.get("ITL (s)", {}).get(label))
        runs.append(run)
    runs.sort(key=lambda item: (str(item["scenario"]), item.get("parallel") or 0, str(item["path"])))
    return runs


def extract_parallel(name: str) -> Optional[float]:
    for part in name.split("_"):
        if part.isdigit():
            return float(part)
    return None


def seconds_to_ms(value: Any) -> Optional[float]:
    number = safe_float(value)
    return number * 1000.0 if number is not None else None


TARGET_SPECS = {
    "success_rate_pct": ("成功率(%)", "higher"),
    "qps": ("QPS(req/s)", "higher"),
    "output_tps": ("输出吞吐(tok/s)", "higher"),
    "avg_ttft_ms": ("平均TTFT(ms)", "lower"),
    "p95_ttft_ms": ("P95 TTFT(ms)", "lower"),
    "p99_ttft_ms": ("P99 TTFT(ms)", "lower"),
    "avg_tpot_ms": ("平均TPOT(ms)", "lower"),
    "avg_latency_ms": ("平均E2E(ms)", "lower"),
    "p95_latency_ms": ("P95 E2E(ms)", "lower"),
    "p99_latency_ms": ("P99 E2E(ms)", "lower"),
}


def evaluate_targets(config: Dict[str, Any], runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    targets = config.get("targets", {})
    for key, (label, direction) in TARGET_SPECS.items():
        target = targets.get(key)
        if target is None:
            continue
        candidates = [(run.get(key), run) for run in runs if run.get(key) is not None]
        if not candidates:
            rows.append({"metric": label, "target": target, "actual": None, "gap": None, "passed": False, "run": "-"})
            continue
        best_value, best_run = max(candidates, key=lambda item: item[0]) if direction == "higher" else min(
            candidates, key=lambda item: item[0]
        )
        if direction == "higher":
            passed = best_value >= float(target)
            gap = best_value - float(target)
        else:
            passed = best_value <= float(target)
            gap = float(target) - best_value
        rows.append(
            {
                "metric": label,
                "target": target,
                "actual": best_value,
                "gap": gap,
                "passed": passed,
                "run": f"{best_run.get('scenario')}@P{fmt(best_run.get('parallel'), 0)}",
            }
        )
    return rows


def token_metrics_unavailable(config: Dict[str, Any], runs: List[Dict[str, Any]]) -> bool:
    token_cfg = config.get("token_accounting", {})
    if token_cfg.get("on_missing_usage") == "skip_token_metrics":
        return True
    if not runs:
        return False
    return all((run.get("avg_input_tokens") in {None, 0} and run.get("avg_output_tokens") in {None, 0}) for run in runs)


def best_run_by(runs: List[Dict[str, Any]], key: str, direction: str) -> Optional[Dict[str, Any]]:
    candidates = [run for run in runs if run.get(key) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[key]) if direction == "higher" else min(candidates, key=lambda item: item[key])


def generate_report(config: Dict[str, Any], results_dir: Path, output_file: Path) -> str:
    runs = collect_runs(results_dir)
    unavailable_tokens = token_metrics_unavailable(config, runs)
    target_rows = evaluate_targets(config, runs)
    best_qps = best_run_by(runs, "qps", "higher")
    success_target = float(config.get("targets", {}).get("success_rate_pct") or 99)
    safe_runs = [run for run in runs if (run.get("success_rate_pct") or 0) >= success_target]
    safe_concurrency = max((run.get("parallel") or 0 for run in safe_runs), default=None)

    lines: List[str] = []
    lines.append("# Model Benchmark 压测报告")
    lines.append("")
    lines.append("## 1. 测试信息")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| 生成时间 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |")
    lines.append(f"| 模型 | {config['model'].get('name')} |")
    lines.append(f"| API | {config['model'].get('api')} |")
    lines.append(f"| API URL | {config['model'].get('api_url')} |")
    lines.append(f"| 数据集类型 | {config['dataset'].get('type')} |")
    lines.append(f"| Token计量 | {config['token_accounting'].get('mode')} / {config['token_accounting'].get('on_missing_usage')} |")
    lines.append(f"| 结果目录 | {results_dir} |")
    lines.append("")

    if unavailable_tokens:
        lines.append("> Token 指标不可计或仅为 0：报告中的 token 吞吐、TPOT、平均输入/输出 token 不作为结论依据。")
        lines.append("")

    lines.append("## 2. 目标达标总览")
    lines.append("")
    if target_rows:
        lines.append("| 指标 | 目标 | 最佳实测 | 差距 | 是否达标 | 来源 |")
        lines.append("|------|------|----------|------|----------|------|")
        for row in target_rows:
            status = "达标" if row["passed"] else "未达标"
            lines.append(
                f"| {row['metric']} | {fmt(row['target'])} | {fmt(row['actual'])} | {fmt(row['gap'])} | {status} | {row['run']} |"
            )
    else:
        lines.append("未配置目标指标。")
    lines.append("")

    lines.append("## 3. 结论建议")
    lines.append("")
    if best_qps:
        lines.append(
            f"- QPS 峰值出现在 `{best_qps.get('scenario')}` 并发 {fmt(best_qps.get('parallel'), 0)}：{fmt(best_qps.get('qps'))} req/s。"
        )
    else:
        lines.append("- 未找到可用 QPS 结果。")
    if safe_concurrency is not None:
        lines.append(f"- 按成功率目标 {success_target:.2f}% 计算，最高安全并发为 {fmt(safe_concurrency, 0)}。")
    else:
        lines.append(f"- 没有并发级别达到成功率目标 {success_target:.2f}%。")
    lines.append("")

    lines.append("## 4. 基本性能表")
    lines.append("")
    lines.append("| 场景 | 并发 | 总请求 | 成功率(%) | QPS | 输出吞吐(tok/s) | 总吞吐(tok/s) |")
    lines.append("|------|------|--------|-----------|-----|-----------------|---------------|")
    for run in runs:
        lines.append(
            f"| {run['scenario']} | {fmt(run.get('parallel'), 0)} | {fmt(run.get('total_requests'), 0)} | "
            f"{fmt(run.get('success_rate_pct'))} | {fmt(run.get('qps'))} | "
            f"{fmt(run.get('output_tps'), unavailable=unavailable_tokens)} | {fmt(run.get('total_tps'), unavailable=unavailable_tokens)} |"
        )
    lines.append("")

    for title, prefix in [
        ("TTFT - 首 Token 延迟 (ms)", "ttft"),
        ("TPOT - 每输出 Token 延迟 (ms)", "tpot"),
        ("E2E - 端到端延迟 (ms)", "latency"),
        ("ITL - Token 间隔延迟 (ms)", "itl"),
    ]:
        metric_unavailable = unavailable_tokens and prefix == "tpot"
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| 场景 | 并发 | 平均 | P50 | P90 | P95 | P99 |")
        lines.append("|------|------|------|-----|-----|-----|-----|")
        for run in runs:
            lines.append(
                f"| {run['scenario']} | {fmt(run.get('parallel'), 0)} | {fmt(run.get(f'avg_{prefix}_ms'), unavailable=metric_unavailable)} | "
                f"{fmt(run.get(f'p50_{prefix}_ms'), unavailable=metric_unavailable)} | "
                f"{fmt(run.get(f'p90_{prefix}_ms'), unavailable=metric_unavailable)} | "
                f"{fmt(run.get(f'p95_{prefix}_ms'), unavailable=metric_unavailable)} | "
                f"{fmt(run.get(f'p99_{prefix}_ms'), unavailable=metric_unavailable)} |"
            )
        lines.append("")

    lines.append("## 9. Token 统计")
    lines.append("")
    lines.append("| 场景 | 并发 | 平均输入Tokens | 平均输出Tokens |")
    lines.append("|------|------|----------------|----------------|")
    for run in runs:
        lines.append(
            f"| {run['scenario']} | {fmt(run.get('parallel'), 0)} | "
            f"{fmt(run.get('avg_input_tokens'), unavailable=unavailable_tokens)} | "
            f"{fmt(run.get('avg_output_tokens'), unavailable=unavailable_tokens)} |"
        )
    lines.append("")

    lines.append("## 10. 错误摘要")
    lines.append("")
    failed_runs = [run for run in runs if (run.get("failed_requests") or 0) > 0]
    if failed_runs:
        lines.append("| 场景 | 并发 | 失败请求 | 原始路径 |")
        lines.append("|------|------|----------|----------|")
        for run in failed_runs:
            lines.append(
                f"| {run['scenario']} | {fmt(run.get('parallel'), 0)} | {fmt(run.get('failed_requests'), 0)} | {run['path']} |"
            )
    else:
        lines.append("未发现失败请求。")
    lines.append("")

    lines.append("## 11. 原始结果路径")
    lines.append("")
    for run in runs:
        lines.append(f"- `{run['scenario']}` 并发 {fmt(run.get('parallel'), 0)}: `{run['path']}`")
    if not runs:
        lines.append("- 未找到 `benchmark_summary.json`。")
    lines.append("")

    report = "\n".join(lines)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report, encoding="utf-8")
    return report


def run_report(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else copy.deepcopy(DEFAULT_CONFIG)
    results_dir = Path(args.results_dir)
    output = Path(args.output) if args.output else results_dir / "model_benchmark_report.md"
    generate_report(config, results_dir, output)
    print(f"报告已生成: {output}")
    return 0


def extract_usage_tokens(responses: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    for response in reversed(responses):
        if not isinstance(response, dict):
            continue
        usage = response.get("usage")
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            if prompt is not None and completion is not None:
                return int(prompt), int(completion)
    return None


def attach_safe_to_dict(args_obj: Any) -> None:
    """Mask secrets in EvalScope's config logging and benchmark_args.json."""
    original_to_dict = args_obj.to_dict

    def safe_to_dict() -> Dict[str, Any]:
        data = copy.deepcopy(original_to_dict())
        data = {key: value for key, value in data.items() if not callable(value)}
        if data.get("api_key"):
            data["api_key"] = mask_secret(str(data["api_key"]))
        headers = data.get("headers")
        if isinstance(headers, dict):
            for key, value in list(headers.items()):
                if key.lower() == "authorization":
                    text = str(value)
                    if text.lower().startswith("bearer "):
                        headers[key] = "Bearer " + mask_secret(text[7:])
                    else:
                        headers[key] = mask_secret(text)
        return data

    args_obj.to_dict = safe_to_dict


def _evalscope_run(args: argparse.Namespace) -> int:
    payload = load_json(Path(args.payload))
    if not isinstance(payload, dict):
        raise ConfigError(f"Invalid payload: {args.payload}")
    eval_args = payload["evalscope_args"]
    api_key_env = eval_args.pop("api_key_env", None)
    env_file = eval_args.pop("env_file", None)
    api_key = os.environ.get(api_key_env or "")
    if not api_key and env_file:
        api_key = load_env_file(Path(env_file)).get(api_key_env or "", "")
    if not api_key:
        raise ConfigError(f"Missing API key. Set {api_key_env} or provide environment.env_file.")
    eval_args["api_key"] = api_key
    token_cfg = payload.get("token_accounting", {})
    mode = token_cfg.get("mode", "auto")
    on_missing = token_cfg.get("on_missing_usage", "fallback_tokenizer")

    from collections import defaultdict

    local_evalscope = PROJECT_ROOT / "evalscope"
    if local_evalscope.exists() and str(local_evalscope) not in sys.path:
        sys.path.insert(0, str(local_evalscope))

    from evalscope.perf.arguments import Arguments
    from evalscope.perf.main import run_perf_benchmark
    from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
    from evalscope.perf.plugin.registry import register_api

    @register_api("openai_optional_usage")
    class OpenAIOptionalUsagePlugin(OpenaiPlugin):  # type: ignore
        def parse_responses(self, responses, request=None, **kwargs):  # noqa: ANN001
            usage = extract_usage_tokens(responses)
            if mode == "api_usage" and usage is not None:
                return usage
            if mode == "auto" and usage is not None:
                return usage
            if mode == "api_usage" and on_missing == "fail":
                raise ValueError("API response does not contain usage and on_missing_usage=fail.")
            if mode in {"tokenizer", "auto"} or on_missing == "fallback_tokenizer":
                if self.tokenizer is not None:
                    delta_contents = defaultdict(list)
                    for response in responses:
                        if not isinstance(response, dict):
                            continue
                        for choice in response.get("choices", []) or []:
                            idx = choice.get("index", 0)
                            if response.get("object") == "chat.completion.chunk":
                                delta = choice.get("delta", {}) or {}
                                delta_contents[idx].append((delta.get("content") or "") + (delta.get("reasoning_content") or ""))
                            elif response.get("object") == "chat.completion":
                                message = choice.get("message", {}) or {}
                                delta_contents[idx].append((message.get("content") or "") + (message.get("reasoning_content") or ""))
                            elif response.get("object") == "text_completion":
                                delta_contents[idx].append(choice.get("text") or "")
                    input_tokens = self._count_input_tokens(request)
                    output_tokens = 0
                    for pieces in delta_contents.values():
                        output_tokens += self._count_output_tokens("".join(pieces))
                    return input_tokens, output_tokens
                if on_missing == "fail":
                    raise ValueError("No usage and no tokenizer_path available.")
            if on_missing == "skip_token_metrics":
                return 0, 0
            raise ValueError("Unable to determine token counts.")

        async def process_request(self, client_session, url, headers, body):  # noqa: ANN001
            output = await super().process_request(client_session, url, headers, body)
            if output.success and not (output.generated_text or "").strip():
                output.success = False
                output.error = "Empty response body or empty generated_text."
            return output

    task_cfg = Arguments(**eval_args)
    attach_safe_to_dict(task_cfg)
    run_perf_benchmark(task_cfg)
    return 0


def legacy_config_from_args(args: argparse.Namespace, scenario: str) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["model"]["name"] = args.model
    config["model"]["api_url"] = args.api_url
    config["model"]["api_key_env"] = args.api_key_env or "MODEL_BENCHMARK_API_KEY"
    if args.api_key:
        os.environ[config["model"]["api_key_env"]] = args.api_key
    config["dataset"]["type"] = args.dataset
    if args.dataset_path:
        config["dataset"]["path"] = args.dataset_path
    if args.tokenizer_path:
        config["model"]["tokenizer_path"] = args.tokenizer_path
    config["run"]["outputs_dir"] = args.output_dir
    if getattr(args, "connect_timeout", None) is not None:
        config["run"]["connect_timeout"] = args.connect_timeout
    if getattr(args, "read_timeout", None) is not None:
        config["run"]["read_timeout"] = args.read_timeout
    config["dataset"]["output_tokens"] = args.max_tokens
    if getattr(args, "min_tokens", None) is not None:
        # Legacy scripts used --min-tokens to mean random input length. The
        # unified config uses random_prompt_tokens for that concept.
        config["dataset"]["random_prompt_tokens"] = args.min_tokens
    if scenario == "gradient":
        config["scenarios"]["gradient"]["parallels"] = numeric_list(args.parallels)
        config["scenarios"]["gradient"]["max_tokens"] = args.max_tokens
        config["scenarios"]["gradient"]["min_number"] = 200
    elif scenario == "stability":
        config["scenarios"]["stability"].update(
            {
                "parallel": args.parallel,
                "duration_minutes": args.duration,
                "window_minutes": args.window,
                "max_tokens": args.max_tokens,
            }
        )
    elif scenario == "sla":
        config["targets"]["success_rate_pct"] = args.target_success_rate
        if args.target_ttft is not None:
            config["targets"]["avg_ttft_ms"] = args.target_ttft
        if args.target_tpot is not None:
            config["targets"]["avg_tpot_ms"] = args.target_tpot
        config["scenarios"]["sla"].update(
            {
                "enabled": True,
                "lower_bound": args.min_parallel,
                "upper_bound": args.max_parallel,
                "number_multiplier": max(1, float(args.number_per_test) / max(1, float(args.min_parallel))),
            }
        )
    return config


def run_legacy_benchmark(args: argparse.Namespace) -> int:
    config = legacy_config_from_args(args, "gradient")
    run_id = datetime.now().strftime("benchmark_%Y%m%d_%H%M%S")
    rc = run_evalscope_scenario(config, "gradient", config["scenarios"]["gradient"], run_id, dry_run=args.dry_run)
    if rc == 0 and not args.skip_report and not args.dry_run:
        base = project_path(config["run"]["outputs_dir"]) / run_id
        generate_report(config, base, base / "benchmark_report.md")
    return rc


def run_legacy_stability(args: argparse.Namespace) -> int:
    config = legacy_config_from_args(args, "stability")
    run_id = datetime.now().strftime("stability_%Y%m%d_%H%M%S")
    rc = run_stability(config, run_id, dry_run=args.dry_run)
    if rc == 0 and not args.dry_run:
        base = project_path(config["run"]["outputs_dir"]) / run_id
        generate_report(config, base, base / "stability_report.md")
    return rc


def run_legacy_sla(args: argparse.Namespace) -> int:
    config = legacy_config_from_args(args, "sla")
    run_id = datetime.now().strftime("sla_%Y%m%d_%H%M%S")
    rc = run_sla(config, run_id, dry_run=args.dry_run)
    if rc == 0 and not args.dry_run:
        base = project_path(config["run"]["outputs_dir"]) / run_id
        generate_report(config, base, base / "sla_report.md")
    return rc


def add_common_legacy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="simulated", choices=["simulated", "random", "openqa", "line_by_line"])
    parser.add_argument("--dataset-path")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--min-tokens", type=int)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--connect-timeout", type=int)
    parser.add_argument("--read-timeout", type=int)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model Benchmark Skill unified runner")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check OS, Python, venv, EvalScope, API key, and optional endpoint reachability")
    doctor.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    doctor.add_argument("--check-endpoint", action="store_true")
    doctor.add_argument("--require-api-key", action="store_true")
    doctor.add_argument("--timeout", type=int, default=5)
    doctor.set_defaults(func=run_doctor)

    bootstrap = sub.add_parser("bootstrap", help="Create venv and install EvalScope with domestic mirrors")
    bootstrap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    bootstrap.add_argument("--dry-run", action="store_true")
    bootstrap.set_defaults(func=run_bootstrap)

    menu = sub.add_parser("menu", help="Interactive config wizard")
    menu.add_argument("--output", default=str(PROJECT_ROOT / "configs" / "model_benchmark.local.yaml"))
    menu.add_argument("--base-config", default=str(DEFAULT_CONFIG_PATH))
    menu.set_defaults(func=run_menu)

    run = sub.add_parser("run", help="Run smoke, gradient, sla, stability, length_matrix, or all enabled scenarios")
    run.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    run.add_argument("--scenario", default="all", choices=["all", "smoke", "gradient", "sla", "stability", "length_matrix"])
    run.add_argument("--run-id")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--skip-report", action="store_true")
    run.set_defaults(func=run_benchmark)

    report = sub.add_parser("report", help="Generate a detailed Markdown report from EvalScope outputs")
    report.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    report.add_argument("--results-dir", required=True)
    report.add_argument("--output")
    report.set_defaults(func=run_report)

    private = sub.add_parser("_evalscope-run", help=argparse.SUPPRESS)
    private.add_argument("--payload", required=True)
    private.set_defaults(func=_evalscope_run)

    legacy_bench = sub.add_parser("legacy-benchmark", help="Compatibility wrapper for the old benchmark.sh interface")
    add_common_legacy_args(legacy_bench)
    legacy_bench.add_argument("--parallels", default="1,2,5,8,10,15,20,25,30,35,40,50,60,80,100")
    legacy_bench.add_argument("--warmup-number", type=int)
    legacy_bench.add_argument("--skip-warmup", action="store_true")
    legacy_bench.add_argument("--skip-report", action="store_true")
    legacy_bench.set_defaults(func=run_legacy_benchmark)

    legacy_stability = sub.add_parser("legacy-stability", help="Compatibility wrapper for the old stability_test.sh interface")
    add_common_legacy_args(legacy_stability)
    legacy_stability.add_argument("--parallel", type=int, default=20)
    legacy_stability.add_argument("--duration", type=float, default=30)
    legacy_stability.add_argument("--window", type=float, default=5)
    legacy_stability.set_defaults(func=run_legacy_stability)

    legacy_sla = sub.add_parser("legacy-sla", help="Compatibility wrapper for the old sla_autotune.py interface")
    add_common_legacy_args(legacy_sla)
    legacy_sla.add_argument("--target-ttft", type=float)
    legacy_sla.add_argument("--target-tpot", type=float)
    legacy_sla.add_argument("--target-success-rate", type=float, default=99)
    legacy_sla.add_argument("--min-parallel", type=int, default=1)
    legacy_sla.add_argument("--max-parallel", type=int, default=100)
    legacy_sla.add_argument("--number-per-test", type=int, default=200)
    legacy_sla.set_defaults(func=run_legacy_sla)

    convert = sub.add_parser("convert-dataset", help="Convert JSONL/TXT data to EvalScope openqa JSONL")
    convert.add_argument("input")
    convert.add_argument("output")
    convert.add_argument("--format", default="auto", choices=["auto", "messages", "openqa", "text"])

    def _convert(ns: argparse.Namespace) -> int:
        total, success, failed = convert_to_openqa(Path(ns.input), Path(ns.output), ns.format)
        print_json({"total": total, "success": success, "failed": failed, "output": ns.output})
        return 0 if failed == 0 else 1

    convert.set_defaults(func=_convert)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 0:
        argv = ["menu", "--output", str(PROJECT_ROOT / "configs" / "model_benchmark.local.yaml")]
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
