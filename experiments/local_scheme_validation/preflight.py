"""Read-only model/environment check; no training, downloads or environment changes.

Only the requested report directory is written. The model is loaded from a fixed
local Hugging Face snapshot with remote code disabled.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def inventory_cache(root: Path):
    inventory = []
    for model_dir in sorted(root.glob("models--*")):
        files = [p for p in model_dir.rglob("*") if p.is_file() and not p.is_symlink()]
        snapshots = []
        for snapshot in sorted((model_dir / "snapshots").glob("*")):
            if snapshot.is_dir():
                snapshots.append({
                    "revision": snapshot.name,
                    "broken_links": [p.name for p in snapshot.iterdir()
                                     if p.is_symlink() and not p.exists()],
                    "files": sorted(p.name for p in snapshot.iterdir()),
                })
        inventory.append({
            "model_id": model_dir.name.removeprefix("models--").replace("--", "/"),
            "bytes_excluding_snapshot_symlinks": sum(p.stat().st_size for p in files),
            "snapshots": snapshots,
            "forward_executed": model_dir.name == "models--Qwen--Qwen2.5-0.5B-Instruct",
        })
    return inventory


def run(output_dir: Path, cache_root: Path):
    import torch
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = cache_root / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / REVISION
    required = ["config.json", "generation_config.json", "model.safetensors",
                "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Required local snapshot files missing: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("This preflight requires the already configured local CUDA runtime.")

    print(f"Local model: {MODEL_ID}@{REVISION}", flush=True)
    print("Status: awaiting_scheme. This checks runtime feasibility only.", flush=True)
    model_files = {name: {"bytes": (snapshot / name).stat().st_size,
                          "sha256": sha256(snapshot / name)} for name in required}
    free_before, total = torch.cuda.mem_get_info()
    gpu_query = subprocess.run([
        "nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader"], capture_output=True, text=True, check=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16).to("cuda").eval()
    inputs = tokenizer("This is an offline model check.", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        logits = model(**inputs).logits
    torch.cuda.synchronize()
    forward = {
        "passed": bool(torch.isfinite(logits).all().item()),
        "finite": bool(torch.isfinite(logits).all().item()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input_shape": list(inputs["input_ids"].shape),
        "logits_shape": list(logits.shape),
        "dtype": str(next(model.parameters()).dtype),
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "load_and_forward_seconds": time.perf_counter() - started,
        "training_steps": 0,
        "semantic_accuracy_evaluated": False,
    }
    del logits, inputs, model, tokenizer
    torch.cuda.empty_cache()

    memory = {}
    memory_file = Path("/proc/meminfo")
    if memory_file.exists():
        for line in memory_file.read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemFree", "MemAvailable"}:
                memory[key + "_kib"] = int(value.strip().split()[0])
    report = {
        "status": "awaiting_scheme",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Local runtime and one offline forward only; no scheme-effectiveness claim.",
        "environment_changed": False, "downloads_performed": False,
        "remote_code_allowed": False, "local_files_only": True,
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["torch", "transformers", "peft", "accelerate", "safetensors"]},
        "peft_import_version": peft.__version__,
        "gpu": {"name": torch.cuda.get_device_name(0), "cuda_available": True,
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "total_bytes": total, "free_bytes_before_model": free_before,
                "nvidia_smi_before_model": gpu_query.stdout.strip(),
                "nvidia_smi_exit_code": gpu_query.returncode},
        "memory": memory, "model_id": MODEL_ID, "revision": REVISION,
        "snapshot": str(snapshot), "model_files": model_files,
        "script_sha256": sha256(Path(__file__)),
        "forward": forward, "cache_inventory": inventory_cache(cache_root),
        "next_step": "Read and agree on the actual scheme, then freeze a synthetic-data validation protocol.",
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "PREFLIGHT.md").write_text(
        "# 本机预检：等待方案正文\n\n"
        "状态：`awaiting_scheme`。本次仅验证本机环境与一个固定基模的离线前向计算。"
        "未训练、未下载模型、未修改环境；尚未验证研究方案是否有效或带来提升。\n\n"
        f"- 基模：`{MODEL_ID}`\n- Revision：`{REVISION}`\n"
        f"- GPU：{report['gpu']['name']}\n"
        f"- Python：`{sys.executable}`\n"
        f"- 参数量：{forward['parameters']:,}\n"
        f"- BF16 输出形状：`{forward['logits_shape']}`，全部有限：`{forward['finite']}`\n"
        f"- 实测显存峰值（allocated）：{forward['cuda_peak_allocated_mib']:.2f} MiB\n"
        f"- 加载与前向耗时：{forward['load_and_forward_seconds']:.3f} 秒\n\n"
        "此显存结果仅针对一条 7-token 输入的推理；不能作为训练或长上下文的显存估计。"
        "其他缓存仅列目录及文件完整性，未额外运行模型。\n\n"
        "所有模型文件的 SHA-256、当前资源和包版本见 `preflight.json`，"
        "实际标准输出与标准错误保存在 `console.log`。\n\n"
        "复现（WSL Ubuntu）：\n\n```bash\n"
        "cd /mnt/c/Users/lgd/PycharmProjects/JupyterProject1\n"
        "/home/lgd/anaconda3/bin/python experiments/local_scheme_validation/preflight.py\n"
        "```\n\n"
        "读取指定网页方案之后，才制定模型训练、模拟数据、对照组和独立留出集。\n",
        encoding="utf-8")
    print(json.dumps({"status": report["status"], "forward": forward,
                      "model_weights_sha256": model_files["model.safetensors"]["sha256"],
                      "report_directory": str(output_dir)}, ensure_ascii=False, indent=2), flush=True)
    if not forward["passed"]:
        raise RuntimeError("Nonfinite outputs from local model forward.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/local_scheme_validation/20260923_preflight")
    parser.add_argument("--cache-root", type=Path, default=Path.home() / ".cache/huggingface/hub")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "console.log").open("w", encoding="utf-8") as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            run(args.output_dir, args.cache_root)


if __name__ == "__main__":
    main()
