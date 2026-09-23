"""Pure local throughput smoke; no task data or retained trained weights.

Fixed synthetic repeated input, three warm-up steps, five timed steps. This
measures feasibility only, never predictive accuracy or a research hypothesis.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .preflight import MODEL_ID, REVISION, PROJECT_ROOT, Tee, sha256
except ImportError:
    from preflight import MODEL_ID, REVISION, PROJECT_ROOT, Tee, sha256


def batch_from_ids(sequences, pad_id):
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long, device="cuda")
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        input_ids[row, -len(sequence):] = torch.tensor(sequence, dtype=torch.long, device="cuda")
        attention_mask[row, -len(sequence):] = 1
    position_ids = (attention_mask.cumsum(dim=-1) - 1).masked_fill(attention_mask == 0, 0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/local_scheme_validation/20260923_preflight")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "benchmark_console.log").open("w", encoding="utf-8") as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            run(args.output_dir)


def run(output_dir):
    torch.manual_seed(9023)
    torch.cuda.manual_seed_all(9023)
    torch.set_num_threads(4)
    snapshot = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / REVISION
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], bias="none")).to("cuda")
    model.config.use_cache = False
    phrase = (
        "This is a controlled local throughput check. The lamp is on and the door is closed. "
        "Read the sentence and return the letter A. No research data is used in this check. "
    )
    ids = tokenizer.encode(phrase * 8, add_special_tokens=False)[:128]
    assert len(ids) == 128
    label_ids = tokenizer.encode("A", add_special_tokens=False)
    assert len(label_ids) == 1
    training_inputs = batch_from_ids([ids] * 8, tokenizer.pad_token_id)
    labels = torch.full((8,), label_ids[0], dtype=torch.long, device="cuda")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-4)
    config = {
        "batch_size": 8, "sequence_length": 128, "warmup_steps": 3, "timed_steps": 5,
        "seed": 9023, "base_dtype": "bfloat16", "target_modules": ["q_proj", "v_proj"],
        "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.0,
        "lr": 2e-4, "optimizer": "AdamW", "weight_decay": optimizer.param_groups[0]["weight_decay"],
        "padding_side": tokenizer.padding_side, "explicit_position_ids": True,
        "logits_to_keep": 1, "use_cache": False, "gradient_checkpointing": False,
        "loss": "full-vocabulary next-token cross entropy, always target A",
        "target_token_id": label_ids[0], "trainable_parameters": sum(p.numel() for p in parameters),
        "trainable_dtypes": sorted({str(p.dtype) for p in parameters}),
        "prompt_token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "all_8_inputs_identical": True,
    }
    print(json.dumps({"status": "throughput_only", "config": config}, indent=2), flush=True)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for step in range(8):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(**training_inputs, logits_to_keep=1, use_cache=False).logits[:, -1, :]
        loss = F.cross_entropy(logits.float(), labels)
        loss.backward()
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        finite_gradients = bool(torch.stack([gradient.isfinite().all() for gradient in gradients]).all().item())
        finite_loss = bool(loss.isfinite().item())
        if not (finite_gradients and finite_loss and len(gradients) == len(parameters)):
            raise RuntimeError("Missing/nonfinite trainable gradients or nonfinite loss.")
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        row = {"step": step + 1, "phase": "warmup" if step < 3 else "timed", "seconds": elapsed,
               "loss": float(loss.detach()), "finite_loss": finite_loss, "finite_gradients": finite_gradients,
               "gradient_tensor_count": len(gradients)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    training_peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    training_peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
    optimizer.zero_grad(set_to_none=True)
    del logits, loss, gradients, optimizer, training_inputs, labels
    model.eval()
    # This has real left padding: 64-token and 128-token synthetic inputs.
    controlled_inputs = [ids[:64], ids]
    with torch.inference_mode():
        batched_logits = model(**batch_from_ids(controlled_inputs, tokenizer.pad_token_id),
                               logits_to_keep=1, use_cache=False).logits[:, -1, :].float()
        single_logits = torch.cat([
            model(**batch_from_ids([sequence], tokenizer.pad_token_id),
                  logits_to_keep=1, use_cache=False).logits[:, -1, :].float()
            for sequence in controlled_inputs], dim=0)
    # BF16 step around |logit|=16 is 0.125; combine absolute and relative tolerances.
    atol, rtol = 0.125, 0.02
    differences = (batched_logits - single_logits).abs()
    consistency = {
        "tested_after_8_smoke_steps": True, "input_lengths": [64, 128],
        "atol": atol, "rtol": rtol, "max_absolute_logit_difference": float(differences.max()),
        "mean_absolute_logit_difference": float(differences.mean()),
        "allclose": bool(torch.allclose(batched_logits, single_logits, atol=atol, rtol=rtol)),
        "argmax_equal": bool(torch.equal(batched_logits.argmax(-1), single_logits.argmax(-1))),
        "all_logits_finite": bool(batched_logits.isfinite().all() and single_logits.isfinite().all()),
        "scope": "BF16 numerical smoke only; no exact invariance or semantic accuracy claim.",
    }
    times = [row["seconds"] for row in rows if row["phase"] == "timed"]
    report = {
        "status": "throughput_only", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID, "revision": REVISION, "script_sha256": sha256(Path(__file__)),
        "scheme_effectiveness_evaluated": False, "real_train_dev_test_data_accessed": False,
        "local_files_only": True, "trust_remote_code": False, "checkpoint_saved": False,
        "config": config, "steps": rows, "seconds_per_step_mean": sum(times) / len(times),
        "seconds_per_step_min": min(times), "seconds_per_step_max": max(times),
        "examples_per_second": 8 * len(times) / sum(times),
        "input_tokens_per_second": 8 * 128 * len(times) / sum(times),
        "training_peak_allocated_mib": training_peak_allocated,
        "training_peak_reserved_mib": training_peak_reserved,
        "batch_single_consistency": consistency,
        "passed": all(row["finite_gradients"] and row["finite_loss"] for row in rows)
                  and consistency["allclose"] and consistency["all_logits_finite"],
        "limitations": "Eight identical artificial inputs; task-specific lengths, losses and relational expansion may cost more. No trained weights retained.",
    }
    (output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"steps", "config"}}, indent=2), flush=True)
    del parameters, model, base, batched_logits, single_logits, differences
    torch.cuda.empty_cache()
    if not report["passed"]:
        raise RuntimeError("Throughput smoke or batch/single numerical check failed; inspect benchmark.json.")


if __name__ == "__main__":
    main()
