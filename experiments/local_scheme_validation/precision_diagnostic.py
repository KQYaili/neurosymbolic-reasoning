"""Isolate synthetic batch/padding numerics without training or real task data."""
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import batch_from_ids
from preflight import PROJECT_ROOT, REVISION


torch.set_num_threads(4)
snapshot = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / REVISION
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                           dtype=torch.bfloat16).to("cuda").eval()
phrase = ("This is a controlled local throughput check. The lamp is on and the door is closed. "
          "Read the sentence and return the letter A. No research data is used in this check. ")
ids = tokenizer.encode(phrase * 8, add_special_tokens=False)[:128]
short = ids[:64]


@torch.inference_mode()
def run(label):
    inputs = {
        "single64": batch_from_ids([short], tokenizer.pad_token_id),
        "single128": batch_from_ids([ids], tokenizer.pad_token_id),
        "batch_same64": batch_from_ids([short, short], tokenizer.pad_token_id),
        "batch_mixed128": batch_from_ids([short, ids], tokenizer.pad_token_id),
    }
    values = {key: model(**batch, logits_to_keep=1, use_cache=False).logits[:, -1].float()
              for key, batch in inputs.items()}
    comparisons = {}
    for name, single, batched in [("batch_dimension_short", values["single64"][0], values["batch_same64"][0]),
                                  ("padding_short", values["single64"][0], values["batch_mixed128"][0]),
                                  ("batch_dimension_long", values["single128"][0], values["batch_mixed128"][1])]:
        delta = (single - batched).abs()
        comparisons[name] = {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                             "abc_abs": delta[[32, 33, 34]].tolist(),
                             "allclose": bool(torch.allclose(single, batched, atol=0.125, rtol=0.02))}
    print(label, json.dumps(comparisons), flush=True)
    return comparisons


results = {"bf16_default": run("bf16_default")}
with sdpa_kernel(SDPBackend.MATH):
    results["bf16_math_attention"] = run("bf16_math_attention")
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
results["bf16_full_reduction"] = run("bf16_full_reduction")
model.float()
torch.backends.cuda.matmul.allow_tf32 = False
results["fp32_no_tf32"] = run("fp32_no_tf32")
report = {"status": "synthetic_numerical_diagnostic_only", "training_steps": 0, "results": results}
path = PROJECT_ROOT / "runs/local_scheme_validation/20260923_preflight/precision_diagnostic.json"
path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
del model
torch.cuda.empty_cache()
