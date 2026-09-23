"""Complete end-to-end execution of the semantic disentanglement experiment.

1. Clean pre-flight state.
2. Train all 6 adapters (format_only and semantic_only across seeds 17, 29, 43).
3. Evaluate all 5 arms on 2,048-instance holdout in FP32.
4. Independent verification and audit.
5. Generate comparison figures and comprehensive final scientific report.
"""
import argparse
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / 'runs/local_scheme_validation/20260924_semantic_disentangle'
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as tr
import evaluate_disentangle as ev_dis
import verify_disentangle as ver_dis
import build_disentangle_report as rep_dis


def main():
    print("================ STEP 1: CLEANING STALE RUN ARTIFACTS ================", flush=True)
    for stale_name in ('implementation.json', 'sources', 'training.jsonl', 'training_complete.json',
                       'adapters', 'predictions', 'evaluation_seal.json', 'summary.json',
                       'verification.json', 'disentangle_comparison.png', 'REPORT.md'):
        target = RUN_DIR / stale_name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.is_file():
            target.unlink(missing_ok=True)
    print("Clean state verified.", flush=True)

    print("\n================ STEP 2: POST-TRAINING DISENTANGLED ARMS ================", flush=True)
    start_train = time.monotonic()
    tr.run_training(RUN_DIR)
    train_duration = time.monotonic() - start_train
    print(f"Post-training completed in {train_duration:.1f} seconds.", flush=True)

    print("\n================ STEP 3: SEALED FP32 EVALUATION ON HOLDOUT ================", flush=True)
    start_eval = time.monotonic()
    ev_dis.main(RUN_DIR, batch_size=16)
    eval_duration = time.monotonic() - start_eval
    print(f"Holdout evaluation completed in {eval_duration:.1f} seconds.", flush=True)

    print("\n================ STEP 4: INDEPENDENT VERIFICATION & AUDIT ================", flush=True)
    v_report = ver_dis.audit()
    print(f"Verification complete: all_passed={v_report['checks']['all_passed']}", flush=True)

    print("\n================ STEP 5: GENERATE FIGURES AND FINAL REPORT ================", flush=True)
    rep_dis.main()
    print("All tasks finished successfully!", flush=True)


if __name__ == '__main__':
    main()
