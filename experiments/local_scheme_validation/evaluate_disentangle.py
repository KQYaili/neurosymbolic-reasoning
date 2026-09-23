"""Sealed evaluation of semantic vs format disentangled models on the 20260924 Holdout split.

Evaluates 5 arms:
1. base (seed 0, no adapter)
2. consistency (reused control, seeds 17, 29, 43)
3. relational (reused control, seeds 17, 29, 43)
4. format_only (newly trained, seeds 17, 29, 43)
5. semantic_only (newly trained, seeds 17, 29, 43)

All inference runs in FP32 with TF32 disabled on 2,048 OOD instances per task.
Pre-registered stopping gates are audited against pre-declared thresholds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as tr
import evaluate as ev

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'runs/local_scheme_validation/20260924_semantic_disentangle'


def compare_holdout(run, task, a, b, seeds, repetitions, bootstrap_seed):
    differences = {k: [] for k in ('accuracy', 'base_accuracy', 'atc', 'rvr')}
    pred_dir = run / 'predictions'
    for seed in seeds:
        fa = pred_dir / f'{a}_seed{seed}_holdout_{task}.npz'
        fb = pred_dir / f'{b}_seed{seed}_holdout_{task}.npz'
        with np.load(fa) as x, np.load(fb) as y:
            assert np.array_equal(x['bundle_ids'], y['bundle_ids']), f'Bundle ID mismatch between {fa} and {fb}'
            assert np.array_equal(x['labels'], y['labels']), f'Label mismatch between {fa} and {fb}'
            xm = ev.metric_arrays(x['probabilities'], x['labels'], x.get('option_ids'))
            ym = ev.metric_arrays(y['probabilities'], y['labels'], y.get('option_ids'))
            for k in differences:
                differences[k].append(xm[k] - ym[k])
    report = {}
    for key, value in differences.items():
        delta = np.asarray(value)  # (seeds, bundles)
        per_bundle = delta.mean(0)
        rng = np.random.default_rng(bootstrap_seed)
        boot = np.array([per_bundle[rng.integers(0, len(per_bundle), len(per_bundle))].mean() for _ in range(repetitions)])
        report[key] = {
            'mean': float(delta.mean()),
            'by_seed': delta.mean(1).tolist(),
            'paired_bundle_95ci': np.quantile(boot, [0.025, 0.975]).tolist()
        }
    return report


def main(run_dir, batch_size):
    protocol = json.loads((run_dir / 'protocol.json').read_text(encoding='utf-8'))
    complete = json.loads((run_dir / 'training_complete.json').read_text(encoding='utf-8'))
    impl = json.loads((run_dir / 'implementation.json').read_text(encoding='utf-8'))
    holdout_manifest = json.loads((run_dir / 'holdout_manifest.json').read_text(encoding='utf-8'))

    # Verify contracts
    assert complete['contract_sha256'] == tr.sha256(run_dir / 'implementation.json')
    assert impl['protocol_sha256'] == tr.sha256(run_dir / 'protocol.json')
    assert tr.sha256(run_dir / 'data_holdout.json') == holdout_manifest['data_holdout_sha256']
    assert tr.sha256(run_dir / 'oracle_holdout.json') == holdout_manifest['oracle_holdout_sha256']

    for record in complete['records']:
        assert record['steps'] == protocol['steps']
        assert record['contract_sha256'] == complete['contract_sha256']
        assert tr.sha256(run_dir / record['adapter_path'] / 'adapter_model.safetensors') == record['adapter_sha256']

    # Load holdout data and oracle
    data_holdout = json.loads((run_dir / 'data_holdout.json').read_text(encoding='utf-8'))
    oracle_holdout = json.loads((run_dir / 'oracle_holdout.json').read_text(encoding='utf-8'))['sources']

    # Determine adapter directories for all arms to evaluate
    all_arms = ['base', 'consistency', 'relational', 'format_only', 'semantic_only']
    adapter_targets = []
    
    # 1. Base model
    adapter_targets.append({'arm': 'base', 'seed': 0, 'adapter_path': None})

    # 2. Reused controls
    for arm in ('consistency', 'relational'):
        for seed in protocol['seeds']:
            rel_path = protocol['reused_controls'][arm].format(seed=seed)
            full_path = ROOT / rel_path
            assert full_path.exists(), f'Missing reused checkpoint: {full_path}'
            adapter_targets.append({
                'arm': arm,
                'seed': seed,
                'adapter_path': full_path,
                'reused': True
            })

    # 3. Disentangle arms
    for record in complete['records']:
        adapter_targets.append({
            'arm': record['arm'],
            'seed': record['seed'],
            'adapter_path': run_dir / record['adapter_path'],
            'reused': False
        })

    # Write evaluation seal
    seal = {
        'protocol_sha256': tr.sha256(run_dir / 'protocol.json'),
        'contract_sha256': complete['contract_sha256'],
        'training_complete_sha256': tr.sha256(run_dir / 'training_complete.json'),
        'data_holdout_sha256': holdout_manifest['data_holdout_sha256'],
        'oracle_holdout_sha256': holdout_manifest['oracle_holdout_sha256'],
        'new_adapters': {r['adapter_path']: r['adapter_sha256'] for r in complete['records']},
        'reused_controls': protocol['reused_controls'],
        'evaluator_sha256': tr.sha256(__file__),
        'batch_size': batch_size,
        'dtype': 'float32_no_tf32'
    }
    tr.save_json(run_dir / 'evaluation_seal.json', seal, immutable=True)

    tr.setup(protocol['bootstrap_seed'])
    pred_dir = run_dir / 'predictions'
    pred_dir.mkdir(exist_ok=True)

    results = []
    token_cache = None

    for row in adapter_targets:
        arm = row['arm']
        seed = row['seed']
        done_file = pred_dir / f'{arm}_seed{seed}_metrics.json'
        if done_file.exists():
            cached = json.loads(done_file.read_text(encoding='utf-8'))
            results.append(cached)
            print(f'RESUME EVAL {arm} seed={seed}', flush=True)
            continue

        started = time.monotonic()
        model, tokenizer, answer_ids = tr.base_and_tokenizer(protocol, with_adapter=False)
        if arm != 'base':
            model = PeftModel.from_pretrained(model, row['adapter_path'], is_trainable=False)
        model.float().eval()

        if token_cache is None:
            token_cache = {
                v['id']: tr.encode_prompt(tokenizer, v['prompt'])
                for task in ('nli', 'mcqa')
                for b in data_holdout['splits']['ood'][task]
                for v in b['variants']
            }
            assert max(map(len, token_cache.values())) <= protocol['max_tokens']

        record = {
            'arm': arm,
            'seed': seed,
            'adapter_path': str(row['adapter_path']) if row['adapter_path'] else None,
            'ood': {},
            'prediction_sha256': {}
        }

        for task in ('nli', 'mcqa'):
            bundles = data_holdout['splits']['ood'][task]
            metrics, arrays = ev.evaluate_group(model, tokenizer, answer_ids, bundles, oracle_holdout, token_cache, batch_size)
            filename = f'{arm}_seed{seed}_holdout_{task}.npz'
            np.savez_compressed(pred_dir / filename, **arrays)
            record['prediction_sha256'][filename] = tr.sha256(pred_dir / filename)
            record['ood'][task] = metrics
            print(f'EVAL {arm} seed={seed} holdout/{task} accuracy={metrics["accuracy"]:.4f} ATC={metrics["atc"]:.4f} RVR={metrics["rvr"]:.4f}', flush=True)

        record['seconds'] = time.monotonic() - started
        tr.save_json(done_file, record, immutable=True)
        results.append(record)
        del model
        torch.cuda.empty_cache()

    # Aggregate metrics across seeds
    aggregate = {'ood': {}}
    for task in ('nli', 'mcqa'):
        aggregate['ood'][task] = {}
        for arm in all_arms:
            selected = [r['ood'][task] for r in results if r['arm'] == arm]
            aggregate['ood'][task][arm] = {
                key: {
                    'mean': float(np.mean([r[key] for r in selected])),
                    'std': float(np.std([r[key] for r in selected], ddof=1)) if len(selected) > 1 else None
                }
                for key in ('accuracy', 'base_accuracy', 'atc', 'rvr', 'macro_f1_all_views',
                            'raw_first_token_invalid_rate', 'raw_first_token_accuracy', 'mean_entropy',
                            'allowed_label_probability_mass', 'valid_label_probability_mass')
            }

    # Statistical comparisons
    comparisons = {'ood': {}}
    for task in ('nli', 'mcqa'):
        comparisons['ood'][task] = {}
        for a, b in protocol['comparisons']:
            comp_key = f'{a}_minus_{b}'
            comparisons['ood'][task][comp_key] = compare_holdout(
                run_dir, task, a, b, protocol['seeds'], protocol['bootstrap_repetitions'], protocol['bootstrap_seed']
            )

    # Stopping gates audit
    # Pre-registered gates: candidate (semantic_only) vs baseline (consistency) on holdout NLI
    sg_spec = protocol['stopping_gates']
    nli_comp = comparisons['ood']['nli'][f"{sg_spec['candidate']}_minus_{sg_spec['baseline']}"]
    mcqa_comp = comparisons['ood']['mcqa'][f"{sg_spec['candidate']}_minus_{sg_spec['baseline']}"]

    delta_atc_pp = nli_comp['atc']['mean'] * 100.0
    delta_atc_ci_low_pp = nli_comp['atc']['paired_bundle_95ci'][0] * 100.0
    delta_acc_ci_low_pp = nli_comp['accuracy']['paired_bundle_95ci'][0] * 100.0
    delta_rvr_ci_high_pp = nli_comp['rvr']['paired_bundle_95ci'][1] * 100.0
    mcqa_atc_ci_low_pp = mcqa_comp['atc']['paired_bundle_95ci'][0] * 100.0

    stopping_gates = {
        'candidate': sg_spec['candidate'],
        'baseline': sg_spec['baseline'],
        'target_task': 'nli',
        'atc_min_gain_met': delta_atc_pp >= sg_spec['atc_min_gain_pp'],
        'atc_ci_positive_met': delta_atc_ci_low_pp > sg_spec['atc_ci_lower_min_pp'],
        'accuracy_noninferior_met': delta_acc_ci_low_pp > sg_spec['accuracy_ci_lower_min_pp'],
        'rvr_reduced_met': delta_rvr_ci_high_pp < sg_spec['rvr_ci_upper_max_pp'],
        'mcqa_guard_met': mcqa_atc_ci_low_pp > -1.0,
        'values': {
            'delta_nli_atc_mean_pp': delta_atc_pp,
            'delta_nli_atc_ci_low_pp': delta_atc_ci_low_pp,
            'delta_nli_accuracy_ci_low_pp': delta_acc_ci_low_pp,
            'delta_nli_rvr_ci_high_pp': delta_rvr_ci_high_pp,
            'delta_mcqa_atc_ci_low_pp': mcqa_atc_ci_low_pp
        }
    }
    stopping_gates['all_passed'] = (
        stopping_gates['atc_min_gain_met'] and
        stopping_gates['atc_ci_positive_met'] and
        stopping_gates['accuracy_noninferior_met'] and
        stopping_gates['rvr_reduced_met'] and
        stopping_gates['mcqa_guard_met']
    )

    summary = {
        'results': results,
        'aggregate': aggregate,
        'comparisons': comparisons,
        'stopping_gates': stopping_gates,
        'stopping_decision': 'PROCEED' if stopping_gates['all_passed'] else 'DEFINITIVE_STOP',
        'evaluation_seal_sha256': tr.sha256(run_dir / 'evaluation_seal.json'),
        'holdout_manifest_sha256': tr.sha256(run_dir / 'holdout_manifest.json')
    }
    tr.save_json(run_dir / 'summary.json', summary)
    print("\n================== EVALUATION COMPLETE ==================", flush=True)
    print(f"Stopping decision: {summary['stopping_decision']}", flush=True)
    print(json.dumps(stopping_gates, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    main(args.run_dir.resolve(), args.batch_size)
