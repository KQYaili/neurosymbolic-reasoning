"""Independently recompute saved metrics and inspect final model invariants for V2."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.neurosymbolic_nli.data import load_data
from experiments.neurosymbolic_nli.model_v2 import NeurosymbolicNLIv2


def verify_results(run_dir):
    run_dir = Path(run_dir).resolve()
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    protocol = json.loads((run_dir / 'protocol.json').read_text(encoding='utf-8'))
    completion = json.loads((run_dir / 'training_complete.json').read_text(encoding='utf-8'))
    implementation = json.loads((run_dir / 'implementation.json').read_text(encoding='utf-8'))
    data = load_data(run_dir)
    expected_labels = data['splits']['test']['labels'].numpy()
    expected_ids = np.asarray(data['splits']['test']['row_ids'])

    checks = {}
    expected_runs = len(protocol['arms']) * len(protocol['seeds'])
    checks['fifteen_final_runs'] = len(summary['results']) == len(completion['records']) == expected_runs
    checks['frozen_protocol'] = hashlib.sha256((run_dir / 'protocol.json').read_bytes()).hexdigest() == summary['protocol_sha256']
    checks['frozen_sources'] = all(
        hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() == value
        for name, value in implementation['source_sha256'].items()
        if (Path(__file__).parent / name).exists()
    )

    paired_correct = {}
    for result in summary['results']:
        key = f"{result['arm']}_seed{result['seed']}"
        path = run_dir / f'{key}_test_predictions.npz'
        with np.load(path, allow_pickle=False) as z:
            y, pred, probs = z['labels'], z['predictions'], z['probabilities']
            assert np.array_equal(z['row_ids'], expected_ids), f"Row ID mismatch in {key}"
            assert np.array_equal(y, expected_labels), f"Label mismatch in {key}"
            assert probs.shape == (9824, 3) and np.isfinite(probs).all(), f"Invalid probs in {key}"
            assert np.allclose(probs.sum(axis=1), 1, atol=1e-5), f"Probs do not sum to 1 in {key}"
            assert np.array_equal(pred, probs.argmax(1)), f"Argmax mismatch in {key}"
            correct = (pred == y)
            assert abs(correct.mean() - result['test']['accuracy']) < 1e-7, f"Acc mismatch in {key}"
            paired_correct[(result['arm'], result['seed'])] = correct

    # Check that paired initial states match exactly across arms within same architecture family
    by_seed_enc = {}
    for rec in completion['records']:
        s, enc = rec['seed'], rec['encoder']
        by_seed_enc.setdefault((s, enc), []).append(rec['initial_state_sha256'])
    for (s, enc), hashes in by_seed_enc.items():
        assert len(set(hashes)) == 1, f"Seed {s} enc {enc} initial states do not match across arms!"
    checks['paired_initial_weights'] = True

    # Padding invariance on final checkpoints
    embeddings = data['embeddings']
    sample_tokens = data['splits']['test']['hypotheses'][:5]
    sample_lens = data['splits']['test']['hypothesis_lengths'][:5]
    pad_extra = torch.zeros(5, 10, dtype=sample_tokens.dtype)
    sample_padded = torch.cat([sample_tokens, pad_extra], dim=1)

    checks['padding_invariance_all_checkpoints'] = True
    for rec in completion['records']:
        ckpt = torch.load(rec['checkpoint'], map_location='cpu', weights_only=False)
        m = NeurosymbolicNLIv2(embeddings, **ckpt['model_config']).eval()
        m.load_state_dict(ckpt['model_state'])
        with torch.no_grad():
            r1, _ = m.encode(sample_tokens, sample_lens)
            r2, _ = m.encode(sample_padded, sample_lens)
            diff = (r1 - r2).abs().max().item()
            if diff > 1e-5:
                checks['padding_invariance_all_checkpoints'] = False
                break

    report = {
        'run_dir': str(run_dir),
        'checks': checks,
        'all_passed': all(checks.values()),
        'comparisons': summary.get('comparisons', {}),
        'aggregate': summary['aggregate']
    }
    (run_dir / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    res = verify_results(parser.parse_args().run_dir)
    print(json.dumps(res, indent=2))
