"""Independently recompute saved metrics and inspect final model invariants."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .data import load_data
    from .model import NeuralOrderNLI
else:
    from data import load_data
    from model import NeuralOrderNLI


def verify_results(run_dir):
    run_dir = Path(run_dir).resolve()
    summary = json.loads((run_dir / 'summary.json').read_text())
    protocol = json.loads((run_dir / 'protocol.json').read_text())
    completion = json.loads((run_dir / 'training_complete.json').read_text())
    implementation = json.loads((run_dir / 'implementation.json').read_text())
    data = load_data(run_dir)
    expected_labels = data['splits']['test']['labels'].numpy()
    expected_ids = np.asarray(data['splits']['test']['row_ids'])
    checks = {}
    checks['six_final_runs'] = len(summary['results']) == len(completion['records']) == 6
    checks['frozen_protocol'] = hashlib.sha256((run_dir / 'protocol.json').read_bytes()).hexdigest() == summary['protocol_sha256']
    checks['frozen_sources'] = all(
        hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == value
        for name, value in implementation['source_sha256'].items())
    paired_correct = {}
    artifacts = {}
    for result in summary['results']:
        key = f"{result['arm']}_seed{result['seed']}"
        path = run_dir / f'{key}_test_predictions.npz'
        with np.load(path, allow_pickle=False) as z:
            y, pred, probs = z['labels'], z['predictions'], z['probabilities']
            assert np.array_equal(z['row_ids'], expected_ids)
            assert np.array_equal(y, expected_labels)
            assert probs.shape == (9824, 3) and np.isfinite(probs).all()
            assert np.allclose(probs.sum(axis=1), 1, atol=1e-6)
            assert np.array_equal(pred, probs.argmax(1))
            correct = pred == y
            matrix = np.zeros((3, 3), dtype=int)
            np.add.at(matrix, (y, pred), 1)
            f1s = [2 * matrix[k, k] / max(matrix[k, :].sum() + matrix[:, k].sum(), 1)
                   for k in range(3)]
            assert abs(correct.mean() - result['test']['accuracy']) < 1e-12
            assert abs(np.mean(f1s) - result['test']['macro_f1']) < 1e-12
            assert matrix.tolist() == result['test']['confusion_matrix']
            paired_correct[(result['arm'], result['seed'])] = correct
        record = next(r for r in completion['records'] if r['arm'] == result['arm'] and r['seed'] == result['seed'])
        assert [h['epoch'] for h in record['history']] == [1, 2, 3, 4]
        assert all(np.isfinite([h['train_loss'], h['train_accuracy'], h['dev']['accuracy']]).all()
                   for h in record['history'])
        checkpoint_path = run_dir / 'checkpoints' / f'{key}_final.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert checkpoint['record'] == record
        assert checkpoint['protocol_sha256'] == summary['protocol_sha256']
        assert checkpoint['data_sha256'] == summary['data_sha256']
        assert all(torch.isfinite(t).all() for t in checkpoint['model_state'].values())
        artifacts[str(checkpoint_path.relative_to(run_dir))] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        del checkpoint
        artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        checks[f'{key}_metrics_and_final_epoch'] = True
    for seed in protocol['seeds']:
        records = [r for r in completion['records'] if r['seed'] == seed]
        checks[f'seed{seed}_paired_initial_state'] = len({r['initial_state_sha256'] for r in records}) == 1
        checks[f'seed{seed}_same_parameter_count'] = len({r['parameters'] for r in records}) == 1
    differences = [float(paired_correct[('order_regularized', seed)].mean() - paired_correct[('neural', seed)].mean())
                   for seed in protocol['seeds']]
    checks['paired_difference_recomputed'] = np.allclose(differences, summary['order_minus_neural_accuracy_by_seed'], atol=1e-15)
    logic = json.loads((run_dir / 'logic_audit.json').read_text())
    checks['mathematical_checks'] = logic['all_passed'] and all(logic['checks'].values())
    model_smoke = json.loads((run_dir / 'model_smoke.json').read_text())
    checks['fresh_model_smoke'] = model_smoke['passed']
    # Check both trained arms with natural sentences from dev, without re-evaluating test.
    torch.set_num_threads(4)
    batch = data['splits']['dev']
    invariants = {}
    for arm in protocol['arms']:
        checkpoint_path = run_dir / 'checkpoints' / f'{arm}_seed17_final.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert checkpoint['protocol_sha256'] == summary['protocol_sha256']
        assert checkpoint['data_sha256'] == summary['data_sha256']
        model = NeuralOrderNLI(data['embeddings'], **checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state'], strict=True)
        model.eval()
        tokens, lengths = batch['premises'][:64], batch['premise_lengths'][:64]
        with torch.inference_mode():
            hidden, order = model.encode(tokens, lengths)
            hidden_pad, order_pad = model.encode(torch.nn.functional.pad(tokens, (0, 13)), lengths)
            single, single_order = model.encode(tokens[:1], lengths[:1])
        invariants[arm] = {'padding_max_abs': float((hidden - hidden_pad).abs().max()),
                           'batch_max_abs': float((hidden[:1] - single).abs().max()),
                           'order_coordinate_std_mean': float(order.std(dim=0).mean()),
                           'positive_order_coordinates': bool((order > 0).all())}
        checks[f'{arm}_trained_padding_invariance'] = torch.allclose(hidden, hidden_pad, atol=1e-6)
        checks[f'{arm}_trained_batch_invariance'] = torch.allclose(hidden[:1], single, atol=1e-6)
        checks[f'{arm}_trained_noncollapsed'] = invariants[arm]['order_coordinate_std_mean'] > 1e-4
        artifacts[str(checkpoint_path.relative_to(run_dir))] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    report = {'all_passed': all(checks.values()), 'checks': {k: bool(v) for k, v in checks.items()},
              'trained_model_invariants': invariants, 'artifacts_sha256': artifacts,
              'scope': 'recomputed all six saved test predictions; paired starts; final epochs; source/data integrity; trained dev-sentence invariants'}
    (run_dir / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    assert report['all_passed'], report
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(verify_results(args.run_dir), indent=2))
