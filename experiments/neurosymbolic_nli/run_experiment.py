"""Run a frozen, paired SNLI experiment; test is evaluated after all training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .model import NeuralOrderNLI, order_penalty
    from .data import load_data
else:
    from model import NeuralOrderNLI, order_penalty
    from data import load_data


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(split, batch_size, seed=None):
    dataset = TensorDataset(split['premises'], split['hypotheses'],
                            split['premise_lengths'], split['hypothesis_lengths'], split['labels'])
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=seed is not None,
                      generator=generator, num_workers=0, pin_memory=torch.cuda.is_available())


def classification_metrics(labels, predictions):
    matrix = np.bincount(labels * 3 + predictions, minlength=9).reshape(3, 3)
    tp = np.diag(matrix)
    precision = tp / np.maximum(matrix.sum(axis=0), 1)
    recall = tp / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    return {'accuracy': float(tp.sum() / matrix.sum()), 'macro_f1': float(f1.mean()),
            'class_f1': f1.tolist(), 'confusion_matrix': matrix.tolist(), 'examples': int(matrix.sum())}


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    labels, probabilities, energies, reverse = [], [], [], []
    for batch in loader:
        p, h, pl, hl, y = batch
        logits, energy, reverse_energy = model(p.to(device), h.to(device), pl, hl)
        labels.append(y.numpy())
        probabilities.append(logits.softmax(-1).cpu().numpy())
        energies.append(energy.cpu().numpy())
        reverse.append(reverse_energy.cpu().numpy())
    labels = np.concatenate(labels)
    probabilities = np.concatenate(probabilities)
    predictions = probabilities.argmax(axis=1)
    arrays = {'labels': labels, 'predictions': predictions, 'probabilities': probabilities,
              'energy': np.concatenate(energies), 'reverse_energy': np.concatenate(reverse)}
    metrics = classification_metrics(labels, predictions)
    metrics['mean_energy_entailment'] = float(arrays['energy'][labels == 0].mean())
    metrics['mean_energy_non_entailment'] = float(arrays['energy'][labels != 0].mean())
    return metrics, arrays


def append_json(path, entry):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')


def main(run_dir):
    run_dir = Path(run_dir).resolve()
    protocol_bytes = (run_dir / 'protocol.json').read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    cache_path = run_dir / 'data.pt'
    cache = load_data(run_dir)
    data_sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    splits = cache['splits']
    embeddings = cache['embeddings']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ('model.py', 'data.py', 'run_experiment.py')}
    implementation = {'source_sha256': source_hashes, 'precision': 'float32, TF32 disabled',
                      'gradient_clip_max_norm': 1.0, 'embedding_trainable': True,
                      'vocabulary_min_frequency': 2, 'torch_version': str(torch.__version__)}
    implementation_path = run_dir / 'implementation.json'
    if implementation_path.exists():
        assert json.loads(implementation_path.read_text()) == implementation, 'Implementation changed; cannot resume'
    else:
        implementation_path.write_text(json.dumps(implementation, indent=2) + '\n', encoding='utf-8')
    model_config = {key: protocol[key] for key in ('hidden_dim', 'order_dim', 'dropout')}
    dev_loader = make_loader(splits['dev'], protocol['batch_size'])
    ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    log = run_dir / 'training.jsonl'
    records = []
    for seed in protocol['seeds']:
        seed_all(seed)
        initial = NeuralOrderNLI(embeddings, **model_config)
        initial_state = {k: v.detach().clone() for k, v in initial.state_dict().items()}
        initial_hash = hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial_state.values())).hexdigest()
        del initial
        for arm, arm_config in protocol['arms'].items():
            checkpoint_path = ckpt_dir / f'{arm}_seed{seed}_final.pt'
            if checkpoint_path.exists():
                existing = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                assert existing['protocol_sha256'] == protocol_sha and existing['data_sha256'] == data_sha
                print(f'Resume: final checkpoint already exists for {arm}, seed {seed}', flush=True)
                records.append(existing['record'])
                del existing
                continue
            seed_all(seed)
            model = NeuralOrderNLI(embeddings, **model_config)
            model.load_state_dict(initial_state, strict=True)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'])
            history = []
            started = time.monotonic()
            print(f'TRAIN {arm} seed={seed} device={device}', flush=True)
            for epoch in range(protocol['epochs']):
                loader = make_loader(splits['train'], protocol['batch_size'], seed + epoch * 1000)
                model.train()
                loss_total = ce_total = order_total = correct = n_seen = 0
                for step, batch in enumerate(loader, 1):
                    p, h, pl, hl, y = batch
                    p, h, y = p.to(device), h.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits, energy, _ = model(p, h, pl, hl)
                    ce = nn.functional.cross_entropy(logits, y)
                    order = order_penalty(energy, y, protocol['order_margin'])
                    loss = ce + arm_config['order_weight'] * order
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Non-finite loss: {arm}, seed={seed}, epoch={epoch}')
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError('Non-finite gradient norm')
                    optimizer.step()
                    n = y.shape[0]
                    loss_total += float(loss.detach()) * n
                    ce_total += float(ce.detach()) * n
                    order_total += float(order.detach()) * n
                    correct += int((logits.argmax(-1) == y).sum())
                    n_seen += n
                    if step == 1 or step % 50 == 0:
                        print(f'  {arm} seed={seed} epoch={epoch+1} batch={step}/{len(loader)} '
                              f'loss={loss_total/n_seen:.4f} acc={correct/n_seen:.4f}', flush=True)
                dev_metrics, _ = evaluate(model, dev_loader, device)
                epoch_record = {'arm': arm, 'seed': seed, 'epoch': epoch + 1,
                                'train_loss': loss_total / n_seen, 'train_ce': ce_total / n_seen,
                                'train_order': order_total / n_seen, 'train_accuracy': correct / n_seen,
                                'dev': dev_metrics, 'elapsed_seconds': time.monotonic() - started}
                history.append(epoch_record)
                append_json(log, epoch_record)
                print(f'  DONE epoch={epoch+1}, dev acc={dev_metrics["accuracy"]:.4f}, '
                      f'macroF1={dev_metrics["macro_f1"]:.4f}', flush=True)
            record = {'arm': arm, 'seed': seed, 'initial_state_sha256': initial_hash,
                      'parameters': sum(p.numel() for p in model.parameters()), 'history': history,
                      'training_seconds': time.monotonic() - started,
                      'checkpoint': str(checkpoint_path), 'device': str(device)}
            torch.save({'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'model_config': model_config, 'protocol_sha256': protocol_sha,
                        'data_sha256': data_sha, 'record': record}, checkpoint_path)
            records.append(record)
            del model, optimizer
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    # No official-test predictions are computed until every frozen run finishes.
    (run_dir / 'training_complete.json').write_text(json.dumps({
        'protocol_sha256': protocol_sha, 'data_sha256': data_sha, 'records': records
    }, indent=2) + '\n', encoding='utf-8')
    results, paired = [], {}
    test_loader = make_loader(splits['test'], protocol['batch_size'])
    for record in records:
        checkpoint = torch.load(record['checkpoint'], map_location='cpu', weights_only=False)
        model = NeuralOrderNLI(embeddings, **checkpoint['model_config']).to(device)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        metrics, predictions = evaluate(model, test_loader, device)
        name = f'{record["arm"]}_seed{record["seed"]}'
        prediction_path = run_dir / f'{name}_test_predictions.npz'
        np.savez_compressed(prediction_path, **predictions,
                            row_ids=np.asarray(splits['test']['row_ids']))
        result = {k: record[k] for k in ('arm', 'seed', 'parameters', 'training_seconds')}
        result.update({'test': metrics, 'predictions': str(prediction_path)})
        results.append(result)
        paired[(record['arm'], record['seed'])] = predictions['predictions'] == predictions['labels']
        print(f'TEST {name}: accuracy={metrics["accuracy"]:.4f}, macroF1={metrics["macro_f1"]:.4f}', flush=True)
        del model, checkpoint
    by_arm = {}
    for arm in protocol['arms']:
        arm_results = [r for r in results if r['arm'] == arm]
        by_arm[arm] = {metric: {'mean': float(np.mean([r['test'][metric] for r in arm_results])),
                                'std': float(np.std([r['test'][metric] for r in arm_results], ddof=1))}
                       for metric in ('accuracy', 'macro_f1')}
    per_seed_differences = [float(paired[('order_regularized', s)].mean() - paired[('neural', s)].mean())
                            for s in protocol['seeds']]
    by_example = np.stack([paired[('order_regularized', s)].astype(float)
                           - paired[('neural', s)].astype(float) for s in protocol['seeds']]).mean(0)
    rng = np.random.default_rng(20260915)
    bootstrap = np.array([by_example[rng.integers(0, len(by_example), len(by_example))].mean()
                          for _ in range(2000)])
    summary = {'protocol_sha256': protocol_sha, 'data_sha256': data_sha, 'results': results,
               'aggregate': by_arm, 'order_minus_neural_accuracy_by_seed': per_seed_differences,
               'order_minus_neural_accuracy_mean': float(np.mean(per_seed_differences)),
               'paired_example_bootstrap_95ci': np.quantile(bootstrap, [0.025, 0.975]).tolist(),
               'ci_scope': 'paired test-example resampling conditional on the three fitted seed pairs; not seed uncertainty',
               'notes': ['Official SNLI 3-class classification, not causal-effect estimation.',
                         'Final-epoch checkpoints only; no test-based tuning or checkpoint selection.']}
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'aggregate': by_arm, 'paired_difference': summary['order_minus_neural_accuracy_mean'],
                      'ci': summary['paired_example_bootstrap_95ci']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    main(parser.parse_args().run_dir)
