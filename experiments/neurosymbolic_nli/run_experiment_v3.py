"""Frozen, resumable V3 training and evaluation. No V1/V2 artifacts are changed."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neurosymbolic_nli.data import load_data, sha256_file
from experiments.neurosymbolic_nli.model_v3 import AlignedNLI, reverse_partial_loss, contradiction_consistency
from experiments.neurosymbolic_nli.model_v2 import NeurosymbolicNLIv2


def write_json(path, value, immutable=False):
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if immutable and path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise RuntimeError(f'Frozen artifact changed: {path}')
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def loader(split, batch_size, seed=None):
    ds = TensorDataset(*[split[k] for k in ('premises', 'hypotheses', 'premise_lengths', 'hypothesis_lengths', 'labels')])
    return DataLoader(ds, batch_size=batch_size, shuffle=seed is not None, num_workers=0,
                      generator=torch.Generator().manual_seed(seed) if seed is not None else None)


def on_device(batch, device, separate_widths=True):
    p, h, pl, hl, y = batch
    if separate_widths:
        p, h = p[:, :int(pl.max())], h[:, :int(hl.max())]
    return p.to(device), h.to(device), pl, hl, y.to(device)


def metrics(y, probs, reverse_probs):
    pred = probs.argmax(-1)
    cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    tp = np.diag(cm)
    f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
    confidence = probs.max(-1)
    ece = 0.0
    for i in range(10):
        take = (confidence >= i / 10) & ((confidence < (i+1)/10) if i < 9 else (confidence <= 1))
        if take.any():
            ece += take.mean() * abs((pred[take] == y[take]).mean() - confidence[take].mean())
    return {'accuracy': float((pred == y).mean()), 'macro_f1': float(f1.mean()),
            'class_f1': f1.tolist(), 'confusion_matrix': cm.tolist(), 'examples': len(y),
            'nll': float(-np.log(np.maximum(probs[np.arange(len(y)), y], 1e-15)).mean()),
            'ece_10bins': float(ece),
            'contradiction_probability_swap_gap': float(np.abs(probs[:, 1] - reverse_probs[:, 1]).mean()),
            'contradiction_decision_swap_disagreement': float(((pred == 1) != (reverse_probs.argmax(-1) == 1)).mean())}


@torch.inference_mode()
def evaluate(model, split, device, batch_size, alignment='learned', legacy=False):
    model.eval()
    ps, rps, ys = [], [], []
    for batch in loader(split, batch_size):
        p, h, pl, hl, y = on_device(batch, device, separate_widths=not legacy)
        if legacy:
            logits = model(p, h, pl, hl)[0]
            reverse = model(h, p, hl, pl)[0]
        else:
            logits, reverse = model(p, h, pl, hl, alignment=alignment)
        ps.append(logits.softmax(-1).cpu().numpy())
        rps.append(reverse.softmax(-1).cpu().numpy())
        ys.append(y.cpu().numpy())
    probs, rp, y = np.concatenate(ps), np.concatenate(rps), np.concatenate(ys)
    result = {'labels': y, 'predictions': probs.argmax(-1), 'probabilities': probs,
              'reverse_probabilities': rp, 'row_ids': np.asarray(split['row_ids'])}
    return metrics(y, probs, rp), result


def make_contract(run, protocol):
    names = ['model_v3.py', 'run_experiment_v3.py', 'data.py', 'prepare_holdout_v3.py', 'model_v2.py']
    contract = {'protocol_sha256': sha256_file(run / 'protocol.json'),
                'train_data_sha256': sha256_file(ROOT / protocol['train_cache'] / 'data.pt'),
                'holdout_sha256': sha256_file(run / 'holdout.pt'),
                'holdout_manifest_sha256': sha256_file(run / 'holdout_manifest.json'),
                'source_sha256': {name: sha256_file(Path(__file__).with_name(name)) for name in names},
                'torch': str(torch.__version__), 'numpy': str(np.__version__),
                'precision': 'FP32; TF32 disabled; deterministic algorithms enabled'}
    write_json(run / 'implementation.json', contract, immutable=True)
    snapshots = run / 'sources'
    snapshots.mkdir(exist_ok=True)
    for name in names:
        target = snapshots / name
        if target.exists():
            assert sha256_file(target) == contract['source_sha256'][name]
        else:
            target.write_bytes(Path(__file__).with_name(name).read_bytes())
    return sha256_file(run / 'implementation.json')


def state_hash(state):
    dig = hashlib.sha256()
    for k, v in state.items():
        dig.update(k.encode())
        dig.update(v.detach().cpu().numpy().tobytes())
    return dig.hexdigest()


def train(run, protocol, cache, contract_sha, device):
    records = []
    (run / 'checkpoints').mkdir(exist_ok=True)
    for seed in protocol['seeds']:
        seed_all(seed)
        initial = AlignedNLI(cache['embeddings'], **protocol['model_config'])
        initial_state = {k: v.clone() for k, v in initial.state_dict().items()}
        initial_sha = state_hash(initial_state)
        del initial
        for arm, config in protocol['arms'].items():
            path = run / 'checkpoints' / f'{arm}_seed{seed}_final.pt'
            if path.exists():
                ckpt = torch.load(path, map_location='cpu', weights_only=True)
                assert ckpt['contract_sha256'] == contract_sha
                assert ckpt['record']['initial_state_sha256'] == initial_sha
                records.append(ckpt['record'])
                print(f'RESUME verified final {arm} seed={seed}', flush=True)
                continue
            seed_all(seed)
            model = AlignedNLI(cache['embeddings'], **protocol['model_config']).to(device)
            model.load_state_dict(initial_state, strict=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'])
            history = []
            started = time.monotonic()
            for epoch in range(protocol['epochs']):
                model.train()
                sums = np.zeros(5)
                n = 0
                for batch in loader(cache['splits']['train'], protocol['batch_size'], seed + epoch * 1000):
                    p, h, pl, hl, y = on_device(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    logits, reverse = model(p, h, pl, hl, alignment=config['alignment'])
                    ce = F.cross_entropy(logits, y)
                    partial = reverse_partial_loss(reverse, y)
                    consistency = contradiction_consistency(logits, reverse)
                    loss = ce + config['reverse_weight'] * partial + config['consistency_weight'] * consistency
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite loss')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['gradient_clip'])
                    if not bool(torch.isfinite(norm)):
                        raise FloatingPointError('Nonfinite gradient')
                    optimizer.step()
                    size = len(y)
                    sums += np.array([float(loss.detach()), float(ce.detach()), float(partial.detach()),
                                      float(consistency.detach()), float((logits.argmax(-1) == y).float().mean())]) * size
                    n += size
                dev, _ = evaluate(model, cache['splits']['dev'], device, protocol['batch_size'], config['alignment'])
                row = {'arm': arm, 'seed': seed, 'epoch': epoch + 1, 'seen': n,
                       **dict(zip(['loss', 'ce', 'reverse_partial', 'consistency', 'train_accuracy'], (sums/n).tolist())),
                       'dev': dev, 'elapsed_seconds': time.monotonic() - started}
                history.append(row)
                with (run / 'training.jsonl').open('a', encoding='utf-8') as out:
                    out.write(json.dumps(row) + '\n')
                print(f'{arm} seed={seed} epoch={epoch+1}/4 train={row["train_accuracy"]:.4f} '
                      f'dev={dev["accuracy"]:.4f} swap_gap={dev["contradiction_probability_swap_gap"]:.4f} '
                      f'sec={row["elapsed_seconds"]:.1f}', flush=True)
            record = {'arm': arm, 'seed': seed, 'history': history, 'initial_state_sha256': initial_sha,
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'checkpoint': str(path.relative_to(run)), 'training_seconds': time.monotonic()-started}
            saved = {'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                     'model_config': protocol['model_config'], 'contract_sha256': contract_sha, 'record': record}
            temp = path.with_suffix('.tmp')
            torch.save(saved, temp)
            temp.replace(path)
            records.append(record)
            del model, optimizer
    write_json(run / 'training_complete.json', {'contract_sha256': contract_sha, 'records': records}, immutable=True)
    candidates = ['aligned_ce', 'aligned_logic']
    means = {a: float(np.mean([r['history'][-1]['dev']['accuracy'] for r in records if r['arm'] == a])) for a in candidates}
    chosen = max(candidates, key=lambda a: means[a])
    selection = {'selected_arm': chosen, 'default_seed': 17, 'final_dev_means': means,
                 'criterion': protocol['selection'], 'contract_sha256': contract_sha,
                 'checkpoint_sha256': {r['checkpoint']: sha256_file(run / r['checkpoint']) for r in records}}
    write_json(run / 'selection.json', selection, immutable=True)
    print('DEV-ONLY SELECTION ' + json.dumps(selection['final_dev_means']) + ' -> ' + chosen, flush=True)
    return records


def paired_comparison(run, split_name, a, b, seeds, groups, protocol):
    changes = []
    for seed in seeds:
        with np.load(run / f'{a}_seed{seed}_{split_name}.npz') as x, np.load(run / f'{b}_seed{seed}_{split_name}.npz') as y:
            assert np.array_equal(x['row_ids'], y['row_ids']) and np.array_equal(x['labels'], y['labels'])
            changes.append((x['predictions'] == x['labels']).astype(float) - (y['predictions'] == y['labels']).astype(float))
    array = np.asarray(changes)
    _, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=array.mean(0))
    counts = np.bincount(inverse)
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    boot = []
    for _ in range(protocol['bootstrap_repetitions']):
        take = rng.integers(0, len(counts), len(counts))
        boot.append(sums[take].sum()/counts[take].sum())
    return {'by_seed': array.mean(1).tolist(), 'mean': float(array.mean()),
            'paired_group_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist(), 'groups': len(counts),
            'scope': protocol['confidence_interval']}


def assess(run, protocol, cache, contract_sha, device):
    complete = json.loads((run / 'training_complete.json').read_text())
    selection = json.loads((run / 'selection.json').read_text())
    assert complete['contract_sha256'] == selection['contract_sha256'] == contract_sha
    for path, checksum in selection['checkpoint_sha256'].items():
        assert sha256_file(run / path) == checksum
    holdout = torch.load(run / 'holdout.pt', map_location='cpu', weights_only=True)
    holdout = holdout.get('split', holdout.get('holdout', holdout))
    splits = {'dev': cache['splits']['dev'], 'holdout': holdout, 'test': cache['splits']['test']}
    records = list(complete['records'])
    v2 = ROOT / 'runs/neurosymbolic_nli/20260915_order_v2'
    # Verify the reference implementation before using its frozen checkpoint files.
    v2_impl = json.loads((v2 / 'implementation.json').read_text())
    for name, sha in v2_impl['source_sha256'].items():
        assert sha256_file(Path(__file__).with_name(name)) == sha
    for seed in protocol['seeds']:
        for name, old in [('v2_neural_gru', 'neural_gru'), ('v2_order_gru', 'v2_order_gru')]:
            records.append({'arm': name, 'seed': seed, 'legacy': True,
                            'checkpoint': str(v2 / 'checkpoints' / f'{old}_seed{seed}_final.pt')})
    results = []
    for record in records:
        legacy = record.get('legacy', False)
        ckpt_path = Path(record['checkpoint']) if legacy else run / record['checkpoint']
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        if legacy:
            assert ckpt['protocol_sha256'] == sha256_file(v2 / 'protocol.json')
            assert ckpt['data_sha256'] == sha256_file(v2 / 'data.pt')
            model = NeurosymbolicNLIv2(cache['embeddings'], **ckpt['model_config']).to(device)
            mode = 'learned'
        else:
            assert ckpt['contract_sha256'] == contract_sha
            model = AlignedNLI(cache['embeddings'], **ckpt['model_config']).to(device)
            mode = protocol['arms'][record['arm']]['alignment']
        model.load_state_dict(ckpt['model_state'], strict=True)
        row = {'arm': record['arm'], 'seed': record['seed'], 'checkpoint_sha256': sha256_file(ckpt_path),
               'parameters': sum(p.numel() for p in model.parameters())}
        for split_name, split in splits.items():
            result, arrays = evaluate(model, split, device, protocol['batch_size'], mode, legacy=legacy)
            path = run / f'{record["arm"]}_seed{record["seed"]}_{split_name}.npz'
            np.savez_compressed(path, **arrays)
            row[split_name] = result
        results.append(row)
        print(f'EVAL {record["arm"]} seed={record["seed"]} '
              f'holdout={row["holdout"]["accuracy"]:.4f} test(exploratory)={row["test"]["accuracy"]:.4f}', flush=True)
        del model, ckpt
    # Group all evaluations by image ID when available, otherwise normalized premise.
    groups = {}
    for name, split in splits.items():
        groups[name] = split.get('image_ids', [' '.join(s.casefold().split()) for s in split['premise_texts']])
    comparisons = {split: {f'{a}_minus_{b}': paired_comparison(run, split, a, b, protocol['seeds'], groups[split], protocol)
                           for a, b in protocol['comparison_pairs']} for split in splits}
    aggregate = {split: {arm: {metric: {'mean': float(np.mean([r[split][metric] for r in results if r['arm'] == arm])),
                                        'std': float(np.std([r[split][metric] for r in results if r['arm'] == arm], ddof=1))}
                                for metric in ['accuracy', 'macro_f1', 'nll', 'ece_10bins',
                                               'contradiction_probability_swap_gap', 'contradiction_decision_swap_disagreement']}
                         for arm in dict.fromkeys(r['arm'] for r in results)} for split in splits}
    summary = {'contract_sha256': contract_sha, 'selection': selection, 'results': results,
               'aggregate': aggregate, 'comparisons': comparisons,
               'holdout_scope': protocol['holdout'], 'test_scope': protocol['official_test']}
    write_json(run / 'summary.json', summary)
    print(json.dumps({'selected': selection['selected_arm'], 'holdout': aggregate['holdout'],
                      'logic_difference': comparisons['holdout']['aligned_logic_minus_aligned_ce']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='runs/neurosymbolic_nli/20260915_review_v3')
    parser.add_argument('--phase', choices=['train', 'evaluate', 'all'], default='all')
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    device = configure()
    contract_sha = make_contract(run, protocol)
    cache = load_data(ROOT / protocol['train_cache'])
    if args.phase in ('train', 'all'):
        train(run, protocol, cache, contract_sha, device)
    if args.phase in ('evaluate', 'all'):
        assess(run, protocol, cache, contract_sha, device)


if __name__ == '__main__':
    main()
