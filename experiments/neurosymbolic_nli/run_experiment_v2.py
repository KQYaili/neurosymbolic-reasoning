"""Run frozen, paired SNLI experiment for V2 across multiple arms and seeds."""
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

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.neurosymbolic_nli.model_v2 import (
    NeurosymbolicNLIv2, v1_order_penalty, v2_threeway_loss
)
from experiments.neurosymbolic_nli.data import load_data


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(split, batch_size, seed=None):
    dataset = TensorDataset(
        split['premises'], split['hypotheses'],
        split['premise_lengths'], split['hypothesis_lengths'],
        split['labels']
    )
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=seed is not None,
        generator=generator, num_workers=0, pin_memory=torch.cuda.is_available()
    )


def classification_metrics(labels, predictions):
    matrix = np.bincount(labels * 3 + predictions, minlength=9).reshape(3, 3)
    tp = np.diag(matrix)
    precision = tp / np.maximum(matrix.sum(axis=0), 1)
    recall = tp / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    return {
        'accuracy': float(tp.sum() / matrix.sum()),
        'macro_f1': float(f1.mean()),
        'class_f1': f1.tolist(),
        'confusion_matrix': matrix.tolist(),
        'examples': int(matrix.sum())
    }


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    labels, probabilities, fwd_energies, rev_energies, cos_sims, jaccards = [], [], [], [], [], []
    for batch in loader:
        p, h, pl, hl, y = batch
        logits, fwd_e, rev_e, cos_sim, jaccard = model(p.to(device), h.to(device), pl, hl)
        labels.append(y.numpy())
        probabilities.append(logits.softmax(-1).cpu().numpy())
        fwd_energies.append(fwd_e.cpu().numpy())
        rev_energies.append(rev_e.cpu().numpy())
        cos_sims.append(cos_sim.cpu().numpy())
        jaccards.append(jaccard.cpu().numpy())

    labels = np.concatenate(labels)
    probabilities = np.concatenate(probabilities)
    predictions = probabilities.argmax(axis=1)
    fwd_arr = np.concatenate(fwd_energies)
    rev_arr = np.concatenate(rev_energies)
    cos_arr = np.concatenate(cos_sims)
    jac_arr = np.concatenate(jaccards)

    arrays = {
        'labels': labels,
        'predictions': predictions,
        'probabilities': probabilities,
        'fwd_energy': fwd_arr,
        'rev_energy': rev_arr,
        'cos_sim': cos_arr,
        'jaccard': jac_arr
    }
    metrics = classification_metrics(labels, predictions)
    # Energy breakdowns per class
    metrics['mean_fwd_energy_entailment'] = float(fwd_arr[labels == 0].mean())
    metrics['mean_fwd_energy_contradiction'] = float(fwd_arr[labels == 1].mean())
    metrics['mean_fwd_energy_neutral'] = float(fwd_arr[labels == 2].mean())
    metrics['mean_cos_sim_contradiction'] = float(cos_arr[labels == 1].mean())
    metrics['mean_cos_sim_entailment'] = float(cos_arr[labels == 0].mean())

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

    source_hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('model_v2.py', 'data.py', 'run_experiment_v2.py')
        if (Path(__file__).with_name(name)).exists()
    }
    implementation = {
        'source_sha256': source_hashes,
        'precision': 'float32, TF32 disabled',
        'gradient_clip_max_norm': 1.0,
        'embedding_trainable': True,
        'vocabulary_min_frequency': 2,
        'torch_version': str(torch.__version__)
    }
    implementation_path = run_dir / 'implementation.json'
    implementation_path.write_text(json.dumps(implementation, indent=2) + '\n', encoding='utf-8')

    model_config_base = {
        key: protocol[key] for key in ('hidden_dim', 'order_dim', 'dropout', 'geo_bottleneck')
    }
    dev_loader = make_loader(splits['dev'], protocol['batch_size'])
    ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    log = run_dir / 'training.jsonl'
    records = []

    for seed in protocol['seeds']:
        # Generate initial state dicts for each architecture family to ensure strict pairing
        initial_states = {}
        for enc_type in ('gru', 'bigru_attn'):
            seed_all(seed)
            cfg = dict(model_config_base, encoder_type=enc_type)
            init_model = NeurosymbolicNLIv2(embeddings, **cfg)
            initial_states[enc_type] = {
                'state': {k: v.detach().clone() for k, v in init_model.state_dict().items()},
                'hash': hashlib.sha256(b''.join(v.numpy().tobytes() for v in init_model.state_dict().values())).hexdigest(),
                'config': cfg
            }
            del init_model

        for arm, arm_config in protocol['arms'].items():
            checkpoint_path = ckpt_dir / f'{arm}_seed{seed}_final.pt'
            enc_type = arm_config['encoder']
            arm_model_cfg = initial_states[enc_type]['config']

            if checkpoint_path.exists():
                existing = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                assert existing['protocol_sha256'] == protocol_sha and existing['data_sha256'] == data_sha
                print(f'Resume: final checkpoint already exists for {arm}, seed {seed}', flush=True)
                records.append(existing['record'])
                del existing
                continue

            seed_all(seed)
            model = NeurosymbolicNLIv2(embeddings, **arm_model_cfg)
            model.load_state_dict(initial_states[enc_type]['state'], strict=True)
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'])
            history = []
            started = time.monotonic()
            print(f'TRAIN {arm} (enc={enc_type}) seed={seed} device={device}', flush=True)

            for epoch in range(protocol['epochs']):
                loader = make_loader(splits['train'], protocol['batch_size'], seed + epoch * 1000)
                model.train()
                loss_total = ce_total = aux_total = correct = n_seen = 0

                for step, batch in enumerate(loader, 1):
                    p, h, pl, hl, y = batch
                    p, h, y = p.to(device), h.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)

                    logits, fwd_e, rev_e, cos_sim, jaccard = model(p, h, pl, hl)
                    ce = nn.functional.cross_entropy(logits, y)

                    loss_type = arm_config.get('loss_type', 'none')
                    aux_weight = arm_config.get('aux_weight', 0.0)

                    if loss_type == 'v1':
                        aux_loss = v1_order_penalty(fwd_e, y, protocol.get('order_margin', 0.2))
                    elif loss_type == 'v2_threeway':
                        aux_loss = v2_threeway_loss(fwd_e, rev_e, cos_sim, jaccard, y)
                    else:
                        aux_loss = torch.tensor(0.0, device=device)

                    loss = ce + aux_weight * aux_loss
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
                    aux_total += float(aux_loss.detach()) * n
                    correct += int((logits.argmax(-1) == y).sum())
                    n_seen += n

                    if step == 1 or step % 50 == 0:
                        print(f'  {arm} seed={seed} epoch={epoch+1} batch={step}/{len(loader)} '
                              f'loss={loss_total/n_seen:.4f} ce={ce_total/n_seen:.4f} acc={correct/n_seen:.4f}', flush=True)

                dev_metrics, _ = evaluate(model, dev_loader, device)
                epoch_record = {
                    'arm': arm, 'seed': seed, 'epoch': epoch + 1,
                    'train_loss': loss_total / n_seen,
                    'train_ce': ce_total / n_seen,
                    'train_aux': aux_total / n_seen,
                    'train_accuracy': correct / n_seen,
                    'dev': dev_metrics,
                    'elapsed_seconds': time.monotonic() - started
                }
                history.append(epoch_record)
                append_json(log, epoch_record)
                print(f'  DONE epoch={epoch+1}, dev acc={dev_metrics["accuracy"]:.4f}, '
                      f'macroF1={dev_metrics["macro_f1"]:.4f}', flush=True)

            record = {
                'arm': arm,
                'seed': seed,
                'encoder': enc_type,
                'loss_type': arm_config['loss_type'],
                'aux_weight': arm_config['aux_weight'],
                'initial_state_sha256': initial_states[enc_type]['hash'],
                'parameters': sum(p.numel() for p in model.parameters()),
                'history': history,
                'training_seconds': time.monotonic() - started,
                'checkpoint': str(checkpoint_path),
                'device': str(device)
            }
            torch.save({
                'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'model_config': arm_model_cfg,
                'protocol_sha256': protocol_sha,
                'data_sha256': data_sha,
                'record': record
            }, checkpoint_path)
            records.append(record)
            del model, optimizer
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # Step 2: Frozen Test Evaluation (strictly evaluated after all training completes)
    (run_dir / 'training_complete.json').write_text(json.dumps({
        'protocol_sha256': protocol_sha,
        'data_sha256': data_sha,
        'records': records
    }, indent=2) + '\n', encoding='utf-8')

    results, paired = [], {}
    test_loader = make_loader(splits['test'], protocol['batch_size'])

    for record in records:
        checkpoint = torch.load(record['checkpoint'], map_location='cpu', weights_only=False)
        model = NeurosymbolicNLIv2(embeddings, **checkpoint['model_config']).to(device)
        model.load_state_dict(checkpoint['model_state'], strict=True)

        metrics, predictions = evaluate(model, test_loader, device)
        name = f'{record["arm"]}_seed{record["seed"]}'
        prediction_path = run_dir / f'{name}_test_predictions.npz'
        np.savez_compressed(
            prediction_path, **predictions,
            row_ids=np.asarray(splits['test']['row_ids'])
        )
        result = {k: record[k] for k in ('arm', 'seed', 'encoder', 'loss_type', 'parameters', 'training_seconds')}
        result.update({'test': metrics, 'predictions': str(prediction_path)})
        results.append(result)
        paired[(record['arm'], record['seed'])] = predictions['predictions'] == predictions['labels']
        print(f'TEST {name}: accuracy={metrics["accuracy"]:.4f}, macroF1={metrics["macro_f1"]:.4f}', flush=True)
        del model, checkpoint

    by_arm = {}
    for arm in protocol['arms']:
        arm_results = [r for r in results if r['arm'] == arm]
        by_arm[arm] = {
            metric: {
                'mean': float(np.mean([r['test'][metric] for r in arm_results])),
                'std': float(np.std([r['test'][metric] for r in arm_results], ddof=1))
            }
            for metric in ('accuracy', 'macro_f1')
        }

    # Paired differences within GRU track:
    # v1_order_gru vs neural_gru
    diff_v1_gru = [
        float(paired[('v1_order_gru', s)].mean() - paired[('neural_gru', s)].mean())
        for s in protocol['seeds']
    ]
    # v2_order_gru vs neural_gru
    diff_v2_gru = [
        float(paired[('v2_order_gru', s)].mean() - paired[('neural_gru', s)].mean())
        for s in protocol['seeds']
    ]
    # Paired differences within BiGRU track:
    # v2_order_bigru_attn vs neural_bigru_attn
    diff_v2_bigru = [
        float(paired[('v2_order_bigru_attn', s)].mean() - paired[('neural_bigru_attn', s)].mean())
        for s in protocol['seeds']
    ]

    # Paired bootstrap 95% CI for v2_order_gru vs neural_gru
    by_example_gru = np.stack([
        paired[('v2_order_gru', s)].astype(float) - paired[('neural_gru', s)].astype(float)
        for s in protocol['seeds']
    ]).mean(0)
    rng = np.random.default_rng(20260915)
    bs_gru = np.array([
        by_example_gru[rng.integers(0, len(by_example_gru), len(by_example_gru))].mean()
        for _ in range(2000)
    ])

    summary = {
        'protocol_sha256': protocol_sha,
        'data_sha256': data_sha,
        'results': results,
        'aggregate': by_arm,
        'comparisons': {
            'v1_minus_neural_gru': {
                'by_seed': diff_v1_gru,
                'mean': float(np.mean(diff_v1_gru))
            },
            'v2_minus_neural_gru': {
                'by_seed': diff_v2_gru,
                'mean': float(np.mean(diff_v2_gru)),
                'bootstrap_95ci': np.quantile(bs_gru, [0.025, 0.975]).tolist()
            },
            'v2_minus_neural_bigru_attn': {
                'by_seed': diff_v2_bigru,
                'mean': float(np.mean(diff_v2_bigru))
            }
        },
        'notes': [
            'Official SNLI 3-class classification, paired evaluation across 3 seeds (17, 29, 43).',
            'Track A compares neural_gru vs v1_order_gru vs v2_order_gru under identical weights.',
            'Track B compares neural_bigru_attn vs v2_order_bigru_attn under identical weights.'
        ]
    }
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print("\n=== SUMMARY COMPLETE ===")
    print(json.dumps({'aggregate': by_arm, 'comparisons': summary['comparisons']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    main(parser.parse_args().run_dir)
