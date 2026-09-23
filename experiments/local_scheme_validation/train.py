"""Local LoRA post-training on labelled anchors and mathematically valid relations.

Training never opens oracle.json. Final-step adapters, frozen manifests and exact
batch schedules make the four arms comparable. This is single-token SFT, not RL.
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'runs/local_scheme_validation/20260923_relational_v2'
LABELS = ['A', 'B', 'C', 'D']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4*1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def save_json(path, value, immutable=False):
    if immutable and path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == value, f'Frozen artifact changed: {path}'
        return
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def base_and_tokenizer(protocol, with_adapter=True):
    snapshot = Path(protocol['snapshot'])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    ids = [tokenizer.encode(x, add_special_tokens=False) for x in LABELS]
    assert all(len(x) == 1 for x in ids), ids
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                                dtype=torch.bfloat16 if with_adapter else torch.float32,
                                                attn_implementation='sdpa').cuda()
    model.config.use_cache = False
    if with_adapter:
        model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', **protocol['lora']))
    return model, tokenizer, torch.tensor([x[0] for x in ids], device='cuda')


def encode_prompt(tokenizer, prompt):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                                         add_generation_prompt=True)


def collate(token_lists, pad_id):
    width = max(map(len, token_lists))
    inputs = torch.full((len(token_lists), width), pad_id, dtype=torch.long, device='cuda')
    mask = torch.zeros_like(inputs)
    for row, tokens in enumerate(token_lists):
        inputs[row, -len(tokens):] = torch.tensor(tokens, device='cuda')
        mask[row, -len(tokens):] = 1
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
    return {'input_ids': inputs, 'attention_mask': mask, 'position_ids': positions}


def last_logits(model, batch):
    return model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1].float()


def semantic_js(a_logits, b_logits, a_options, b_options):
    """Map probabilities from displayed label position to semantic option identity."""
    pa, pb = a_logits.softmax(-1), b_logits.softmax(-1)
    oa = torch.tensor(a_options, device=pa.device)
    ob = torch.tensor(b_options, device=pa.device)
    pa, pb = pa.gather(1, oa.argsort(1)), pb.gather(1, ob.argsort(1))
    middle = (pa + pb) / 2
    return .5 * ((pa * (pa.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum(-1)
                 + (pb * (pb.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum(-1)).mean()


def allowed_mass(log_probs, allowed, answer_ids):
    mask = torch.tensor([[label in row for label in LABELS] for row in allowed], device=log_probs.device)
    assert bool(mask.any(-1).all())
    return -torch.logsumexp(log_probs[:, answer_ids].masked_fill(~mask, -torch.inf), dim=-1)


def train_pools(data):
    # Deliberately no oracle import or reads, including for hidden training labels.
    source = data['splits']['train']
    labelled = {task: [b for b in source[task] if b['variants'][0].get('supervised_label') is not None]
                for task in ('nli', 'mcqa')}
    unlabelled = {task: [b for b in source[task] if b['variants'][0].get('supervised_label') is None]
                  for task in ('nli', 'mcqa')}
    assert all(labelled.values()) and all(unlabelled.values())
    for rows in unlabelled.values():
        for b in rows:
            assert all(v.get('supervised_label') is None for v in b['variants'])
    return labelled, unlabelled


def schedule(data, protocol, seed):
    labelled, unlabelled = train_pools(data)
    rng = random.Random(seed)
    schedules = []
    for step in range(protocol['steps']):
        n = [rng.choice(labelled['nli']) for _ in range(protocol['bundles_per_group'])]
        m = [rng.choice(labelled['mcqa']) for _ in range(protocol['bundles_per_group'])]
        u = [rng.choice(unlabelled['mcqa']) for _ in range(protocol['bundles_per_group'])]
        un = [rng.choice(unlabelled['nli']) for _ in range(protocol['bundles_per_group'])]
        transformed = 1 + step % 3
        rows = [(b, 0) for b in n] + [(b, 1) for b in n]
        rows += [(b, 0) for b in m] + [(b, transformed) for b in m]
        rows += [(b, 0) for b in u] + [(b, transformed) for b in u]
        rows += [(b, 0) for b in un] + [(b, 1) for b in un]
        schedules.append(rows)
    return schedules


def objective(logits, rows, answer_ids, weights, k):
    logp = logits.log_softmax(-1)
    labels = lambda pairs: torch.tensor([LABELS.index(b['variants'][v]['supervised_label']) for b, v in pairs], device=logits.device)
    if weights.get('compute_matched', False):
        loss = F.nll_loss(logp, answer_ids[labels(rows)])
        return loss, {'anchor': loss}
    nli_label, mc_label = labels(rows[:k]), labels(rows[2*k:3*k])
    anchor = .5 * (F.nll_loss(logp[:k], answer_ids[nli_label]) + F.nll_loss(logp[2*k:3*k], answer_ids[mc_label]))
    nli_allowed = [b['variants'][v]['allowed_labels'] for b, v in rows[k:2*k]]
    set_losses = allowed_mass(logp[k:2*k], nli_allowed, answer_ids)
    exact_mask = torch.tensor([len(a) == 1 for a in nli_allowed], device=logits.device)
    # All NLI terms retain the same denominator; a new rule doesn't silently reweight C.
    exact_nli = (set_losses * exact_mask).mean()
    partial_nli = (set_losses * ~exact_mask).mean()
    exact_mc = F.nll_loss(logp[3*k:4*k], answer_ids[labels(rows[3*k:4*k])])
    choices = [[b['variants'][v]['option_ids'] for b, v in group] for group in (rows[4*k:5*k], rows[5*k:6*k])]
    equivariance = semantic_js(logits[4*k:5*k, answer_ids], logits[5*k:6*k, answer_ids], *choices)
    # Unlabelled NLI uses only a gold-free relation marginal, never a guessed label.
    pf = logits[6*k:7*k, answer_ids[:3]].softmax(-1)[:, 1]
    pr = logits[7*k:8*k, answer_ids[:3]].softmax(-1)[:, 1]
    a, b = torch.stack((pf, 1-pf), -1), torch.stack((pr, 1-pr), -1)
    mid = (a+b)/2
    swap_js = .5 * ((a*(a.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(-1)
                    +(b*(b.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(-1)).mean()
    loss = anchor + weights['exact_weight'] * .5 * (exact_nli + exact_mc)
    loss = loss + weights['partial_weight'] * .5 * partial_nli + weights['equivariance_weight'] * equivariance
    loss = loss + weights['swap_weight'] * swap_js
    return loss, {'anchor': anchor, 'exact_nli': exact_nli, 'partial_nli': partial_nli,
                  'exact_mc': exact_mc, 'equivariance': equivariance, 'unlabelled_nli_swap_js': swap_js}


def contract(run, protocol):
    names = ['train.py', 'synthetic_data.py', 'extend_evaluation.py']
    value = {'protocol_sha256': sha256(run / 'protocol.json'), 'data_sha256': sha256(run / 'train_public.json'),
             'data_manifest_sha256': sha256(run / 'data_manifest.json'),
             'sources': {name: sha256(Path(__file__).with_name(name)) for name in names},
             'base_files': {name: sha256(Path(protocol['snapshot']) / name) for name in
                            ['model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json']},
             'torch_version': str(torch.__version__), 'train_reads_oracle': False}
    save_json(run / 'implementation.json', value, immutable=True)
    sources = run / 'sources'
    sources.mkdir(exist_ok=True)
    for name in names:
        target = sources / name
        if target.exists():
            assert sha256(target) == value['sources'][name]
        else:
            target.write_bytes(Path(__file__).with_name(name).read_bytes())
    return sha256(run / 'implementation.json')


def run_training(run):
    protected = {(run/'oracle.json').resolve(), (run/'data.json').resolve()}
    def deny_hidden_labels(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            if Path(os.fsdecode(args[0])).resolve() in protected:
                raise PermissionError('Training cannot open evaluator-only data or oracle')
    sys.addaudithook(deny_hidden_labels)
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    data = json.loads((run / 'train_public.json').read_text(encoding='utf-8'))
    setup(protocol['seeds'][0])
    contract_sha = contract(run, protocol)
    records = []
    for seed in protocol['seeds']:
        common_batches = schedule(data, protocol, seed)
        schedule_sha = hashlib.sha256(json.dumps([[(b['id'], v) for b, v in batch] for batch in common_batches]).encode()).hexdigest()
        for arm, weights in protocol['arms'].items():
            destination = run / 'adapters' / f'{arm}_seed{seed}'
            record_file = destination / 'record.json'
            if record_file.exists():
                record = json.loads(record_file.read_text())
                assert record['contract_sha256'] == contract_sha and record['schedule_sha256'] == schedule_sha
                assert sha256(destination / 'adapter_model.safetensors') == record['adapter_sha256']
                records.append(record)
                print(f'RESUME {arm} {seed}', flush=True)
                continue
            if destination.exists():
                raise RuntimeError(f'Incomplete adapter directory requires inspection: {destination}')
            setup(seed)
            model, tokenizer, answer_ids = base_and_tokenizer(protocol)
            adapter = get_peft_model_state_dict(model)
            initial_sha = hashlib.sha256(b''.join(k.encode()+v.detach().cpu().float().numpy().tobytes() for k,v in adapter.items())).hexdigest()
            token_cache = {}
            for batch in common_batches:
                for b, v in batch:
                    key = (b['id'], v)
                    if key not in token_cache:
                        token_cache[key] = encode_prompt(tokenizer, b['variants'][v]['prompt'])
            batches = common_batches
            if weights.get('compute_matched', False):
                pools, _ = train_pools(data)
                pool_tokens = {b['id']: encode_prompt(tokenizer, b['variants'][0]['prompt'])
                               for values in pools.values() for b in values}
                rng = random.Random(seed + 20000)
                batches = []
                for batch in common_batches:
                    replacement = []
                    for source, view in batch:
                        target_length = len(token_cache[(source['id'], view)])
                        candidates = pools[source['task']]
                        distances = [abs(len(pool_tokens[b['id']])-target_length) for b in candidates]
                        closest = [b for b,d in zip(candidates,distances) if d == min(distances)]
                        chosen = rng.choice(closest)
                        replacement.append((chosen,0))
                    batches.append(replacement)
                for values in pools.values():
                    for b in values:
                        token_cache[(b['id'],0)] = pool_tokens[b['id']]
            reference_token_count = sum(len(token_cache[(b['id'],v)]) for batch in common_batches for b,v in batch)
            max_length = max(map(len, token_cache.values()))
            assert max_length <= protocol['max_tokens'], f'No silent truncation: {max_length}'
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=protocol['learning_rate'], weight_decay=0.0)
            history = []
            token_count = 0
            padded_count = 0
            model.train()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            for step, rows in enumerate(batches):
                optimizer.zero_grad(set_to_none=True)
                tokens = [token_cache[(b['id'], v)] for b,v in rows]
                batch = collate(tokens, tokenizer.pad_token_id)
                logits = last_logits(model, batch)
                loss, parts = objective(logits, rows, answer_ids, weights, protocol['bundles_per_group'])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['gradient_clip'])
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError('Nonfinite gradient')
                optimizer.step()
                token_count += sum(map(len, tokens))
                padded_count += batch['input_ids'].numel()
                record = {'arm': arm, 'seed': seed, 'step': step+1, 'loss': float(loss.detach()),
                          **{key: float(value.detach()) for key,value in parts.items()},
                          'elapsed_seconds': time.monotonic()-started}
                history.append(record)
                with (run / 'training.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record) + '\n')
                if step == 0 or (step+1) % 16 == 0:
                    print(f'TRAIN {arm} seed={seed} step={step+1}/{len(batches)} loss={record["loss"]:.4f} '
                          f'anchor={record["anchor"]:.4f} sec={record["elapsed_seconds"]:.1f}', flush=True)
            torch.cuda.synchronize()
            destination.mkdir(parents=True)
            model.save_pretrained(destination, safe_serialization=True)
            record = {'arm': arm, 'seed': seed, 'contract_sha256': contract_sha,
                      'schedule_sha256': schedule_sha, 'initial_adapter_sha256': initial_sha,
                      'adapter_sha256': sha256(destination / 'adapter_model.safetensors'),
                      'steps': len(batches), 'forward_examples': len(batches)*8*protocol['bundles_per_group'],
                      'hidden_label_access_guard': True,
                      'nonpadding_tokens': token_count, 'padded_tokens': padded_count,
                      'reference_nonpadding_tokens': reference_token_count,
                      'token_budget_relative_error': abs(token_count-reference_token_count)/reference_token_count,
                      'actual_batch_schedule_sha256': hashlib.sha256(json.dumps([[(b['id'],v) for b,v in batch] for batch in batches]).encode()).hexdigest(),
                      'parameters_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'seconds': time.monotonic()-started,
                      'peak_memory_mib': torch.cuda.max_memory_allocated()/1024**2,
                      'history': history, 'adapter_path': str(destination.relative_to(run))}
            save_json(record_file, record, immutable=True)
            assert record['token_budget_relative_error'] <= .01
            records.append(record)
            del model, optimizer, adapter, logits, batch, loss, parts
            torch.cuda.empty_cache()
    save_json(run / 'training_complete.json', {'contract_sha256': contract_sha, 'records': records}, immutable=True)
    print('All frozen final-step adapters complete; oracle/test evaluation can now run.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run_training(args.run_dir.resolve())
