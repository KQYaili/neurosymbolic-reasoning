"""Sealed-label evaluation of frozen adapters; all inference is FP32.

Every generated view is scored. Probabilities are saved before metric calculation.
No generation parsing, checkpoint selection or test-driven training is performed.
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


def metric_arrays(probabilities, truth, option_ids=None):
    prediction = probabilities.argmax(-1)
    correct = prediction == truth
    if option_ids is None:
        violations = ((prediction[:, 0] == 1) != (prediction[:, 1] == 1)).astype(float)
    else:
        semantic = np.take_along_axis(option_ids, prediction[..., None], axis=-1)[..., 0]
        violations = (semantic[:, 1:] != semantic[:, :1]).mean(-1)
    return {'accuracy': correct.mean(-1), 'base_accuracy': correct[:, 0].astype(float),
            'atc': correct.all(-1).astype(float), 'rvr': violations}


def summarize(probabilities, truth, raw_ids, answer_ids, valid_mass, option_ids=None):
    arrays = metric_arrays(probabilities, truth, option_ids)
    pred = probabilities.argmax(-1)
    nc = probabilities.shape[-1]
    cm = np.bincount((truth*nc+pred).ravel(), minlength=nc*nc).reshape(nc,nc)
    f1 = 2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)
    valid_raw = np.isin(raw_ids, answer_ids)
    correct = pred == truth
    base_correct = correct[:,0]
    result = {k: float(v.mean()) for k,v in arrays.items()}
    result.update({'macro_f1_all_views': float(f1.mean()), 'class_f1': f1.tolist(),
                   'confusion_matrix': cm.tolist(), 'bundles': len(truth), 'views': truth.shape[1],
                   'raw_first_token_invalid_rate': float((~valid_raw).mean()),
                   'raw_first_token_accuracy': float((raw_ids == np.asarray(answer_ids)[truth]).mean()),
                   'valid_label_probability_mass': float(valid_mass.mean()),
                   'mean_entropy': float(-(probabilities*np.log(np.maximum(probabilities,1e-12))).sum(-1).mean()),
                   'prediction_histogram': np.bincount(pred.ravel(), minlength=nc).tolist(),
                   'conditional_transformed_accuracy_given_correct_original': float(correct[base_correct,1:].mean()) if base_correct.any() else None,
                   'all_transforms_correct_given_correct_original': float(correct[base_correct].all(-1).mean()) if base_correct.any() else None})
    return result


@torch.inference_mode()
def evaluate_group(model, tokenizer, answer_ids, bundles, truths, token_cache, batch_size):
    task = bundles[0]['task']
    nclass = 3 if task == 'nli' else 4
    views = len(bundles[0]['variants'])
    labels = answer_ids[:nclass]
    flat = [variant for bundle in bundles for variant in bundle['variants']]
    ps, raw, masses, class_logits = [], [], [], []
    for start in range(0, len(flat), batch_size):
        variants = flat[start:start+batch_size]
        inputs = tr.collate([token_cache[v['id']] for v in variants], tokenizer.pad_token_id)
        logits = tr.last_logits(model, inputs)
        selected = logits[:,labels]
        class_logits.append(selected.cpu().numpy())
        ps.append(selected.softmax(-1).cpu().numpy())
        masses.append(torch.exp(torch.logsumexp(selected,-1)-torch.logsumexp(logits,-1)).cpu().numpy())
        raw.append(logits.argmax(-1).cpu().numpy())
    probabilities = np.concatenate(ps).reshape(len(bundles), views, nclass)
    raw_ids = np.concatenate(raw).reshape(len(bundles), views)
    valid_mass = np.concatenate(masses).reshape(len(bundles), views)
    saved_logits = np.concatenate(class_logits).reshape(len(bundles), views, nclass)
    sorted_logits = np.sort(saved_logits, axis=-1)
    gold = np.array([[tr.LABELS.index(y) for y in truths[b['id']]['variant_labels']] for b in bundles])
    options = np.asarray([[v['option_ids'] for v in b['variants']] for b in bundles]) if task == 'mcqa' else None
    arrays = {'probabilities': probabilities, 'raw_token_ids': raw_ids, 'valid_label_mass': valid_mass,
              'label_logits': saved_logits, 'top2_logit_margin': sorted_logits[..., -1]-sorted_logits[..., -2],
              'labels': gold, 'bundle_ids': np.array([b['id'] for b in bundles]),
              'answer_ids': labels.cpu().numpy()}
    if options is not None:
        arrays['option_ids'] = options
    result = summarize(probabilities,gold,raw_ids,arrays['answer_ids'],valid_mass,options)
    result['near_tie_fraction_margin_below_0_0004'] = float((arrays['top2_logit_margin'] < .0004).mean())
    if task == 'nli':
        # Evaluation-only oracle labels define the sound reverse allowed set.
        allowed = np.where(gold[:, 0] == 1, probabilities[:, 1, 1], 1-probabilities[:, 1, 1])
        result['allowed_label_probability_mass'] = float(allowed.mean())
        result['allowed_mass_scope'] = 'Conditional class mass of the reverse allowed set derived from the true original label'
    else:
        result['allowed_label_probability_mass'] = float(np.take_along_axis(probabilities, gold[..., None], -1).mean())
        result['allowed_mass_scope'] = 'Conditional class mass of the true transported singleton label, all views'
    return result, arrays


def compare(run, split, task, a, b, seeds, repetitions, bootstrap_seed):
    differences = {k:[] for k in ('accuracy','atc','rvr')}
    for seed in seeds:
        with np.load(run/'predictions'/f'{a}_seed{seed}_{split}_{task}.npz') as x, np.load(run/'predictions'/f'{b}_seed{seed}_{split}_{task}.npz') as y:
            assert np.array_equal(x['bundle_ids'],y['bundle_ids']) and np.array_equal(x['labels'],y['labels'])
            xm = metric_arrays(x['probabilities'],x['labels'],x.get('option_ids'))
            ym = metric_arrays(y['probabilities'],y['labels'],y.get('option_ids'))
            for k in differences:
                differences[k].append(xm[k]-ym[k])
    report = {}
    for key, value in differences.items():
        delta = np.asarray(value)
        per_bundle = delta.mean(0)
        rng = np.random.default_rng(bootstrap_seed)
        boot = np.array([per_bundle[rng.integers(0,len(per_bundle),len(per_bundle))].mean() for _ in range(repetitions)])
        report[key] = {'mean':float(delta.mean()), 'by_seed':delta.mean(1).tolist(),
                       'paired_bundle_95ci':np.quantile(boot,[.025,.975]).tolist()}
    return report


def main(run, batch_size):
    protocol = json.loads((run/'protocol.json').read_text())
    complete = json.loads((run/'training_complete.json').read_text())
    impl = json.loads((run/'implementation.json').read_text())
    assert complete['contract_sha256'] == tr.sha256(run/'implementation.json')
    assert impl['protocol_sha256'] == tr.sha256(run/'protocol.json')
    assert impl['data_sha256'] == tr.sha256(run/'train_public.json')
    assert impl['data_manifest_sha256'] == tr.sha256(run/'data_manifest.json')
    for name, checksum in impl['base_files'].items():
        assert tr.sha256(Path(protocol['snapshot'])/name) == checksum
    assert len(complete['records']) == len(protocol['arms'])*len(protocol['seeds'])
    assert {(r['arm'],r['seed']) for r in complete['records']} == {(a,s) for a in protocol['arms'] for s in protocol['seeds']}
    for name, checksum in impl['sources'].items():
        assert tr.sha256(Path(__file__).with_name(name)) == checksum
    for record in complete['records']:
        assert record['steps'] == protocol['steps']
        assert record['contract_sha256'] == complete['contract_sha256']
        assert tr.sha256(run/record['adapter_path']/'adapter_model.safetensors') == record['adapter_sha256']
        config = json.loads((run/record['adapter_path']/'adapter_config.json').read_text())
        for key, value in protocol['lora'].items():
            assert (sorted(config[key]) == sorted(value)) if isinstance(value,list) else config[key] == value
        assert config['task_type'] == 'CAUSAL_LM' and config['peft_type'] == 'LORA'
    # The seal is written before reading even one oracle label or evaluation prompt.
    seal = {'protocol_sha256':tr.sha256(run/'protocol.json'), 'contract_sha256':complete['contract_sha256'],
            'training_complete_sha256':tr.sha256(run/'training_complete.json'),
            'adapters':{r['adapter_path']:r['adapter_sha256'] for r in complete['records']},
            'adapter_configs':{r['adapter_path']:tr.sha256(run/r['adapter_path']/'adapter_config.json') for r in complete['records']},
            'evaluator_sha256':tr.sha256(__file__), 'batch_size':batch_size,'dtype':'float32_no_tf32'}
    tr.save_json(run/'evaluation_seal.json',seal,immutable=True)
    data = json.loads((run/'data.json').read_text(encoding='utf-8'))
    oracle = json.loads((run/'oracle.json').read_text(encoding='utf-8'))['sources']
    manifest = json.loads((run/'data_manifest.json').read_text())
    assert all(tr.sha256(run/name)==checksum for name,checksum in manifest['artifacts_sha256'].items())
    tr.setup(20260923)
    (run/'predictions').mkdir(exist_ok=True)
    rows = [{'arm':'base','seed':0}] + [{'arm':r['arm'],'seed':r['seed'],'adapter_path':r['adapter_path']} for r in complete['records']]
    results = []
    token_cache = None
    for row in rows:
        done_file = run/'predictions'/f'{row["arm"]}_seed{row["seed"]}_metrics.json'
        if done_file.exists():
            cached = json.loads(done_file.read_text())
            assert cached['evaluation_seal_sha256'] == tr.sha256(run/'evaluation_seal.json')
            assert all(tr.sha256(run/'predictions'/name)==checksum for name,checksum in cached['prediction_sha256'].items())
            results.append(cached)
            print('RESUME EVAL',row['arm'],row['seed'],flush=True)
            continue
        started = time.monotonic()
        model, tokenizer, answer_ids = tr.base_and_tokenizer(protocol, with_adapter=False)
        if row['arm'] != 'base':
            model = PeftModel.from_pretrained(model,run/row['adapter_path'],is_trainable=False)
        model.float().eval()
        if token_cache is None:
            token_cache = {v['id']:tr.encode_prompt(tokenizer,v['prompt'])
                           for split in ('dev','test','ood') for bundles in data['splits'][split].values()
                           for b in bundles for v in b['variants']}
            assert max(map(len,token_cache.values())) <= protocol['max_tokens']
        record = {**row,'evaluation_seal_sha256':tr.sha256(run/'evaluation_seal.json'),'prediction_sha256':{}}
        for split in ('dev','test','ood'):
            record[split] = {}
            for task in ('nli','mcqa'):
                metrics, arrays = evaluate_group(model,tokenizer,answer_ids,data['splits'][split][task],oracle,token_cache,batch_size)
                filename = f'{row["arm"]}_seed{row["seed"]}_{split}_{task}.npz'
                np.savez_compressed(run/'predictions'/filename,**arrays)
                record['prediction_sha256'][filename] = tr.sha256(run/'predictions'/filename)
                record[split][task] = metrics
                print(f'EVAL {row["arm"]} seed={row["seed"]} {split}/{task} '
                      f'accuracy={metrics["accuracy"]:.4f} ATC={metrics["atc"]:.4f} RVR={metrics["rvr"]:.4f}',flush=True)
        record['seconds'] = time.monotonic()-started
        tr.save_json(done_file,record,immutable=True)
        results.append(record)
        del model
        torch.cuda.empty_cache()
    aggregate = {}
    for split in ('dev','test','ood'):
        aggregate[split] = {}
        for task in ('nli','mcqa'):
            aggregate[split][task] = {}
            for arm in ['base']+list(protocol['arms']):
                selected = [r[split][task] for r in results if r['arm']==arm]
                aggregate[split][task][arm] = {key:{'mean':float(np.mean([r[key] for r in selected])),
                                                  'std':float(np.std([r[key] for r in selected],ddof=1)) if len(selected)>1 else None}
                                                      for key in ('accuracy','base_accuracy','atc','rvr','macro_f1_all_views',
                                                                  'raw_first_token_invalid_rate','raw_first_token_accuracy','mean_entropy')}
    comparisons = {split:{task:{f'{a}_minus_{b}':compare(run,split,task,a,b,protocol['seeds'],protocol['bootstrap_repetitions'],protocol['bootstrap_seed'])
                                    for a,b in protocol['comparisons']} for task in ('nli','mcqa')} for split in ('test','ood')}
    balanced = {split: {arm: {metric: float(np.mean([aggregate[split][task][arm][metric]['mean'] for task in ('nli','mcqa')]))
                                  for metric in ('accuracy','atc','rvr')}
                            for arm in ['base']+list(protocol['arms'])}
                for split in ('dev','test','ood')}
    gates = {}
    for task in ('nli','mcqa'):
        gates[task] = {}
        for control in ('sft_compute','exact_aug'):
            comp = comparisons['ood'][task][f'relational_minus_{control}']
            gate = {'atc_positive':comp['atc']['paired_bundle_95ci'][0]>0,
                    'accuracy_noninferior':comp['accuracy']['paired_bundle_95ci'][0]>-.01,
                    'rvr_lower':comp['rvr']['paired_bundle_95ci'][1]<0}
            gate['passed'] = all(gate.values())
            gates[task][control] = gate
    summary = {'results':results,'aggregate':aggregate,'task_balanced_mean':balanced,'comparisons':comparisons,'statistical_gates':gates,
               'effectiveness_gates_passed':all(g['passed'] for task in gates.values() for g in task.values()),
               'integrity_independent_verification_pending':True,'ci_scope':protocol['ci_scope'],
               'scope':protocol['objective'],'evaluation_seal_sha256':tr.sha256(run/'evaluation_seal.json')}
    tr.save_json(run/'summary.json',summary)
    print(json.dumps({'statistical_gates':gates,'effectiveness_gates_passed':summary['effectiveness_gates_passed']},indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,default=tr.DEFAULT_RUN)
    parser.add_argument('--batch-size',type=int,default=16)
    args = parser.parse_args()
    main(args.run_dir.resolve(),args.batch_size)
