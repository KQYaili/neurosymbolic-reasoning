"""Independent data, training-artifact, metric and fixed-dev replay verification.

Does not import the evaluator or generator; never selects a model or threshold.
Run after evaluate.py. The CPU audit recomputes truth and every reported metric.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'runs/local_scheme_validation/20260923_relational_v2'
LABELS = 'ABCD'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()


def boolean(node, world):
    op, *children = node
    if op == 'atom': return bool(world & (1 << children[0]))
    if op == 'not': return not boolean(children[0], world)
    values = [boolean(child, world) for child in children]
    if op == 'and': return all(values)
    if op == 'or': return any(values)
    raise ValueError(op)


def relation(p, h):
    return 'A' if p.issubset(h) else 'B' if p.isdisjoint(h) else 'C'


def data_audit(run):
    data, oracle, public = read(run/'data.json'), read(run/'oracle.json')['sources'], read(run/'train_public.json')
    signatures, prompts, counts = {}, {}, {}
    seen_ids = set()
    row_keys = {'id','task','supervised','gold_label','variants'}
    view_keys = {'id','kind','prompt','answer_labels','supervised_label','allowed_labels','option_ids','base_to_variant'}
    for split, tasks in data['splits'].items():
        signatures[split], prompts[split] = set(), set()
        for task, rows in tasks.items():
            assert len(rows) == {'train':1024,'dev':128,'test':256,'ood':2048}[split]
            exposed = 0
            for row in rows:
                assert row['id'] not in seen_ids
                seen_ids.add(row['id'])
                truth = oracle[row['id']]
                if task == 'nli':
                    sets = [{w for w in range(2**truth['atoms']) if boolean(truth[key],w)}
                            for key in ('premise_ast','hypothesis_ast')]
                    assert all(sets)
                    assert [sum(2**w for w in s) for s in sets] == [truth['premise_mask'],truth['hypothesis_mask']]
                    expected = [relation(*sets),relation(*sets[::-1])]
                    signature = (task,truth['atoms'],*sorted(tuple(sorted(s)) for s in sets))
                else:
                    op,a,b,*rest = truth['expression']
                    if op == 'add': answer = a+b
                    elif op == 'subtract': answer = a-b
                    elif op == 'multiply': answer = a*b
                    elif op == 'add_then_multiply': answer = (a+b)*rest[0]
                    elif op == 'multiply_then_add': answer = a*b+rest[0]
                    elif op == 'subtract_then_multiply': answer = (a-b)*rest[0]
                    else: raise ValueError(op)
                    assert answer == truth['answer']
                    if op in ('add','multiply','add_then_multiply','multiply_then_add'): a,b = sorted((a,b))
                    signature = (task,op,a,b,*rest)
                    expected = []
                    semantic_values = None
                    for v in row['variants']:
                        assert sorted(v['option_ids']) == list(range(4)) and len(set(v['option_values'])) == 4
                        values = dict(zip(v['option_ids'],v['option_values']))
                        if semantic_values is None: semantic_values = values
                        assert values == semantic_values
                        expected.append(LABELS[v['option_values'].index(answer)])
                        assert v['base_to_variant'] == [v['option_ids'].index(i) for i in row['variants'][0]['option_ids']]
                    assert len(set(expected)) == 4
                assert signature not in signatures[split]
                signatures[split].add(signature)
                assert expected == truth['variant_labels'] and expected[0] == truth['label']
                prompts[split].update(v['prompt'] for v in row['variants'])
                if row['supervised']:
                    exposed += 1
                    assert split == 'train' and row['gold_label'] == expected[0]
                    for i,v in enumerate(row['variants']):
                        if task == 'mcqa' or i == 0:
                            assert v['supervised_label'] == expected[i]
                        else:
                            assert v['supervised_label'] is None
                            assert v['allowed_labels'] == (['B'] if expected[0] == 'B' else ['A','C'])
                            assert expected[i] in v['allowed_labels']
                else:
                    assert row['gold_label'] is None
                    assert all(v['supervised_label'] is None and v['allowed_labels'] is None for v in row['variants'])
            counts[f'{split}/{task}'] = {'sources':len(rows),'exposed_labels':exposed}
            assert exposed == (102 if split == 'train' else 0)
            if split == 'train':
                expected_public = [{k:([{vk:vv for vk,vv in v.items() if vk in view_keys} for v in r['variants']]
                                       if k == 'variants' else value)
                                    for k,value in r.items() if k in row_keys} for r in rows]
                assert expected_public == public['splits']['train'][task]
    assert set(public) == {'splits'} and set(public['splits']) == {'train'}
    assert seen_ids == set(oracle)
    for a,b in itertools.combinations(signatures,2):
        assert not signatures[a].intersection(signatures[b])
        assert not prompts[a].intersection(prompts[b])
    return data, oracle, counts


def metrics(z):
    p, y = z['probabilities'], z['labels']
    pred = np.argmax(p,axis=2)
    hits = pred == y
    if 'option_ids' in z:
        selected = np.array([[z['option_ids'][i,j,k] for j,k in enumerate(row)] for i,row in enumerate(pred)])
        violation = np.array([sum(x != row[0] for x in row[1:])/(len(row)-1) for row in selected])
    else:
        violation = np.array([int((a == 1) != (b == 1)) for a,b in pred])
    arrays = {'accuracy':hits.sum(1)/hits.shape[1], 'base_accuracy':hits[:,0].astype(float),
              'atc':hits.min(1).astype(float), 'rvr':violation}
    nc = p.shape[-1]
    cm = np.zeros((nc,nc),dtype=int)
    for t,q in zip(y.flat,pred.flat): cm[t,q] += 1
    f1 = [2*cm[k,k]/max(cm[k,:].sum()+cm[:,k].sum(),1) for k in range(nc)]
    out = {k:float(v.mean()) for k,v in arrays.items()}
    out.update(confusion_matrix=cm.tolist(), class_f1=f1, macro_f1_all_views=float(np.mean(f1)),
               prediction_histogram=[int((pred==i).sum()) for i in range(nc)],
               raw_first_token_invalid_rate=float(np.mean(~np.isin(z['raw_token_ids'],z['answer_ids']))),
               raw_first_token_accuracy=float(np.mean(z['raw_token_ids']==z['answer_ids'][y])),
               valid_label_probability_mass=float(z['valid_label_mass'].mean()),
               mean_entropy=float(np.mean(np.sum(-p*np.log(np.clip(p,1e-12,None)),axis=-1))))
    eligible = hits[:,0]
    out['conditional_transformed_accuracy_given_correct_original'] = float(hits[eligible,1:].mean()) if eligible.any() else None
    out['all_transforms_correct_given_correct_original'] = float(hits[eligible].min(1).mean()) if eligible.any() else None
    if nc == 3:
        out['allowed_label_probability_mass'] = float(np.mean([p[i,1,1] if row[0]==1 else p[i,1,0]+p[i,1,2] for i,row in enumerate(y)]))
    else:
        out['allowed_label_probability_mass'] = float(np.mean([p[i,j,k] for i,row in enumerate(y) for j,k in enumerate(row)]))
    return out,arrays


def main(run, replay=True):
    checks = {}
    protocol, impl, complete, seal, summary = [read(run/name) for name in
        ('protocol.json','implementation.json','training_complete.json','evaluation_seal.json','summary.json')]
    assert impl['protocol_sha256'] == seal['protocol_sha256'] == digest(run/'protocol.json')
    assert impl['data_sha256'] == digest(run/'train_public.json')
    assert impl['data_manifest_sha256'] == digest(run/'data_manifest.json')
    assert complete['contract_sha256'] == seal['contract_sha256'] == digest(run/'implementation.json')
    assert seal['training_complete_sha256'] == digest(run/'training_complete.json')
    assert seal['evaluator_sha256'] == digest(Path(__file__).with_name('evaluate.py'))
    assert summary['evaluation_seal_sha256'] == digest(run/'evaluation_seal.json')
    for filename,h in impl['sources'].items():
        assert digest(Path(__file__).with_name(filename)) == digest(run/'sources'/filename) == h
    for filename,h in impl['base_files'].items(): assert digest(Path(protocol['snapshot'])/filename) == h
    for filename,h in read(run/'data_manifest.json')['artifacts_sha256'].items(): assert digest(run/filename) == h
    checks['frozen_source_protocol_model_and_data_hashes'] = True
    data,oracle,counts = data_audit(run)
    checks['independent_truth_split_and_label_access_audit'] = True
    from safetensors.numpy import load_file
    records = complete['records']
    assert len(records) == 15 and {(r['arm'],r['seed']) for r in records} == set(itertools.product(protocol['arms'],protocol['seeds']))
    log = [json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
    assert len(log) == 15*protocol['steps']
    for r in records:
        path = run/r['adapter_path']
        assert digest(path/'adapter_model.safetensors') == r['adapter_sha256'] == seal['adapters'][r['adapter_path']]
        assert digest(path/'adapter_config.json') == seal['adapter_configs'][r['adapter_path']]
        tensors = load_file(str(path/'adapter_model.safetensors'))
        assert all(np.isfinite(v).all() for v in tensors.values())
        assert sum(v.size for v in tensors.values()) == r['parameters_trainable'] == 540672
        assert r['steps'] == 256 and r['forward_examples'] == 4096 and r['hidden_label_access_guard']
        assert r['token_budget_relative_error'] <= .01
        assert np.isclose(r['token_budget_relative_error'],abs(r['nonpadding_tokens']/r['reference_nonpadding_tokens']-1))
        history = [x for x in log if x['arm']==r['arm'] and x['seed']==r['seed']]
        assert history == r['history'] and [h['step'] for h in history] == list(range(1,257))
        assert all(np.isfinite(h['loss']) for h in history)
    for seed in protocol['seeds']:
        paired = [r for r in records if r['seed']==seed]
        for key in ('initial_adapter_sha256','schedule_sha256','reference_nonpadding_tokens'):
            assert len({r[key] for r in paired}) == 1
        normal = [r for r in paired if r['arm'] != 'sft_compute']
        for key in ('actual_batch_schedule_sha256','nonpadding_tokens','padded_tokens'):
            assert len({r[key] for r in normal}) == 1
    checks['all_15_final_adapters_and_paired_training_records'] = True
    cache = {}
    for rec in summary['results']:
        cached = read(run/'predictions'/f'{rec["arm"]}_seed{rec["seed"]}_metrics.json')
        assert rec == cached
        for split in ('dev','test','ood'):
            for task in ('nli','mcqa'):
                name = f'{rec["arm"]}_seed{rec["seed"]}_{split}_{task}.npz'
                assert digest(run/'predictions'/name) == rec['prediction_sha256'][name]
                with np.load(run/'predictions'/name) as f: z = dict(f)
                rows = data['splits'][split][task]
                assert z['bundle_ids'].tolist() == [b['id'] for b in rows]
                assert z['labels'].tolist() == [[LABELS.index(y) for y in oracle[b['id']]['variant_labels']] for b in rows]
                assert np.isfinite(z['probabilities']).all() and np.allclose(z['probabilities'].sum(-1),1,atol=1e-6)
                logits = z['label_logits'].astype(np.float64)
                ex = np.exp(logits-logits.max(-1,keepdims=True)); ex /= ex.sum(-1,keepdims=True)
                assert np.allclose(ex,z['probabilities'],atol=2e-7)
                sorted_logits = np.sort(z['label_logits'],axis=-1)
                assert np.array_equal(sorted_logits[...,-1]-sorted_logits[...,-2],z['top2_logit_margin'])
                m,arr = metrics(z)
                for key,value in m.items():
                    assert rec[split][task][key] is None if value is None else np.allclose(value,rec[split][task][key],atol=2e-7), (name,key)
                cache[(rec['arm'],rec['seed'],split,task)] = arr
    checks['all_96_prediction_files_and_metrics_independently_recomputed'] = True
    for split,tasks in summary['comparisons'].items():
        for task,pairs in tasks.items():
            for pair,metrics_report in pairs.items():
                a,b = pair.split('_minus_')
                for metric,result in metrics_report.items():
                    differences = np.array([cache[a,s,split,task][metric]-cache[b,s,split,task][metric] for s in protocol['seeds']])
                    assert np.allclose(differences.mean(1),result['by_seed'])
                    assert np.isclose(differences.mean(),result['mean'])
                    d = differences.mean(0)
                    rng = np.random.default_rng(protocol['bootstrap_seed'])
                    estimates = []
                    for _ in range(protocol['bootstrap_repetitions']):
                        index = rng.integers(len(d),size=len(d))
                        weights = np.bincount(index,minlength=len(d))
                        estimates.append(float(np.dot(weights,d)/len(d)))
                    assert np.allclose(np.percentile(estimates,[2.5,97.5]),result['paired_bundle_95ci'])
    for split,arms in summary['task_balanced_mean'].items():
        for arm,reported in arms.items():
            seeds = [0] if arm == 'base' else protocol['seeds']
            for metric,value in reported.items():
                assert np.isclose(value,np.mean([cache[arm,s,split,t][metric].mean() for t in ('nli','mcqa') for s in seeds]))
    for task,controls in summary['statistical_gates'].items():
        for control,gate in controls.items():
            c = summary['comparisons']['ood'][task][f'relational_minus_{control}']
            expect = {'atc_positive': c['atc']['paired_bundle_95ci'][0]>0,
                      'accuracy_noninferior': c['accuracy']['paired_bundle_95ci'][0]>-.01,
                      'rvr_lower':c['rvr']['paired_bundle_95ci'][1]<0}
            expect['passed'] = all(expect.values())
            assert expect == gate
    assert summary['effectiveness_gates_passed'] == all(g['passed'] for gs in summary['statistical_gates'].values() for g in gs.values())
    checks['paired_ci_balanced_means_and_prespecified_gates'] = True
    mechanism = read(run/'mechanism_verification.json')
    assert mechanism['all_passed'] and all(mechanism['checks'].values())
    assert mechanism['script_sha256'] == digest(Path(__file__).with_name('check_mechanism.py'))
    checks['independent_logic_and_numerical_mechanism_checks'] = True
    replay_report = replay_dev(run,protocol,data) if replay else None
    checks['fixed_dev_checkpoint_replay'] = replay_report is not None and replay_report['passed']
    result = {'all_passed':all(checks.values()),'checks':checks,'dataset_counts':counts,'dev_replay':replay_report,
              'summary_sha256':digest(run/'summary.json'),'script_sha256':digest(Path(__file__)),
              'effectiveness_gates_passed':summary['effectiveness_gates_passed'],
              'scope':'Integrity pass is separate from statistical effectiveness; no human real-world causal validation.'}
    (run/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def replay_dev(run,protocol,data):
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(protocol['snapshot'],local_files_only=True)
    details = []
    for arm in ('sft_compute','relational'):
        model = AutoModelForCausalLM.from_pretrained(protocol['snapshot'],local_files_only=True,dtype=torch.float32,attn_implementation='sdpa').cuda()
        model = PeftModel.from_pretrained(model,run/'adapters'/f'{arm}_seed17').float().eval()
        assert {p.dtype for p in model.parameters()} == {torch.float32}
        for task in ('nli','mcqa'):
            # First four frozen dev IDs; chosen by position before any outcome.
            rows = data['splits']['dev'][task][:4]
            tokens = [tokenizer.apply_chat_template([{'role':'user','content':v['prompt']}],add_generation_prompt=True) for b in rows for v in b['variants']]
            width = max(map(len,tokens))
            ids = torch.tensor([[tokenizer.eos_token_id]*(width-len(x))+x for x in tokens],device='cuda')
            mask = torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in tokens],device='cuda')
            positions = (mask.cumsum(-1)-1).clamp_min(0)
            answers = [tokenizer.encode(x,add_special_tokens=False)[0] for x in (LABELS[:3] if task=='nli' else LABELS)]
            with torch.inference_mode():
                logits = model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False,logits_to_keep=1).logits[:,-1,answers]
            with np.load(run/'predictions'/f'{arm}_seed17_dev_{task}.npz') as z:
                observed = logits.cpu().numpy().reshape(z['label_logits'][:4].shape)
                delta = float(np.max(np.abs(observed-z['label_logits'][:4])))
                assert np.allclose(observed,z['label_logits'][:4],atol=.0002,rtol=1e-5)
                assert np.array_equal(observed.argmax(-1),z['probabilities'][:4].argmax(-1))
            details.append({'arm':arm,'task':task,'dev_ids':[r['id'] for r in rows],'max_logit_delta':delta})
        del model
        torch.cuda.empty_cache()
    return {'passed':True,'cases':details,'dtype':'float32_no_tf32','no_test_based_selection':True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,default=DEFAULT)
    parser.add_argument('--skip-replay',action='store_true')
    args = parser.parse_args()
    main(args.run_dir.resolve(),not args.skip_replay)
