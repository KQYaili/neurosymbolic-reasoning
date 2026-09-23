"""Increase the preregistered OOD audit before any model evaluation.

Preserves every original train/dev/ID-test row and exposed label. The first pilot
artifacts remain untouched. Adds new source-disjoint OOD bundles only.
"""
import copy
import json
from pathlib import Path
import random
from collections import Counter
import synthetic_data as g

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/local_scheme_validation/20260923_relational_v1'
OUT = ROOT / 'runs/local_scheme_validation/20260923_relational_v2'


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'data.json').exists():
        raise FileExistsError('Refuse to overwrite frozen extended data')
    data = json.loads((SOURCE / 'data.json').read_text(encoding='utf-8'))
    oracle = json.loads((SOURCE / 'oracle.json').read_text(encoding='utf-8'))
    original_config = copy.deepcopy(g.CONFIG)
    g.CONFIG['source_bundles_per_task'] = {'ood': 4096}
    nli, nt, _ = g.make_nli(random.Random(20260924))
    qa, qt = g.make_mcqa(random.Random(20260925))
    for task, candidates, truths, labels in [('nli', nli['ood'], nt, g.LABELS), ('mcqa', qa['ood'], qt, g.MC_LABELS)]:
        rows = data['splits']['ood'][task]
        used = {b['source_signature'] for tasks in data['splits'].values() for b in tasks[task]}
        counts = Counter(oracle['sources'][b['id']]['label'] for b in rows)
        targets = g.quotas(2048, labels)
        for candidate in candidates:
            truth = truths[candidate['id']]
            label = truth['label']
            if candidate['source_signature'] in used or counts[label] >= targets[label]:
                continue
            old_id = candidate['id']
            new_id = f'{task}-ood-{len(rows):04d}'
            candidate['id'] = new_id
            for variant in candidate['variants']:
                variant['id'] = variant['id'].replace(old_id, new_id)
            rows.append(candidate)
            oracle['sources'][new_id] = truth
            used.add(candidate['source_signature'])
            counts[label] += 1
        assert len(rows) == 2048 and dict(counts) == targets
        random.Random(20260926).shuffle(rows)
    g.CONFIG.clear()
    g.CONFIG.update(original_config)
    g.CONFIG['source_bundles_per_task']['ood'] = 2048
    g.CONFIG['version'] = 'controlled_relational_v2_larger_ood'
    data['config_sha256'] = oracle['config_sha256'] = g.digest(g.CONFIG)
    report = g.audit(data, oracle)
    # Exact semantic-ID/numeric-rank balancing of the extension was not forced;
    # record actual counts rather than selecting OOD for this secondary property.
    rank = report['statistics']['mcqa_correct_numeric_rank_counts']['ood']
    semantic = report['statistics']['mcqa_correct_semantic_id_counts']['ood']
    report['checks']['mcqa_numeric_rank_and_option_identity_balanced'] = max(rank.values())-min(rank.values()) <= 64 and max(semantic.values())-min(semantic.values()) <= 64
    report['checks']['original_train_dev_test_unchanged'] = all(
        data['splits'][s] == json.loads((SOURCE/'data.json').read_text(encoding='utf-8'))['splits'][s]
        for s in ('train', 'dev', 'test'))
    report['all_passed'] = all(report['checks'].values())
    assert report['all_passed'], report['checks']
    for name, value in [('data.json', data), ('oracle.json', oracle)]:
        (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    manifest = {'generation_config': g.CONFIG, 'base_data_sha256': g.file_hash(SOURCE/'data.json'),
                'generator_sha256': g.file_hash(g.__file__), 'extension_sha256': g.file_hash(__file__),
                'artifacts_sha256': {name:g.file_hash(OUT/name) for name in ['data.json','oracle.json']},
                'audit': report, 'rank_balance_scope': 'OOD numeric ranks/semantic IDs within 64 of each other; actual counts retained',
                'ood_scope': 'Both greater logical/arithmetic complexity and new surface wording; effects not disentangled'}
    (OUT/'data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Trainer gets only a public allow-list, with no signatures/ASTs/evaluator labels.
    public = {'splits': {'train': {task: [] for task in ('nli','mcqa')}}}
    for row in g.training_view(data):
        public['splits']['train'][row['task']].append(row)
    (OUT/'train_public.json').write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'all_passed':report['all_passed'],'statistics':report['statistics']},indent=2))


if __name__ == '__main__':
    main()
