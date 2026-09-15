"""Execute the added notebook section and current helper API in a fresh kernel."""
import ast
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/neurosymbolic_nli/20260915_order_v1'


def main():
    notebook = nbformat.read(ROOT / '2.neurosymbolic.ipynb', as_version=4)
    helper_names = {'try_gpu', 'try_all_gpus', 'Timer', 'Accumulator', 'Animator',
                    'use_svg_display', 'set_axes', 'tokenize', 'truncate_pad', 'load_array',
                    'set_figsize', 'accuracy', 'evaluate_accuracy_gpu', 'train_batch_ch13',
                    'train_ch13', 'mlp', 'Attend', 'Compare', 'Aggregate', 'DecomposableAttention'}
    helpers = []
    for cell in notebook.cells:
        if cell.cell_type != 'code' or cell.id.startswith('neurosymbolic-nli-v1-'):
            continue
        for node in ast.parse(cell.source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in helper_names:
                helpers.append(ast.get_source_segment(cell.source, node))
    bootstrap = '''import math, os, random, sys, time, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython import display
torch.set_num_threads(4)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
'''
    helper_test = '''
assert tokenize(['a  b', 'c\\td']) == [['a', 'b'], ['c', 'd']]
assert tokenize(['ab'], token='char') == [['a', 'b']]
assert truncate_pad([1, 2, 3], 2, 0) == [1, 2]
assert truncate_pad([1], 3, 0) == [1, 0, 0]
from experiments.neurosymbolic_nli.data import load_data
_check_run = Path('runs/neurosymbolic_nli/20260915_order_v1')
_check_data = load_data(_check_run)
_check_split = _check_data['splits']['train']
class _CheckSNLI(torch.utils.data.Dataset):
    def __len__(self): return 8
    def __getitem__(self, i):
        return (_check_split['premises'][i, :12], _check_split['hypotheses'][i, :12]), _check_split['labels'][i]
_check_loader = torch.utils.data.DataLoader(_CheckSNLI(), batch_size=4)
_check_net = DecomposableAttention(_check_data['vocab'], 100, 32, num_inputs_agg=64)
_check_trainer = torch.optim.Adam(_check_net.parameters(), lr=.001)
_check_before = _check_net.aggregate.linear.weight.detach().clone()
train_ch13(_check_net, _check_loader, _check_loader, nn.CrossEntropyLoss(reduction='none'),
           _check_trainer, 1, [torch.device('cpu')])
assert not torch.equal(_check_before, _check_net.aggregate.linear.weight)
assert all(torch.isfinite(p).all() for p in _check_net.parameters())
_check_acc = evaluate_accuracy_gpu(_check_net, _check_loader, torch.device('cpu'))
assert 0 <= _check_acc <= 1
(_check_run / 'dependency_smoke.json').write_text(json.dumps({
    'passed': True, 'tokenize_word_char': True, 'truncate_pad': True,
    'actual_snli_pairs': 8, 'multi_input_training_batches': 2,
    'weights_updated': True, 'finite_model': True, 'accuracy': _check_acc,
    'scope': 'current notebook helper definitions and original DecomposableAttention; not a quality estimate'
}, indent=2) + '\\n', encoding='utf-8')
print('Dependency smoke passed: real SNLI sentence pairs, list inputs, training and evaluation.')
'''
    toy = next(cell for cell in notebook.cells if cell.cell_type == 'code' and 'def run_order_embedding_toy(' in cell.source)
    execution = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('Fresh-kernel validation of dependency helpers and the appended neuro-symbolic section.'),
        nbformat.v4.new_code_cell(bootstrap, id='validation-bootstrap'),
        nbformat.v4.new_code_cell('\n\n'.join(helpers), id='validation-helpers'),
        nbformat.v4.new_code_cell(helper_test, id='validation-helper-test'),
        nbformat.v4.new_code_cell(toy.source, id=toy.id),
    ] + [cell for cell in notebook.cells if cell.id.startswith('neurosymbolic-nli-v1-')])
    execution.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    client = NotebookClient(execution, timeout=180, kernel_name='python3',
                            resources={'metadata': {'path': str(ROOT)}})
    try:
        client.execute()
    finally:
        nbformat.write(execution, RUN / 'continuation.executed.ipynb')
    errors = [out for cell in execution.cells if cell.cell_type == 'code'
              for out in cell.get('outputs', []) if out.output_type == 'error']
    assert not errors, errors
    report = {'passed': True, 'fresh_kernel': True,
              'executed_code_cells': sum(cell.cell_type == 'code' for cell in execution.cells),
              'appended_code_cells': sum(cell.cell_type == 'code' and cell.id.startswith('neurosymbolic-nli-v1-')
                                         for cell in execution.cells),
              'user_kernel_restarted': False,
              'executed_notebook': str(RUN / 'continuation.executed.ipynb')}
    (RUN / 'notebook_execution.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
