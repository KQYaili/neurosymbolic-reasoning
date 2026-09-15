"""Build and audit a dependency-only notebook candidate; --apply opts in to writing it.

Selections use AST function names and assignments, never notebook cell positions.
The default run writes a candidate, audit, and immutable source backup only.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'runs/neurosymbolic_nli/20260915_order_v1'


def source_of(cell):
    source = cell.get('source', [])
    return source if isinstance(source, str) else ''.join(source)


def bounds(source, node):
    """AST columns are UTF-8 byte offsets, including in Chinese notebook cells."""
    lines = source.splitlines(keepends=True)
    start = sum(len(line) for line in lines[:node.lineno - 1])
    end = sum(len(line) for line in lines[:node.end_lineno - 1])
    start += len(lines[node.lineno - 1].encode('utf-8')[:node.col_offset].decode('utf-8'))
    end += len(lines[node.end_lineno - 1].encode('utf-8')[:node.end_col_offset].decode('utf-8'))
    return start, end


def replace_nodes(source, replacements):
    spans = [(bounds(source, node), replacement) for node, replacement in replacements]
    for (start, end), replacement in sorted(spans, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def is_name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def assigns(node, name):
    return isinstance(node, ast.Assign) and any(is_name(target, name) for target in node.targets)


def repair_cell(source, extract_source):
    tree = ast.parse(source)
    replacements, reasons = [], []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name == 'download_extract':
            if ast.dump(node, include_attributes=False) != ast.dump(
                    ast.parse(extract_source).body[0], include_attributes=False):
                replacements.append((node, extract_source.rstrip()))
                reasons.append('restore cached extraction, progress and tar data filter')
        elif node.name in ('train_ch13', 'train_batch_ch13', 'evaluate_accuracy_gpu'):
            for part in ast.walk(node):
                if (isinstance(part, ast.Call) and is_name(part.func, 'isinstance')
                        and len(part.args) == 2 and is_name(part.args[0], 'X')
                        and is_name(part.args[1], 'list')):
                    replacements.append((part.args[1], '(list, tuple)'))
                    reasons.append(f'{node.name}: accept list and tuple features')
                if (node.name == 'train_ch13' and isinstance(part, ast.Subscript)
                        and isinstance(part.value, ast.Attribute)
                        and is_name(part.value.value, 'features')
                        and part.value.attr == 'shape'
                        and isinstance(part.slice, ast.Constant) and part.slice.value == 0):
                    replacements.append((part, 'labels.shape[0]'))
                    reasons.append('train_ch13: count examples from labels for multi-input models')
        elif node.name == 'predict_snli':
            for part in ast.walk(node):
                if (isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute)
                        and is_name(part.func.value, 'd2l') and part.func.attr == 'try_gpu'):
                    replacements.append((part, 'next(net.parameters()).device'))
                    reasons.append('predict_snli: use model device without missing d2l dependency')
        elif node.name == 'load_data_imdb':
            if not any(assigns(part, 'data_dir') for part in node.body):
                # Replace the first load statement with a local cache lookup plus that statement.
                first_load = next(part for part in node.body if assigns(part, 'train_data'))
                local_cache = (
                    "data_dir = download_extract('aclImdb', 'aclImdb',\n"
                    "                                cache_dir=imdb_cache_dir)\n    "
                    + ast.get_source_segment(source, first_load))
                replacements.append((first_load, local_cache))
                reasons.append('load_data_imdb: avoid later SNLI overwriting global data_dir')

    # This is the IMDb setup cell, identified by the read_imdb definition.
    if any(isinstance(node, ast.FunctionDef) and node.name == 'read_imdb' for node in tree.body):
        has_cache = any(assigns(node, 'imdb_cache_dir') for node in tree.body)
        if not has_cache:
            data_assignment = next((node for node in tree.body if assigns(node, 'data_dir')), None)
            if data_assignment is None:
                raise RuntimeError('IMDb setup found but data_dir assignment was not found')
            cache_setup = (
                "# Keep many IMDb text files on the interpreter's native filesystem.\n"
                "imdb_cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'd2l')\n"
                "data_dir = download_extract('aclImdb', 'aclImdb', cache_dir=imdb_cache_dir)")
            replacements.append((data_assignment, cache_setup))
            reasons.append('initialize IMDb in native home cache instead of mounted Windows data')
            # Remove only the old directory-existence fallback corresponding to IMDb.
            for node in tree.body:
                if (isinstance(node, ast.If) and not node.orelse
                        and any(isinstance(part, ast.Call)
                                and is_name(part.func, 'download_extract')
                                and part.args and isinstance(part.args[0], ast.Constant)
                                and part.args[0].value == 'aclImdb' for part in ast.walk(node))):
                    replacements.append((node, ''))
    repaired = replace_nodes(source, replacements)
    compile(repaired, '<notebook-cell>', 'exec')
    return repaired, list(dict.fromkeys(reasons))


def audit_notebook(notebook):
    from pyflakes.checker import Checker

    parts, line_map, offset = [], [], 0
    for index, cell in enumerate(notebook['cells']):
        if cell.get('cell_type') != 'code':
            continue
        source = source_of(cell)
        compile(source, f'cell_{index}', 'exec')
        part = f'# notebook cell {index}\n' + source.rstrip('\n') + '\n\n'
        parts.append(part)
        for line in range(len(part.splitlines())):
            line_map.append({'cell_index': index, 'cell_id': cell.get('id'),
                             'cell_line': max(1, line)})
        offset += len(part.splitlines())
    checker = Checker(ast.parse(''.join(parts)), filename='2.dependencies.candidate.ipynb')
    undefined = []
    for message in checker.messages:
        if type(message).__name__ in ('UndefinedName', 'UndefinedLocal', 'UndefinedExport'):
            entry = dict(line_map[message.lineno - 1])
            entry.update(type=type(message).__name__, message=message.message % message.message_args)
            undefined.append(entry)
    return {'compiled_code_cells': len(parts), 'undefined_names': undefined,
            'other_pyflakes_messages': len(checker.messages) - len(undefined),
            'scope': 'all code cells compiled; pyflakes undefined-name audit of ordered concatenation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=ROOT / '2.ipynb')
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--apply', action='store_true', help='write candidate to the main notebook after audit')
    args = parser.parse_args()
    notebook_path = args.notebook.resolve()
    original_bytes = notebook_path.read_bytes()
    original_sha256 = hashlib.sha256(original_bytes).hexdigest()
    original = json.loads(original_bytes)
    candidate = copy.deepcopy(original)
    extract_path = ROOT / 'scratch/imdb_extract_fix_20260914_210356/download_extract.py'
    extract_source = extract_path.read_text(encoding='utf-8')
    changed = []
    for index, cell in enumerate(candidate['cells']):
        if cell.get('cell_type') != 'code':
            continue
        before = source_of(cell)
        after, reasons = repair_cell(before, extract_source)
        if after != before:
            cell['source'] = after if isinstance(cell['source'], str) else after.splitlines(keepends=True)
            changed.append({'cell_index': index, 'cell_id': cell.get('id'), 'changes': reasons})
    assert len(candidate['cells']) == len(original['cells'])
    for old, new in zip(original['cells'], candidate['cells']):
        assert {k: v for k, v in old.items() if k != 'source'} == {
            k: v for k, v in new.items() if k != 'source'}, 'Cell metadata or output unexpectedly changed'
    args.run_dir.mkdir(parents=True, exist_ok=True)
    backup = args.run_dir / f'2.before_dependencies.{original_sha256[:12]}.ipynb'
    if backup.exists():
        assert backup.read_bytes() == original_bytes, 'Backup hash collision or edited backup'
    else:
        backup.write_bytes(original_bytes)
    candidate_path = args.run_dir / '2.dependencies.candidate.ipynb'
    candidate_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    # Audit the saved candidate, not a separate in-memory source or the main notebook.
    audit = audit_notebook(json.loads(candidate_path.read_text(encoding='utf-8')))
    report = {'notebook': str(notebook_path), 'original_sha256': original_sha256,
              'backup': str(backup), 'candidate': str(candidate_path),
              'cell_count_before': len(original['cells']), 'cell_count_after': len(candidate['cells']),
              'changed_cells': changed, 'audit': audit, 'applied': False}
    if audit['undefined_names']:
        (args.run_dir / 'dependency_repair_report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        raise RuntimeError(f'Candidate still has undefined names: {audit["undefined_names"]}')
    if args.apply:
        if notebook_path.read_bytes() != original_bytes:
            raise RuntimeError('Main notebook changed while building candidate; rerun against latest version')
        # Preserve the current notebook until the fully written replacement is ready.
        temporary = notebook_path.with_name(notebook_path.name + '.dependencies.tmp')
        temporary.write_bytes(candidate_path.read_bytes())
        if notebook_path.read_bytes() != original_bytes:
            raise RuntimeError('Main notebook changed before replace; candidate retained and main untouched')
        temporary.replace(notebook_path)
        report['applied'] = True
    report_path = args.run_dir / 'dependency_repair_report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
