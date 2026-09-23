"""Build a Chinese review, figures and executed notebook from verified artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/local_scheme_validation/20260923_relational_v2'


def load(name): return json.loads((RUN/name).read_text(encoding='utf-8'))
def pct(x): return f'{100*x:.2f}'
def ci(result): return f"{100*result['mean']:+.2f} [{100*result['paired_bundle_95ci'][0]:+.2f}, {100*result['paired_bundle_95ci'][1]:+.2f}]"


def main():
    s, verification, protocol, training = [load(n) for n in ('summary.json','verification.json','protocol.json','training_complete.json')]
    assert verification['all_passed']
    assert verification['summary_sha256'] == hashlib.sha256((RUN/'summary.json').read_bytes()).hexdigest()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms = list(protocol['arms'])
    fig, axes = plt.subplots(2,3,figsize=(14,7.5),layout='constrained')
    for i,task in enumerate(('nli','mcqa')):
        for j,metric in enumerate(('accuracy','atc','rvr')):
            values = [s['aggregate']['ood'][task][a][metric] for a in arms]
            axes[i,j].bar(np.arange(len(arms)),[100*v['mean'] for v in values],yerr=[100*v['std'] for v in values],capsize=3,
                          color=['#8994a7','#526f90','#3c8f92','#cf9a35','#774fa1'])
            axes[i,j].set_xticks(np.arange(len(arms)),arms,rotation=25,ha='right',fontsize=8)
            axes[i,j].set_title(f'{task.upper()} / {metric.upper()}')
            axes[i,j].set_ylabel('% of bundles or views')
            axes[i,j].set_ylim(0,100)
            axes[i,j].grid(axis='y',alpha=.2)
    fig.suptitle('Frozen local pilot: Qwen2.5-0.5B, 3 seeds; error bars = seed SD (not CI)')
    fig.savefig(RUN/'ood_comparison.png',dpi=180)
    plt.close(fig)
    passed = s['effectiveness_gates_passed']
    lines = ['# 本机关系约束后训练：审查、实现与验证', '',
             ('本轮通过预先固定的 OOD 效果门槛与完整性复核。' if passed else
              '**本轮未通过预先固定的效果门槛，不能向用户确认“该方案确实有效并有提升”。** 完整性复核通过与效果成立是两件事。'), '',
             '已在用户指定的 ChatGPT 网页对话中进行两轮方案讨论并在本机执行；没有使用服务器。网页意见是方案审查，结果来自保留的本地预测文件。', '',
             '## 固定实验', '',
             '- 基模：本地 Qwen2.5-0.5B-Instruct，固定 revision `7ae557604adf67be50417f59c2c2f167def9a775`；LoRA q/v，540,672 个可训练参数。',
             '- 5 个训练版本 × 3 个种子（17、29、43），每个 256 步，只保留最后一步；同种子共享初始化与参考抽样序列。没有挑选 seed、checkpoint 或测试后调参。',
             '- 每任务 1,024 个训练源题，仅 102 个已知标签；dev 128、ID-test 256、OOD 2,048 源题。NLI 每题原/反向两个视图；MCQA 每题四种循环选项排列。',
             '- 训练 BF16；评估 FP32 且禁用 TF32。主指标基于 A/B/C(/D) 标签 token 条件概率，并单独报告全词表首 token 的非法率。没有验证自由生成推理过程。',
             '- NLI 是命题逻辑文本，MCQA 是算术选项题。OOD 同时改变复杂度与措辞；不能外推到真实语言因果识别、干预效应或反事实估计。', '',
             '## 五个对照的含义', '',
             '| 版本 | 监督/约束 |', '|---|---|',
             '| sft | 只用原题已知标签 CE；保留同形批次的零权重计算 |',
             '| sft_compute | 用相同已知标签原题填满整个批次；匹配序列数和有效 token 预算 |',
             '| exact_aug | SFT + 可精确搬运的 MCQA 标签与 NLI 矛盾反向标签 |',
             '| consistency | exact_aug + 无标注 MCQA 语义对齐 JS、NLI 矛盾边缘 JS |',
             '| relational | consistency + 已知非矛盾原题的反向允许集合 {entailment, neutral} |', '',
             '后四者与基础监督共用同一可用 gold 集合。`sft_compute` 的预算匹配不等于完全相同 FLOPs；按长度选重复原题也会改变曝光频次。全词表 allowed-mass 还包含合法答案质量惩罚，不能把其增量全部解释为新语义能力。', '',
             '## OOD 结果', '',
             '数值为三种种子的均值（%）。ATC 表示整组变换都答对；RVR 为关系违反率，越低越好。', '',
             '| 任务 | 版本 | 全视图准确率 | ATC | RVR | 原始首 token 非法率 |', '|---|---|---:|---:|---:|---:|']
    for task in ('nli','mcqa'):
        for arm in ['base']+arms:
            v = s['aggregate']['ood'][task][arm]
            lines.append(f"| {task} | {arm} | {pct(v['accuracy']['mean'])} | {pct(v['atc']['mean'])} | {pct(v['rvr']['mean'])} | {pct(v['raw_first_token_invalid_rate']['mean'])} |")
    lines += ['', '![OOD comparison](ood_comparison.png)', '', '## 配对差值与冻结门槛', '',
              '差值单位为百分点，括号内为 2,000 次配对 bundle bootstrap 的 95% 区间。它以三个已训练种子为条件，不代表完整训练随机性；未进行多重检验校正。', '',
              '| 任务 | 比较 | 准确率差值 [CI] | ATC 差值 [CI] | RVR 差值 [CI] |', '|---|---|---:|---:|---:|']
    for task,pairs in s['comparisons']['ood'].items():
        for pair,v in pairs.items():
            lines.append(f"| {task} | {pair} | {ci(v['accuracy'])} | {ci(v['atc'])} | {ci(v['rvr'])} |")
    lines += ['', '成功要求两个任务分别相对 sft_compute 和 exact_aug 都满足：ATC 区间下界 > 0，准确率下界 > −1pp，RVR 上界 < 0。没有用任务均值掩盖单任务退化。', '',
              '| 任务 | 对照 | ATC 提升 | 准确率非劣 | RVR 下降 | 总门槛 |', '|---|---|---|---|---|---|']
    for task,controls in s['statistical_gates'].items():
        for control,g in controls.items():
            lines.append('| '+ ' | '.join([task,control]+['通过' if g[k] else '未通过' for k in ('atc_positive','accuracy_noninferior','rvr_lower','passed')])+' |')
    lines += ['', '两任务平衡 ATC 均值（描述项）：'+ '；'.join(f"{a} {pct(s['task_balanced_mean']['ood'][a]['atc'])}%" for a in arms)+'。', '',
              '## 资源、复核与限制', '',
              '| 版本/种子 | 有效 token | 预算误差 | 训练秒数 | 峰值显存 MiB |', '|---|---:|---:|---:|---:|']
    for r in training['records']:
        lines.append(f"| {r['arm']}/{r['seed']} | {r['nonpadding_tokens']:,} | {100*r['token_budget_relative_error']:.4f}% | {r['seconds']:.1f} | {r['peak_memory_mib']:.1f} |")
    lines += ['',
              '- 独立重算全部 6,912 源题的真值、语义拆分、102 个标签/任务的输入允许字段；跨 split 语义重合为零。',
              '- 核对 15 个最终 adapter、3,840 条训练日志、数据/代码/协议/模型哈希，并独立重算 96 个预测文件的指标、配对区间与成功门槛。',
              '- 以预定 dev 前四题对 sft_compute/relational 的 seed 17 重载模型并复现 logits；没有通过测试结果挑重放案例。逻辑变换 65,025 组穷举与数值机制检查通过。',
              '- **数据元信息限制**：NLI 源编号区间和 MCQA 编号模 4 可恢复原始标签。训练只编码 prompt，ID 未进入模型输入且未被解析为标签；本轮未发现实际模型泄露。但 public 文件不是强盲化数据，后续必须在新协议下更换为标签无关随机 ID。',
              '- NLI 单一标签预测、MCQA 固定语义错误或均匀分布都可能降低 RVR，所以 RVR 下降不能单独证明正确推理；报告保留标签直方图、熵和原始 logits。',
              '- MCQA JS 为标准等变一致性对照；relational−consistency 才是新增允许集合项的增量。联合 adapter 的跨任务影响不能当作独立 MCQA 新方法。',
              '- 本轮不能证明研究创新、真实自然语言因果推断或服务器规模有效性。若未过门槛，需要新假设、新协议、新 holdout；不能修改本轮测试口径获得“提升”。', '',
              '## 可复查文件', '',
              '- `protocol.json`：冻结协议；`implementation.json`、`evaluation_seal.json`：源/模型哈希。',
              '- `training.jsonl`、`console.log`、`evaluation_console.log`：完整执行记录。',
              '- `adapters/`：15 个最终模型；`predictions/`：各视图概率、logits、margin、原始 token、标签、选项映射。',
              '- `summary.json`：逐 seed 指标/区间；`verification.json`：独立完整性验证，保留摘要哈希。',
              '- `INDEPENDENT_REVIEW.md`、`DISCUSSION.md`：本地独立审查与网页版讨论决策。',
              '', '复现顺序：在项目 WSL 环境运行 `train.py` → `evaluate.py` → `verify_results.py` → `build_report.py`（完整目录下训练/评估先验证后复用最终结果）。', '']
    (RUN/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write_notebook(s,verification)
    print(json.dumps({'report':str(RUN/'REPORT.md'),'effectiveness_gates_passed':passed,'integrity_passed':True},ensure_ascii=False))


def write_notebook(summary,verification):
    import nbformat
    from nbclient import NotebookClient
    prefix = 'relational-local-v4-'
    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {'display_name':'Python 3','language':'python','name':'python3'}
    texts = [
        ('markdown', '## 本机关系约束后训练：冻结对照与独立验证\n\n本节来自真实执行的 Qwen2.5-0.5B 本机试验。5 个版本 × 3 个种子；详细报告见 `runs/local_scheme_validation/20260923_relational_v2/REPORT.md`。\n\n'+
         ('本轮通过冻结的 OOD 门槛，仅支持受控命题/算术试验的有限结论。' if summary['effectiveness_gates_passed'] else '**本轮没有通过冻结的效果门槛，不能确认方案已有效提升。**')+
         '\n\n训练数据的编号可透露标签顺序，但本轮只将 prompt 送入模型；详见独立审查。此实验不等同于真实自然语言的因果推断。'),
        ('code', "from pathlib import Path\nimport json, hashlib\nroot = Path.cwd()\nwhile not (root / 'runs/local_scheme_validation').exists() and root.parent != root:\n    root = root.parent\nrun = root / 'runs/local_scheme_validation/20260923_relational_v2'\nsummary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))\nverification = json.loads((run / 'verification.json').read_text(encoding='utf-8'))\nassert verification['all_passed']\nassert verification['summary_sha256'] == hashlib.sha256((run / 'summary.json').read_bytes()).hexdigest()\nprint('完整性复核:', verification['all_passed'])\nprint('冻结效果门槛:', summary['effectiveness_gates_passed'])\nprint('范围:', summary['scope'])"),
        ('code', "import pandas as pd\nfrom IPython.display import display, Image, Markdown\nrows = []\nfor task, arms in summary['aggregate']['ood'].items():\n    for arm, metrics in arms.items():\n        rows.append({'task': task, 'arm': arm, **{key: round(100 * metrics[key]['mean'], 3) for key in ('accuracy','atc','rvr','raw_first_token_invalid_rate')}})\ndisplay(pd.DataFrame(rows))\ndisplay(Image(filename=str(run / 'ood_comparison.png')))"),
        ('code', "rows = []\nfor task, pairs in summary['comparisons']['ood'].items():\n    for pair, metrics in pairs.items():\n        row = {'task': task, 'comparison': pair}\n        for name, result in metrics.items():\n            lo, hi = result['paired_bundle_95ci']\n            row[name + ' delta [95% CI], pp'] = f\"{100*result['mean']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]\"\n        rows.append(row)\ndisplay(pd.DataFrame(rows))\nprint(json.dumps(summary['statistical_gates'], ensure_ascii=False, indent=2))"),
        ('markdown', '### 继续研究的边界\n\nATC 要求同一源题的所有视图同时正确，RVR 仅衡量关系一致性。置信区间以三个固定训练种子为条件。只有新增允许集合项相对 `consistency` 的差值，才能衡量它的增量；全词表损失同时含格式监督，因此仍需进一步拆分。\n\n原始代码位于 `experiments/local_scheme_validation/`。所有最终 adapter、概率、logits、margin、日志和独立验证均保留。后续改进需要新协议和新测试集；本轮测试数据不得成为调参依据。')]
    for i,(kind,text) in enumerate(texts):
        cell = nbformat.v4.new_markdown_cell(text) if kind=='markdown' else nbformat.v4.new_code_cell(text)
        cell.id = prefix+str(i)
        nb.cells.append(cell)
    client = NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    assert all(o.output_type != 'error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
    durable = ROOT/'2.relational_local.ipynb'
    nbformat.write(nb,durable)
    target = ROOT/'2.ipynb'
    before = target.read_bytes()
    main = nbformat.reads(before.decode('utf-8'),as_version=4)
    retained = [c for c in main.cells if not c.get('id','').startswith(prefix)]
    main.cells = retained + nb.cells
    backup = RUN/f'2.before_local_appendix.{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.ipynb'
    backup.write_bytes(before)
    assert target.read_bytes() == before, 'Concurrent notebook save; standalone notebook is safe, retry merge later.'
    temp = target.with_suffix('.local.tmp.ipynb')
    nbformat.write(main,temp)
    temp.replace(target)
    check = nbformat.read(target,as_version=4)
    assert len(check.cells) == len(retained)+len(nb.cells)
    (RUN/'notebook_execution.json').write_text(json.dumps({'new_kernel_only':True,'user_kernel_restarted':False,
        'code_cells_executed':sum(c.cell_type=='code' for c in nb.cells),'error_outputs':0,
        'appended_cell_ids':[c.id for c in nb.cells], 'notebook_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'standalone_sha256':hashlib.sha256(durable.read_bytes()).hexdigest(),'backup':str(backup)},indent=2)+'\n')


if __name__ == '__main__': main()
