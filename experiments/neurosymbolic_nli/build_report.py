"""Render frozen experiment results and append executable notebook follow-up cells."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/neurosymbolic_nli/20260915_order_v1'
SECTION_ID = 'neurosymbolic-nli-v1-intro'


def build_report():
    summary = json.loads((RUN / 'summary.json').read_text())
    verified = json.loads((RUN / 'verification.json').read_text())
    assert verified['all_passed']
    protocol = json.loads((RUN / 'protocol.json').read_text())
    completion = json.loads((RUN / 'training_complete.json').read_text())
    a, b = summary['aggregate']['neural'], summary['aggregate']['order_regularized']
    delta = 100 * summary['order_minus_neural_accuracy_mean']
    ci = [100 * x for x in summary['paired_example_bootstrap_95ci']]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), layout='constrained')
    colors = {'neural': '#2563eb', 'order_regularized': '#d97706'}
    labels = {'neural': 'Neural (CE)', 'order_regularized': 'CE + order constraint'}
    for j, arm in enumerate(protocol['arms']):
        value = summary['aggregate'][arm]['accuracy']
        axes[0].errorbar(j, 100 * value['mean'], yerr=100 * value['std'],
                         color=colors[arm], capsize=6, fmt='o', markersize=9, linewidth=2)
        axes[0].text(j, 100 * value['mean'] + 1.6, f"{100*value['mean']:.2f}%", ha='center')
    axes[0].set_xticks([0, 1], ['Neural', 'Order regularized'])
    axes[0].set_xlim(-.5, 1.5)
    axes[0].set_ylim(68, 77)
    axes[0].set_ylabel('Official-test accuracy (%)')
    axes[0].set_title('Three seeds; mean ± SD')
    differences = np.array(summary['order_minus_neural_accuracy_by_seed']) * 100
    axes[1].axhline(0, color='#64748b', linewidth=1)
    axes[1].bar(range(3), differences, color='#d97706')
    axes[1].set_xticks(range(3), [str(s) for s in protocol['seeds']])
    axes[1].set_xlabel('Paired initialization seed')
    axes[1].set_ylabel('Accuracy difference (percentage points)')
    axes[1].set_title('Order regularized minus neural')
    for arm in protocol['arms']:
        curves = np.array([[h['dev']['accuracy'] * 100 for h in record['history']]
                           for record in completion['records'] if record['arm'] == arm])
        axes[2].plot(range(1, 5), curves.mean(0), marker='o', color=colors[arm], label=labels[arm])
    axes[2].set_xticks(range(1, 5))
    axes[2].set_xlabel('Epoch (final epoch retained)')
    axes[2].set_ylabel('Official-dev accuracy (%)')
    axes[2].set_title('Training progress; no test selection')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    plot_path = RUN / 'snli_comparison.png'
    fig.savefig(plot_path, dpi=170)
    fig.savefig(RUN / 'snli_comparison.pdf')
    plt.close(fig)
    rows = '\n'.join(f"| {r['arm']} | {r['seed']} | {100*r['test']['accuracy']:.3f}% | {100*r['test']['macro_f1']:.3f}% |"
                     for r in summary['results'])
    conclusion = (f"本轮偏序正则的平均测试准确率为 {100*b['accuracy']['mean']:.3f}%，"
                  f"神经对照为 {100*a['accuracy']['mean']:.3f}%，差值 {delta:+.3f} 个百分点。"
                  "三个种子均下降，因此本轮不支持“偏序正则提高 SNLI 准确率”的结论。")
    report = f'''# 从四句序嵌入演示到可复现的自然语言蕴涵实验

{conclusion}

本项目沿用 `2.ipynb` 末尾的 GRU＋序嵌入路线。任务是 SNLI 三分类（蕴涵、矛盾、中立），不估计干预因果效应。默认推理使用当前较好的神经对照；偏序模型保留为结构研究与消融对照。

## 实际改动

- 用有效长度与 packed GRU 排除 padding；同一句右侧增加 padding 后输出不变。
- 用 Softplus 正值投影替代演示中的 ReLU，增加三分类头，保留中立与矛盾的区别。
- 两组具有相同模型结构、初始化、参数量、数据次序与训练步数。唯一实验变量是偏序辅助损失权重：0 对 0.2。
- 偏序损失只约束真实蕴涵边与非蕴涵边，不把所有反向边自动当负例。
- 增加离散图闭包、路径见证和 SCC 等价类；它们验证给定边上的结构，不证明神经预测边是真实语义。
- 原四句演示改为局部函数，避免覆盖 SNLI 的全局 `vocab`，并修正“零样本”“反对称律”的过度表述。
- 补齐当前笔记本依赖，修复多输入训练计数和模型设备引用，保留 IMDb 本地解压缓存。

## 固定协议

官方 train 分层抽取 60,000 条（每类 20,000）；官方 dev 全部 9,842 条；官方 test 全部 9,824 条。词表仅来自抽样训练集，共 12,187 个词元，GloVe 100 维；句对在三个划分间无重复。种子 17、29、43，每组 4 轮，batch 256，Adam 0.001，梯度范数上限 1，FP32 禁用 TF32。所有训练结束后才对最终 checkpoint 进行测试；不根据测试结果选择 epoch 或调整本轮参数。

## 结果

| 方法 | 种子 | 测试准确率 | 测试 Macro-F1 |
|---|---:|---:|---:|
{rows}

神经对照平均 Macro-F1：{100*a['macro_f1']['mean']:.3f}%；偏序正则：{100*b['macro_f1']['mean']:.3f}%。逐测试样本配对 bootstrap 的差值 95% 区间为 [{ci[0]:+.3f}, {ci[1]:+.3f}] 个百分点；该区间以本次三个已训练种子对为条件，不代表完整的随机种子不确定性。

![SNLI comparison](snli_comparison.png)

## 数学核查与结论边界

精确的坐标支配关系满足反身性、向量反对称性和传递性；非零能量阈值不保证传递。反例 A=0、B=0.2、C=0.4 时，两条相邻边能量均 0.04<0.05，远端边能量为 0.16。近似关系满足有向三角界 `sqrt(E(A,C)) <= sqrt(E(A,B)) + sqrt(E(B,C))`。

自然语言中的语义等价句允许双向蕴涵；先按等价关系取商，才讨论句子语义的偏序。狗跑步与狗吠叫可以同时发生，演示中的动作不匹配只能作为非蕴涵例子。四句中只留出 A→C 关系，不能称为未见句子的零样本验证。

结构审计 18/18 通过；逐样本指标、配对初始化、最终训练步数、数据/代码哈希和训练后 padding 不变性核查见 `verification.json`。数学验证与自然语言预测准确率分别报告，不能互相替代。

针对性诊断还暴露了具体失败：固定 seed17 神经模型把 “A man is sleeping.” → “A man is awake.” 和 “A person is standing.” → “A person is wearing a hat.” 都预测为蕴涵。按同一主体、同一时刻的通常语义，前者应为矛盾、后者为中立。因此该模型仍会依赖词面重叠；这些失败被保留在笔记本中，没有用手工覆盖预测掩盖。三个探针仅是诊断样例，不是新的统计基准。

## 复现与文件

在项目根目录、WSL Python 环境运行：

```bash
python experiments/neurosymbolic_nli/data.py --run-dir runs/neurosymbolic_nli/20260915_order_v1
python experiments/neurosymbolic_nli/run_experiment.py --run-dir runs/neurosymbolic_nli/20260915_order_v1
python experiments/neurosymbolic_nli/verify_results.py --run-dir runs/neurosymbolic_nli/20260915_order_v1
```

已有最终 checkpoint 时复现入口复用它们，不重新训练。`protocol.json`、`implementation.json`、`split_manifest.json` 固定协议与来源；`training.jsonl`/`console.log` 保留训练过程；`checkpoints/` 保留六份最终参数；`*_test_predictions.npz` 保留六组逐样本概率和预测；`summary.json` 与 `verification.json` 保留汇总和核查。

下一步如探索更强语义模型或约束形式，应重新登记实验并仅使用开发集选择方案；本次负结果不能用继续试探测试集掩盖。

## 主要依据

- [Order-Embeddings of Images and Language](https://arxiv.org/abs/1511.06361)：原序嵌入研究的 SNLI 实验把矛盾和中立合并为非蕴涵，不能直接与本项目三分类准确率对比。
- [SNLI 官方数据说明](https://nlp.stanford.edu/projects/snli/)：官方 train/dev/test 划分与三分类标注。
'''
    (RUN / 'REPORT.md').write_text(report, encoding='utf-8')
    return summary, conclusion


def append_notebook(summary, conclusion):
    path = ROOT / '2.ipynb'
    before = path.read_bytes()
    nb = nbformat.reads(before.decode('utf-8'), as_version=4)
    # Preserve the user's existing work; only replace this generated continuation.
    nb.cells = [c for c in nb.cells if not c.get('id', '').startswith('neurosymbolic-nli-v1-')]
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and '偏序集与几何格理论' in cell.source:
            cell.source = cell.source.replace('将逻辑关系离散化为偏序代数', '将集合关系离散化为关系代数')
            cell.source = cell.source.replace('偏序集（Poset）与布尔格（Boolean Lattice）', '偏序集（Poset）与几何包含关系（一般盒集合不等同于布尔格）')
            cell.source = cell.source.replace('蕴涵关系在离散数学中本质上是一个偏序关系（自反性、反对称性、传递性）', '逻辑蕴涵在句子上形成预序；按语义等价取商后才形成偏序（自反性、反对称性、传递性）')
    cells = []

    def md(suffix, text):
        cells.append(nbformat.v4.new_markdown_cell(text, id='neurosymbolic-nli-v1-' + suffix))

    def code(suffix, text):
        cells.append(nbformat.v4.new_code_cell(text, id='neurosymbolic-nli-v1-' + suffix))

    md('intro', '''## 续作：GRU、偏序约束与离散图推理的可验证实验

本节接续上面的四句 `OrderEmbeddingNLI` 演示，使用真实 SNLI 三分类进行配对验证。这里的任务是**自然语言蕴涵识别**，不是干预效应估计。

改进包括有效长度编码、正值序投影、三分类监督、偏序辅助损失，以及可追踪的离散图闭包。对照模型与偏序模型使用同一结构，唯一实验变量是辅助损失权重。下面直接加载已经完成的结果，不会启动大规模训练。''')
    code('load', '''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image

nli_project = Path.cwd()
assert (nli_project / 'experiments/neurosymbolic_nli/model.py').is_file(), '请从项目根目录打开此笔记本'
nli_run = nli_project / 'runs/neurosymbolic_nli/20260915_order_v1'
nli_summary = json.loads((nli_run / 'summary.json').read_text(encoding='utf-8'))
nli_verification = json.loads((nli_run / 'verification.json').read_text(encoding='utf-8'))
assert nli_verification['all_passed']
print('六次最终训练与逐样本核查已完成。')
print('训练/开发/测试：60,000 / 9,842 / 9,824；固定种子：17, 29, 43。')''')
    md('method', r'''### 方法与数学边界

沿用 GRU 编码，但用 `pack_padded_sequence` 排除 padding；序投影用 Softplus，不做单位 L2 归一化。三分类头保留 entailment / contradiction / neutral。

\[
E(P,H)=\frac1d\sum_k\max(0,z_{H,k}-z_{P,k})^2,
\quad L=L_{CE}+\lambda\,\mathbb{E}[\mathbf{1}_{y=E}E(P,H)+\mathbf{1}_{y\ne E}\max(0,m-E(P,H))].
\]

两组使用相同初始化、训练样本顺序和最终训练轮数，分别取 $\lambda=0$ 与 $0.2$，$m=0.2$。不自动把反向蕴涵标为负例。精确坐标序的传递性不能推广为任意正阈值的分类保证；序损失也不能替代自然语言测试集。''')
    code('model', '''from experiments.neurosymbolic_nli.model import NeuralOrderNLI, order_penalty
import inspect
print(inspect.getsource(NeuralOrderNLI))
print(inspect.getsource(order_penalty))''')
    code('logic', '''from experiments.neurosymbolic_nli.logic_audit import run_logic_audit
nli_logic = run_logic_audit(nli_run)
assert nli_logic['all_passed']
display(pd.DataFrame({'check': list(nli_logic['checks']), 'passed': list(nli_logic['checks'].values())}))
print('非零阈值反例：', nli_logic['threshold_counterexample'])
print('已知边上的离散推理路径：', nli_logic['held_out_edge_demo'])''')
    code('results', '''nli_rows = [{'method': r['arm'], 'seed': r['seed'],
             'test_accuracy_%': 100 * r['test']['accuracy'],
             'test_macro_F1_%': 100 * r['test']['macro_f1']}
            for r in nli_summary['results']]
display(pd.DataFrame(nli_rows).round(3))
display(Image(filename=str(nli_run / 'snli_comparison.png')))
print('平均准确率差值（偏序组－神经组，百分点）：',
      100 * nli_summary['order_minus_neural_accuracy_mean'])''')
    code('predict', '''from experiments.neurosymbolic_nli.inference import predict_pairs
nli_examples = [
    ('A dog is running .', 'An animal is running .'),
    ('A man is sleeping .', 'A man is awake .'),
    ('A person is standing .', 'A person is wearing a hat .'),
]
# 固定第一个种子展示，不按测试集挑选最好种子；以下是模型预测，不是形式证明。
nli_predictions = predict_pairs(nli_run, nli_examples, arm='neural', seed=17, device='cpu')
nli_probe_expected = ['entailment', 'contradiction', 'neutral']
for nli_prediction, nli_expected in zip(nli_predictions, nli_probe_expected):
    nli_prediction['expected_diagnostic_label'] = nli_expected
    nli_prediction['matches_expectation'] = nli_prediction['prediction'] == nli_expected
display(pd.DataFrame(nli_predictions))''')
    code('reproduce', '''import sys
nli_reproduce_command = [sys.executable,
    str(nli_project / 'experiments/neurosymbolic_nli/run_experiment.py'),
    '--run-dir', str(nli_run)]
print('复现命令：', nli_reproduce_command)
print('完整说明：', nli_run / 'REPORT.md')
print('核查通过项目：', sum(nli_verification['checks'].values()))''')
    md('conclusion', conclusion + '''

本轮采用普通神经对照作为默认分类模型，保留偏序模型与离散闭包用于进一步研究。18 项结构核查通过，六次测试预测均独立重算通过；这些结果证明实现与验证可复现，不能证明逻辑约束一定提高自然语言准确率。

原四句示例只留出一条关系边，并未留出句子；“吠叫”与“跑步”也不是逻辑矛盾。严格结构性质、近似阈值行为和真实文本泛化必须分开报告。

诊断样例中仍有明显错误（如 sleeping→awake 被预测为蕴涵），说明模型仍可能依赖词面重叠。这里完整展示失败，模型输出不能视为形式逻辑证明。

依据：[序嵌入原论文](https://arxiv.org/abs/1511.06361)、[SNLI 官方数据说明](https://nlp.stanford.edu/projects/snli/)。''')
    nb.cells.extend(cells)
    nbformat.validate(nb)
    assert path.read_bytes() == before, 'Notebook changed during append; rerun to merge current work'
    (RUN / '2.before_continuation.ipynb').write_bytes(before)
    text = nbformat.writes(nb) + '\n'
    path.write_text(text, encoding='utf-8')
    output = ROOT / '2.neurosymbolic.ipynb'
    output.write_text(text, encoding='utf-8')
    (RUN / 'notebook_append.json').write_text(json.dumps({
        'main': str(path), 'durable_copy': str(output),
        'added_cell_ids': [c.id for c in cells], 'cell_count': len(nb.cells)
    }, indent=2) + '\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    summary, conclusion = build_report()
    print(append_notebook(summary, conclusion))
