"""Generate comprehensive comparison report and plots for V2."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def build_report(run_dir):
    run_dir = Path(run_dir).resolve()
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    verified = json.loads((run_dir / 'verification.json').read_text(encoding='utf-8'))
    assert verified['all_passed'], "Cannot build report from unverified run!"
    protocol = json.loads((run_dir / 'protocol.json').read_text(encoding='utf-8'))
    completion = json.loads((run_dir / 'training_complete.json').read_text(encoding='utf-8'))

    # Generate multi-panel comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')

    arm_labels = {
        'neural_gru': 'Neural GRU',
        'v1_order_gru': 'Binary penalty GRU (λ=0.2)',
        'v2_order_gru': '3-Way heuristic GRU (λ=0.03)',
        'neural_bigru_attn': 'Neural BiGRU-Attn',
        'v2_order_bigru_attn': '3-Way heuristic BiGRU (λ=0.03)'
    }
    arm_colors = {
        'neural_gru': '#2563eb',
        'v1_order_gru': '#dc2626',
        'v2_order_gru': '#16a34a',
        'neural_bigru_attn': '#7c3aed',
        'v2_order_bigru_attn': '#059669'
    }

    # Panel 1: Bar chart with error bars across all arms
    arms = list(protocol['arms'].keys())
    x_positions = range(len(arms))
    means = [100 * summary['aggregate'][a]['accuracy']['mean'] for a in arms]
    stds = [100 * summary['aggregate'][a]['accuracy']['std'] for a in arms]

    bars = axes[0].bar(x_positions, means, yerr=stds, capsize=5,
                       color=[arm_colors[a] for a in arms], alpha=0.85)
    axes[0].set_xticks(x_positions, [arm_labels[a] for a in arms], rotation=25, ha='right', fontsize=9)
    axes[0].set_ylim(70, 76)
    axes[0].set_ylabel('Official Test Accuracy (%)')
    axes[0].set_title('Test Accuracy by Model Arm (Mean ± SD, 3 Seeds)')
    for bar, mean_val in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width() / 2, mean_val + 0.3, f"{mean_val:.2f}%",
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Panel 2: Paired difference by seed for GRU track: V1 minus Neural vs V2 minus Neural
    diff_v1 = np.array(summary['comparisons']['v1_minus_neural_gru']['by_seed']) * 100
    diff_v2 = np.array(summary['comparisons']['v2_minus_neural_gru']['by_seed']) * 100
    width = 0.35
    seeds = protocol['seeds']
    x_idx = np.arange(len(seeds))

    axes[1].axhline(0, color='#64748b', linestyle='--', linewidth=1)
    axes[1].bar(x_idx - width/2, diff_v1, width, label='Binary penalty - Neural', color='#dc2626', alpha=0.8)
    axes[1].bar(x_idx + width/2, diff_v2, width, label='3-Way heuristic - Neural', color='#16a34a', alpha=0.8)
    axes[1].set_xticks(x_idx, [f"Seed {s}" for s in seeds])
    axes[1].set_ylabel('Paired Accuracy Difference (pp)')
    axes[1].set_title('Track A (GRU): Paired Difference vs Neural Baseline')
    axes[1].legend(fontsize=8.5)

    # Panel 3: Training Dev Accuracy curves over epochs
    for a in arms:
        records_a = [r for r in completion['records'] if r['arm'] == a]
        curves = np.array([[h['dev']['accuracy'] * 100 for h in r['history']] for r in records_a])
        axes[2].plot(range(1, protocol['epochs'] + 1), curves.mean(axis=0),
                     marker='o', label=arm_labels[a], color=arm_colors[a], linewidth=1.5)

    axes[2].set_xticks(range(1, protocol['epochs'] + 1))
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Official Dev Accuracy (%)')
    axes[2].set_title('Dev Accuracy Trajectory (Mean of 3 Seeds)')
    axes[2].legend(fontsize=7.5)

    for ax in axes:
        ax.grid(axis='y', alpha=0.25)
        ax.set_axisbelow(True)

    plot_path = run_dir / 'snli_v2_comparison.png'
    fig.savefig(plot_path, dpi=180)
    fig.savefig(run_dir / 'snli_v2_comparison.pdf')
    plt.close(fig)

    # Write Markdown Report
    rows = []
    for r in summary['results']:
        acc = r['test']['accuracy'] * 100
        f1 = r['test']['macro_f1'] * 100
        rows.append(f"| {r['arm']} | {r['seed']} | {acc:.3f}% | {f1:.3f}% |")
    results_table = "\n".join(rows)

    comp = summary['comparisons']
    v1_mean = comp['v1_minus_neural_gru']['mean'] * 100
    v2_mean = comp['v2_minus_neural_gru']['mean'] * 100
    ci_v2 = [x * 100 for x in comp['v2_minus_neural_gru']['bootstrap_95ci']]
    v2_bigru_mean = comp['v2_minus_neural_bigru_attn']['mean'] * 100
    architecture_difference = 100 * (
        summary['aggregate']['neural_bigru_attn']['accuracy']['mean']
        - summary['aggregate']['neural_gru']['accuracy']['mean'])
    objective_difference = v2_mean - v1_mean
    aggregate_rows = []
    for arm in arms:
        aggregate = summary['aggregate'][arm]
        acc, f1 = aggregate['accuracy'], aggregate['macro_f1']
        aggregate_rows.append(
            f"| `{arm}` | {100*acc['mean']:.3f}% ± {100*acc['std']:.3f} pp "
            f"| {100*f1['mean']:.3f}% ± {100*f1['std']:.3f} pp |")
    aggregate_table = '\n'.join(aggregate_rows)

    report_content = f"""# Neurosymbolic NLI V2 实验复核报告

本报告根据已有 `summary.json` 纠正原报告的理论解释与数值结论；没有改动 V2 模型、训练脚本、协议或检查点。V2 三项几何惩罚在 GRU 上相对同架构无辅助损失基线平均变化 **{v2_mean:+.3f} 个百分点**，已有区间包含零，尚不能据此认定提升。BiGRU 注意力池化也没有带来原报告所称的大幅提升。

任务是 SNLI 三分类自然语言推断，不估计干预效应。训练使用 60,000 条样本、种子 17/29/43、4 个 epoch 的最终检查点；同架构各臂共享初始化与数据顺序。官方测试集此前已用于 V1，这里属于继续探索后的评测，不是新的独立留出验证。原报告备份见 `../20260915_review_v3/v2_original_report/REPORT.md`。

## 一、 实际实现与数学限制

1. **三项启发式几何惩罚**：蕴涵降低正向能量并抬高反向能量；矛盾降低余弦相似度；中立抬高双向能量。这些是建模假设，不是三值逻辑公理。蕴涵允许语义等价，不能普遍强制反向不蕴涵；中立也允许反向蕴涵。矢量正交不等价于命题互斥，代码亦没有中立的非零重叠下界。V1 的交叉熵始终保留三分类，只在辅助能量损失中合并两类非蕴涵，不能据此称其分类语义错误。
2. **LayerNorm 并非无条件有利**：输出经逐句 LayerNorm 再 Softplus。若共享缩放参数 gamma 全为正，严格坐标包含只能发生在两个输出相等时，因为归一化坐标总和均为零。复核的 15 个最终检查点 gamma 全为正，因此当前表示无法实现严格的零能量单向包含。LayerNorm 的可学习仿射参数仍能改变尺度，未证明其消除了尺度问题。
3. **几何特征适配器**：将四个标量特征映射到八维是扩维，不是降维；原 V1 也未拼接高维几何坐标。没有单独的特征适配器消融，不能声称消除了过拟合。
4. **BiGRU 加句内注意力池化**：代码独立编码两句话，再拼接句向量特征，没有跨句词语对齐。其改动包含方向与循环单元宽度变化；是否更好须看实测。
5. **消融范围**：`v1_order_gru` 是在 V2 架构上使用二元辅助损失，并非原 V1 的完全重现。它的边界为 0.15，原 V1 为 0.2；V2 还同时改动了投影和分类特征。`v2_order_gru` 同时改动损失形式及权重（0.2 到 0.03），不能单独识别损失形式的收益。

## 二、 完整实验结果（已使用过的官方测试集）

| 模型臂 (Arm) | 种子 (Seed) | 测试准确率 (Accuracy) | 测试 Macro-F1 |
|---|---:|---:|---:|
{results_table}

### 汇总统计（3 个种子均值 ± 样本标准差）

| 模型臂 | 准确率 | Macro-F1 |
|---|---:|---:|
{aggregate_table}

## 三、 配对差值分析

- **Track A: GRU 架构内的配对对比**
  - `v1_order_gru` 相较于 `neural_gru` 平均变化 **{v1_mean:+.3f} 个百分点**，方向为正，没有重现原 V1 的负向结果。
  - `v2_order_gru` 相较于 `neural_gru` 平均变化 **{v2_mean:+.3f} 个百分点**，逐样本配对 95% Bootstrap 区间为 **[{ci_v2[0]:+.3f}, {ci_v2[1]:+.3f}] 个百分点**，包含零。
  - 三项惩罚相较于二元惩罚均值变化 **{objective_difference:+.3f} 个百分点**；本轮不能说明三项惩罚优于二元惩罚。
- **Track B: BiGRU-Attn 架构内的配对对比**
  - `v2_order_bigru_attn` 相较于 `neural_bigru_attn` 平均变化 **{v2_bigru_mean:+.3f} 个百分点**，三个种子中分别为正、零、负。
  - 无辅助损失的 BiGRU-Attn 相较于 GRU 均值变化 **{architecture_difference:+.3f} 个百分点**，不支持“大幅提升”。

上述 Bootstrap 先平均三个固定种子模型的逐样本配对差，再对样本重抽样；它没有涵盖完整的训练种子不确定性，也没有按共享图像/前提进行簇重抽样。三种子标准差与该区间表达不同的不确定性，不能互相替代。

![SNLI V2 Comparison](snli_v2_comparison.png)

## 四、 后续改进依据

现有结果没有建立三项几何惩罚的可靠优势，也没有测量“共现特征被破坏”这一机制。下一版应修正不成立的语义约束，加入真正的跨句词语对齐，并在相同网络、初始化和数据下比较有无逻辑辅助损失；用开发集选择方案、用新的冻结留出集验证。标准集合语义允许的关系（如矛盾对称性、反向标签的部分约束）比强制所有蕴涵不对称更合适，但仍须通过公平消融检验。

完整 review、数学反例与执行建议见 `../20260915_review_v3/REVIEW_V2.md`。原 `verification.json` 验证的是列出的工程检查，不能证明上述语义假设或提升结论。
"""
    (run_dir / 'REPORT.md').write_text(report_content, encoding='utf-8')
    print(f"REPORT.md and plots generated successfully in {run_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    build_report(parser.parse_args().run_dir)
