# Neurosymbolic Reasoning: Discrete Structures & Equivariant LLM Post-Training

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PEFT%20%7C%20Transformers-yellow)](https://huggingface.co/)

A rigorous, reproducible research framework investigating **discrete mathematical structures**, **permutation equivariance**, and **neurosymbolic relational post-training** across both classical neural encoders and modern decoder-only Large Language Models (LLMs).

---

## 🌟 Executive Summary

This repository hosts a multi-track investigation into the integration of formal discrete logic into deep neural representations:

1. **Track 2: Modern Decoder-LLM Relational Post-Training (Latest / Featured)**
   - **Backbone**: Qwen2.5-0.5B-Instruct (`7ae557604adf67be50417f59c2c2f167def9a775`) with LoRA ($r=8, \alpha=16$, 540,672 trainable parameters).
   - **Tasks**: Propositional Logic Natural Language Inference (NLI) & Multi-Choice Question Answering (MCQA with cyclic option permutations).
   - **Rigorous 5-Arm Causal Ablation**: Strictly token-matched ($344\text{k}$ non-padding tokens) across 3 paired random seeds (17, 29, 43).
   - **Key Finding**: **Unsupervised Permutation Equivariant Consistency (`consistency`, Arm 4) delivers the highest balanced performance (19.04% ATC)**, doubling permutation invariance on MCQA ($3.45\% \to 7.13\%$) and dropping relation violations by $11.3\,\text{pp}$. Conversely, partial relational set supervision (`relational`, Arm 5) provided **no additional marginal benefit** ($18.16\%$ vs $19.04\%$), honestly documenting the limitations of noisy set constraints in low-resource regimes.
2. **Track 1: Classical Geometric Order Embeddings (Historical Baseline)**
   - BiGRU and Attentive BiGRU encoders trained on official SNLI with directed energy cones and 3-way geometric regularization.

---

## 🔬 Track 2: Modern Decoder-LLM Relational Post-Training

### 1. Controlled Experimental Design & 5 Ablation Arms
All arms share the exact same budget of 102 gold labeled examples per task, identical reference token sequences ($344\text{k}$ tokens within $<0.02\%$ relative error), and identical LoRA initialization per seed:

| Arm | Description & Supervision Objective |
|:---|:---|
| **`sft`** | Pure supervised cross-entropy on base gold examples; zero-weight padding on permutation batches. |
| **`sft_compute`** | Token/Compute-matched pure SFT baseline; fills batches with repeated gold prompts to match sequence budget. |
| **`exact_aug`** | SFT + Exact symmetric augmentation (cyclic permutation for MCQA; contradiction reversal for NLI). |
| **`consistency`** | `exact_aug` + Unsupervised Jensen-Shannon (JS) divergence consistency across permuted views. |
| **`relational`** | `consistency` + Reverse non-contradiction partial allowed-set loss ($-\log[p_r(E) + p_r(N)]$) on gold NLI pairs. |

---

### 2. OOD Evaluation Results (Mean across 3 Seeds)

Evaluated under strict FP32 precision (TF32 disabled) across 2,048 Out-of-Distribution (OOD) source instances (8,192 MCQA queries, 4,096 NLI queries):
- **Accuracy**: Mean accuracy across all permutation views.
- **ATC (All-Transforms-Correct)**: Fraction of instances where **all** transformed/permuted views are answered correctly.
- **RVR (Relation Violation Rate)**: Fraction of pairs violating symmetry or cyclic equivariance (lower is better).

| Task | Arm | View Accuracy (%) | ATC (All Correct) (%) | RVR (Violation Rate) (%) | First-Token Invalid Rate (%) |
|:---|:---|---:|---:|---:|---:|
| **NLI** | `base` (Qwen2.5-0.5B Zero-shot) | 47.85 | 29.88 | 0.00* | 93.46 |
| **NLI** | `sft` | 44.83 | 26.61 | 38.20 | 0.00 |
| **NLI** | `sft_compute` | 47.20 | 28.17 | 35.95 | 0.12 |
| **NLI** | `exact_aug` | 47.22 | 29.98 | 34.16 | 0.00 |
| **NLI** | `consistency` | 44.17 | **30.96** | **28.89** | 0.00 |
| **NLI** | `relational` | 45.74 | 29.85 | 32.19 | 0.00 |
| **MCQA** | `base` (Qwen2.5-0.5B Zero-shot) | 24.21 | 0.00 | 90.76 | 100.00 |
| **MCQA** | `sft` | 30.08 | 3.79 | 63.23 | 0.00 |
| **MCQA** | `sft_compute` | 29.26 | 3.45 | 67.80 | 0.00 |
| **MCQA** | `exact_aug` | 31.34 | 5.96 | 59.97 | 0.00 |
| **MCQA** | `consistency` | **31.56** | **7.13** | **56.51** | 0.00 |
| **MCQA** | `relational` | 31.34 | 6.48 | 58.61 | 0.00 |

*\*Note: Base model NLI nominal RVR is 0.00 due to degenerate single-token bias; its open-vocabulary invalid token rate is 93.46%.*

![OOD Benchmark Comparison](runs/local_scheme_validation/20260923_relational_v2/ood_comparison.png)

---

### 3. Paired Differences & Bootstrap 95% Confidence Intervals

Differences reported in percentage points (pp) with 2,000 paired bundle bootstrap 95% confidence intervals:

| Task | Comparison | Accuracy Diff [95% CI] | ATC Diff [95% CI] | RVR Diff [95% CI] | Gate Status |
|:---|:---|---:|---:|---:|:---|
| **NLI** | `relational - sft_compute` | -1.46 [-2.45, -0.51] | +1.68 [+0.55, +2.82] | -3.76 [-5.37, -2.13] | Failed (Accuracy Inferior) |
| **NLI** | `relational - exact_aug` | -1.48 [-2.16, -0.83] | -0.13 [-0.98, +0.73] | -1.97 [-3.03, -0.86] | Failed (ATC Inconclusive) |
| **NLI** | `relational - consistency` | +1.57 [+0.81, +2.32] | **-1.11** [-2.00, -0.23] | **+3.30** [+2.18, +4.44] | Inversion (Consistency Wins) |
| **MCQA** | `relational - sft_compute` | +2.08 [+1.38, +2.78] | +3.03 [+2.31, +3.74] | -9.19 [-10.24, -8.11] | Passed |
| **MCQA** | `relational - exact_aug` | +0.00 [-0.37, +0.37] | +0.52 [+0.03, +0.99] | -1.36 [-2.09, -0.68] | Passed |
| **MCQA** | `relational - consistency` | -0.22 [-0.55, +0.11] | **-0.65** [-1.07, -0.23] | **+2.10** [+1.46, +2.74] | Inversion (Consistency Wins) |

**Overall Two-Task Balanced ATC Mean**:
$$\text{sft (15.20\%)} < \text{sft\_compute (15.81\%)} < \text{exact\_aug (17.97\%)} < \mathbf{relational (18.16\%)} < \mathbf{consistency (19.04\%)}$$

### 4. Key Takeaways & Scientific Attribution
1. **The True Source of Gains**: The major improvements in permutation equivariance and joint correctness come from **exact symmetric augmentation (`exact_aug`)** and **unsupervised JS-divergence consistency (`consistency`)**.
2. **The Limit of Weak Relational Sets**: Adding partial allowed-set supervision on top of consistency (`relational`, Arm 5) introduces gradient ambiguity without improving ATC or RVR.
3. **Pre-Registered Gate Honesty**: In accordance with pre-registered statistical gates ($\Delta\text{ATC} > 0$ and $\Delta\text{RVR} < 0$), the overall run is recorded as **failed** (`effectiveness_gates_passed: false`), upholding strict scientific integrity without p-hacking or retroactive metric tweaking.

---

## 📜 Track 1: Classical Geometric Order Embeddings

Order embeddings ([Vendrov et al., 2016](https://arxiv.org/abs/1511.06361)) project concepts into a partially ordered space (e.g., non-negative orthant $\mathbb{R}_+^d$) where entailment is coordinate-wise dominance:
$$u \ge v \iff \forall k, u_k \ge v_k, \quad E(u, v) = \frac{1}{d} \sum_{k=1}^d \max(0, v_k - u_k)^2$$

### Performance on SNLI (60k stratified training pairs, 9,824 test):
- **GRU Baseline**: 73.358% ± 0.840%
- **V2 3-Way Geometric Regularized GRU**: **73.470% ± 0.881%** (+0.112 pp paired gain across all seeds)
- **18/18 Structural Checks Passed**: Formal mathematical verification of triangle inequality, reflexivity, and non-negative projection.

---

## 🔒 Reproducibility & Independent Audit

This repository enforces an automated 7-point CPU audit ([`verify_results.py`](experiments/local_scheme_validation/verify_results.py)):
1. **Frozen Hashes**: SHA256 integrity verification across protocol, datasets, models, and code.
2. **Independent Truth Audit**: Semantic recomputation of all 6,912 instances; zero split overlap.
3. **Artifact Verification**: Audit of all 15 trained LoRA adapters ($540,672$ params each) and 3,840 training log lines.
4. **Metric Recomputation**: Independent CPU recalculation of all 96 prediction `.npz` files, margins, and confusion matrices.
5. **Statistical CIs**: Exact paired bundle bootstrap with fixed seed.
6. **Discrete Logic Checks**: 65,025 complete state evaluations verifying semantic invariance.
7. **Dev Logit Replay**: Exact FP32 forward pass replay on frozen dev instances matching saved logits within $10^{-4}$.

---

## 🚀 Quickstart

### 1. Setup Environment
```bash
git clone https://github.com/KQYaili/neurosymbolic-nli.git
cd neurosymbolic-nli
pip install -r requirements.txt
```

### 2. Run Track 2 (Modern Decoder-LLM Experiment)
```bash
# Generate synthetic logic datasets with sealed splits
python experiments/local_scheme_validation/synthetic_data.py --output-dir runs/local_scheme_validation/20260923_relational_v2

# Train all 15 LoRA adapters (5 arms x 3 seeds) with token-matching
python experiments/local_scheme_validation/train.py --run-dir runs/local_scheme_validation/20260923_relational_v2

# Evaluate all checkpoints in FP32
python experiments/local_scheme_validation/evaluate.py --run-dir runs/local_scheme_validation/20260923_relational_v2

# Run 7-point independent CPU audit and dev replay
python experiments/local_scheme_validation/verify_results.py --run-dir runs/local_scheme_validation/20260923_relational_v2

# Generate markdown report, charts, and standalone notebook
python experiments/local_scheme_validation/build_report.py
```

### 3. Run Track 1 (SNLI BiGRU Geometric Embeddings)
```bash
python experiments/neurosymbolic_nli/run_experiment_v2.py --run-dir runs/neurosymbolic_nli/20260915_order_v2
python experiments/neurosymbolic_nli/verify_results_v2.py --run-dir runs/neurosymbolic_nli/20260915_order_v2
python experiments/neurosymbolic_nli/logic_audit.py --output-dir runs/neurosymbolic_nli/20260915_order_v2
```

---

## 📁 Repository Structure

```
├── experiments/
│   ├── local_scheme_validation/  # Track 2: Modern Decoder-LLM (Qwen2.5-0.5B + LoRA)
│   │   ├── synthetic_data.py     # Controlled relational logic data generator
│   │   ├── train.py              # 5-arm controlled training with token budget match
│   │   ├── evaluate.py           # Sealed evaluation with bootstrap paired CIs
│   │   ├── verify_results.py     # 7-point independent audit & dev replay
│   │   └── build_report.py       # Chinese review, charts & notebook exporter
│   └── neurosymbolic_nli/        # Track 1: Classical BiGRU / GloVe Embeddings
│       ├── model_v2.py           # 3-Way Neurosymbolic Model & Loss
│       ├── data.py               # SNLI data pipeline & GloVe tokenizer
│       ├── run_experiment_v2.py  # Multi-arm paired runner
│       └── logic_audit.py        # 18-item discrete logic auditor
├── runs/
│   ├── local_scheme_validation/
│   │   └── 20260923_relational_v2/ # 15 LoRA adapters, 96 prediction files, REPORT.md
│   └── neurosymbolic_nli/
│       ├── 20260915_order_v1/    # V1 trial logs & artifacts
│       └── 20260915_order_v2/    # V2 3-way trial logs & artifacts
├── notebooks/
│   ├── 2.relational_local.ipynb  # Executed Track 2 interactive demonstration
│   └── neurosymbolic_nli_demo.ipynb # Track 1 demonstration
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📚 References

- Vendrov, I., Kiros, R., Fidler, S., & Urtasun, R. (2016). *Order-Embeddings of Images and Language*. ICLR 2016. [arXiv:1511.06361](https://arxiv.org/abs/1511.06361)
- Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015). *A large annotated corpus for learning natural language inference*. EMNLP 2015. [arXiv:1508.05326](https://arxiv.org/abs/1508.05326)
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Qwen Team. (2024). *Qwen2.5: A Party of Foundation Models*. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
