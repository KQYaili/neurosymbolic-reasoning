# 冻结本机实验的独立审查（评估前）

审查日期：2026-09-23。审查范围为 `train.py`、`evaluate.py`、`synthetic_data.py`、`extend_evaluation.py`、`protocol.json` 及已冻结的数据文件。本审查没有修改训练代码、协议或数据，没有启动 GPU，也没有读取模型在 dev/test/OOD 上的预测或按结果选择样本。

## 结论

当前数学映射、损失中的类别索引、逐任务指标与预注册 gate 的实现相符。独立 CPU 重算了全部 6,912 个源问题的真值和跨拆分语义签名，未发现错误答案或跨拆分重复语义问题。初次审查发现的评估完整性检查缺口与缺少任务均值，已反馈并在评估启动前看到主代理修复。

本轮仍需明确两项解释边界：公开训练 ID 可以恢复隐藏标签，但现有训练器不把 ID 输入模型、也不由 ID 构造监督；新增 allowed-mass 项同时包含标签格式概率项，因而其增量不是纯粹的语义集合约束效果。它们不要求重写已经冻结的训练，但限制结论的措辞和下一轮设计。

## 发现与处理

### 1. [P1，评估前已修复] 评估曾未完整核对训练合同

初次读取的 `evaluate.main` 只核对 implementation 文件哈希、源码哈希、adapter 权重与记录数量，未将当前 `protocol.json`、`train_public.json`、`data_manifest.json`、基模/词表文件重新比对 implementation 内的冻结哈希，也未验证所有 `(arm, seed)` 唯一且齐全。LoRA 配置文件不在评估 seal 中。这样，当前磁盘上的协议、词表或 adapter 配置变动可能被作为原训练的评估继续执行。

主代理随后加入上述合同校验、准确的 15 组 `(arm, seed)` 集合与数量校验、每条记录 contract 校验、LoRA 配置对协议的匹配，以及 adapter 配置文件哈希封存。已重新阅读这些修复。最终独立结果验证仍应复核 seal、权重/config 和预测文件，而非仅信任 `summary.json` 的布尔值。

### 2. [P2，需披露；后续新数据修复] public ID 是隐藏标签的可恢复侧信道

`make_nli` 在逐类配额循环内生成数字 ID；`make_mcqa` 令正确位置等于源索引模 4。`training_view` 移除了 AST 和 oracle 字段，但保留这些 ID。实际独立检查：

| 训练任务 | 仅由 ID 恢复原标签的方法 | 正确恢复 | 其中未标注样本 |
|---|---|---:|---:|
| NLI | 数字段 `<342` 为 A，`342..682` 为 B，`>=683` 为 C | 1024/1024 | 922/922 |
| MCQA | 数字段模 4 对应 A/B/C/D | 1024/1024 | 922/922 |

这不是现有模型已利用隐藏标签的证据：`encode_prompt` 仅接收 `variants[v]['prompt']`；ID 仅用于缓存键、批次哈希和文件关联，损失读取明确的 `supervised_label`/`allowed_labels`，没有解析数字 ID。已检查的代码路径没有将 ID 特征输入模型或恢复未知标签。

本轮报告应准确写“未标注样本没有显式标签/AST 字段，训练器未使用可恢复 ID 信息”，不要称其为不可恢复标签的盲化数据。独立验证器应确认送入 tokenizer 的字符串只来自 prompt。下一轮新协议的数据应在标签分组完成后使用独立随机 UUID/随机重编号；不要改动本轮冻结数据或据此重跑挑选结果。

### 3. [P2，解释限制] 新增 allowed-mass 项包含格式概率约束

`train.py` 的 `allowed_mass` 对全词表 log-softmax 取允许标签概率和。对 NLI 非矛盾逆向集合 `{A,C}`，精确分解为：

`-log(P_vocab(A)+P_vocab(C)) = -log(Z_ABC) - log(P_class(A)+P_class(C))`，其中 `Z_ABC=P_vocab(A)+P_vocab(B)+P_vocab(C)`。

因此 relational 相对 consistency 的额外信号同时提高 A/B/C 总概率，并抑制条件类别 B。三个类别内的后一项等价于二元“不是矛盾”监督。不能把其全部增量归因于一种新的通用集合学习方法。已知矛盾标签只参与 exact 项，非 singleton 项通过互补 mask 排除它，因此没有发现重复加权矛盾样本的错误。

当前协议明确使用全词表损失，可以保持冻结。报告应分开呈现有效标签概率质量、raw token 正确/无效比例、条件类别正确性，并将 relational−consistency 的贡献单列。若下一轮要识别纯语义作用，需要预注册格式匹配对照或条件集合损失及全新 holdout。

### 4. [P2，评估前已修复] 缺少协议指定的任务均值

协议的 primary endpoint 除逐任务 OOD ATC 外还指定 task-balanced mean。初版 evaluator 只输出逐任务 aggregate。主代理已补充各训练 seed 的两任务等权均值，再对 seed 汇总的描述项；逐任务 gate 保持不变。不要把 NLI 的两个视图和 MCQA 的四个视图直接混池，否则会改变任务权重。

### 5. [解释限制] token 匹配并非 FLOPs 严格相等

`sft_compute` 使用同任务中 token 长度最近的已标注原题替代每个训练视图；其他 arms 的额外视图及损失计算图与它不同。因此同 256 步、16 条/步、非 padding token 总量误差不超过 1% 支持“token 预算匹配”，不支持逐 FLOP 相等。最近长度采样也改变了标注题的曝光频次。应保留实际 token、padding、时长、采样哈希，并区分“允许使用的 102 个金标签相同”和“每个标签实际出现次数相同”。

## 已完成的独立 CPU 检查

1. 未调用生成器的真值函数：用 Python 集合穷举每个 NLI AST 的所有世界，独立按子集/不相交/剩余三分规则判断正反关系。全部 3,456 个 NLI 源问题通过。
2. 独立按六种算术运算重算答案并在每个 option-values 列表中定位，全部 3,456 个 MCQA 源问题通过；四个选项唯一、语义 ID 排列及 base-to-variant 映射均通过。
3. 独立重建无序 NLI 真值集合对与规范化算术表达式签名。train/dev/test/OOD 的六组两两交集全为 0；全部 6,912 个 ID 唯一且与 oracle 项数一致。
4. public allow-list 仅含预定键；每任务 1,024 个训练源、102 个标注源；未标注所有变换均无显式 supervised/allowed 标签。
5. 从 evaluator AST 仅加载两个纯 NumPy 指标函数，没有 import Torch/PEFT。枚举 NLI 全部 9 个预测对，确认矛盾/非矛盾不一致才计 RVR；测试 MCQA 四个语义选项在四个循环排列下 RVR=0，而固定显示位置 A 得 RVR=1。ATC/accuracy 的轴方向正确。

## 统计与结论边界

- `compare` 先逐 seed 计算同一源问题的成对差，再按源问题对三个固定 seed 求均值并 bootstrap，符合协议声明的“给定三个已训练 seed”的条件 CI。它不包括重新训练的 seed 总体不确定性。三个 seed 的单独差值须保留。
- ATC、平均视图 accuracy、RVR 的正负方向与门槛正确；分别对两任务、两个指定对照执行。某对照 RVR 已经为 0 时，严格下降 gate 可能无法通过；应报告 gate 未通过，不应事后放松门槛。
- 判定有效须同时满足统计 gate 和独立完整性验证。只有 RVR 下降、条件标签概率增大或 loss 下降不够。
- MCQA 的一致性使用已知的选项排列等变性；NLI swap 约束是健全但不完备的。常数/均匀预测仍可满足一致性，故 correctness/ATC 必须保留。
- 该数据验证受控命题 NLI 与算术选择题，不含干预、反事实因果效应或混杂结构，不能据此声称“自然语言因果推断已验证”。OOD 同时改变题目复杂度和措辞，无法单独归因其中一个因素。
- 固定类别单 token 评分与自由生成不同。新增保存 class logits、top-two margin 有助审计 FP32 近似平票；不能按 margin 删除不利样本。

本报告形成于预测结果公开之前。不存在“实验已提升”的结论；该判断须等待全部冻结模型评估与独立复算。
