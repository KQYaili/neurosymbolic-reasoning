# 本机预检：等待方案正文

状态：`awaiting_scheme`。本次仅验证本机环境与一个固定基模的离线前向计算。未训练、未下载模型、未修改环境；尚未验证研究方案是否有效或带来提升。

- 基模：`Qwen/Qwen2.5-0.5B-Instruct`
- Revision：`7ae557604adf67be50417f59c2c2f167def9a775`
- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU
- Python：`/home/lgd/anaconda3/bin/python`
- 参数量：494,032,768
- BF16 输出形状：`[1, 7, 151936]`，全部有限：`True`
- 实测显存峰值（allocated）：989.93 MiB
- 加载与前向耗时：1.247 秒

此显存结果仅针对一条 7-token 输入的推理；不能作为训练或长上下文的显存估计。其他缓存仅列目录及文件完整性，未额外运行模型。

所有模型文件的 SHA-256、当前资源和包版本见 `preflight.json`，实际标准输出与标准错误保存在 `console.log`。

复现（WSL Ubuntu）：

```bash
cd /mnt/c/Users/lgd/PycharmProjects/JupyterProject1
/home/lgd/anaconda3/bin/python experiments/local_scheme_validation/preflight.py
```

读取指定网页方案之后，才制定模型训练、模拟数据、对照组和独立留出集。
