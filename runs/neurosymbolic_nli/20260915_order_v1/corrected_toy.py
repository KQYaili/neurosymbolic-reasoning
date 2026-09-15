import torch
import torch.nn as nn
import torch.optim as optim

# 1. 序嵌入网络：将自然语言映射到非负偏序空间 R_+^d
class OrderEmbeddingNLI(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, poset_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 特征提取 Encoder (此处用 GRU 提取时序语义)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        # 投影至偏序空间，并通过 ReLU 强制落在非负象限 R_+^d
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, poset_dim),
            nn.ReLU()
        )

    def encode(self, x):
        embeds = self.embedding(x)
        _, h_n = self.encoder(embeds)
        # 映射到非负偏序锥中
        poset_vec = self.proj(h_n.squeeze(0))
        return poset_vec

    @staticmethod
    def energy(u, v):
        """
        计算离散偏序违背程度:
        若 u 蕴涵 v，则要求每个维度 u_k >= v_k，违背量 max(0, v - u) 应为 0
        """
        violation = torch.relu(v - u)
        return torch.sum(violation ** 2, dim=-1)

# 2. 偏序约束损失函数 (Max-Margin Order Loss)
class OrderLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, pos_u, pos_v, neg_u, neg_v):
        # 正样本对 (蕴涵): 违序能量趋向于 0
        loss_pos = OrderEmbeddingNLI.energy(pos_u, pos_v)
        # 负样本对 (不蕴涵/矛盾): 违序能量应至少大于预设间隔 margin
        loss_neg = torch.relu(self.margin - OrderEmbeddingNLI.energy(neg_u, neg_v))
        return (loss_pos + loss_neg).mean()



def run_order_embedding_toy(seed=17):
    """Run the original toy locally without overwriting SNLI vocab/model."""
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(seed)
        # 词表构建
        vocab = {"<pad>": 0, "a": 1, "small": 2, "terrier": 3, "dog": 4, 
                 "animal": 5, "barking": 6, "running": 7}

        # 构造蕴涵链样本: A -> B, B -> C
        # A: "a small terrier barking"
        # B: "a dog barking"
        # C: "an animal barking"
        # D: "a small terrier running" (非蕴涵样例；running 与 barking 可以同时成立)
        sent_A = torch.tensor([[1, 2, 3, 6]])
        sent_B = torch.tensor([[1, 4, 6, 0]])
        sent_C = torch.tensor([[1, 5, 6, 0]])
        sent_D = torch.tensor([[1, 2, 3, 7]])

        model = OrderEmbeddingNLI(vocab_size=len(vocab), embed_dim=16, hidden_dim=32, poset_dim=16)
        criterion = OrderLoss(margin=1.0)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        print("=== 开始注入偏序关系并训练 ===")
        # 仅向模型训练 A -> B 和 B -> C，绝不给模型喂 A -> C
        for epoch in range(150):
            optimizer.zero_grad()
            u_A = model.encode(sent_A)
            v_B = model.encode(sent_B)
            v_C = model.encode(sent_C)
            v_D = model.encode(sent_D)

            # 批次包含: 正样本 (A->B), (B->C); 负样本 (B->A 非对称), (A->D 非蕴涵)
            loss1 = criterion(u_A, v_B, v_B, u_A) # 正: A->B, 负: B->A
            loss2 = criterion(v_B, v_C, u_A, v_D) # 正: B->C, 负: A->D
            loss = loss1 + loss2
    
            loss.backward()
            optimizer.step()

        print(f"训练收敛 Loss: {loss.item():.4f}\n")

        # 4. 验证离散数学性质
        model.eval()
        with torch.no_grad():
            u_A = model.encode(sent_A)
            v_B = model.encode(sent_B)
            v_C = model.encode(sent_C)

            e_AB = OrderEmbeddingNLI.energy(u_A, v_B).item()
            e_BC = OrderEmbeddingNLI.energy(v_B, v_C).item()
            e_AC = OrderEmbeddingNLI.energy(u_A, v_C).item()  # 已见句子的未训练边推断
            e_BA = OrderEmbeddingNLI.energy(v_B, u_A).item()  # 指定逆向边检验

        print("=== 离散数学性质推断验证 ===")
        print(f"1. 前提 A 蕴涵 B 能量 (目标 ~0): {e_AB:.4f}")
        print(f"2. 前提 B 蕴涵 C 能量 (目标 ~0): {e_BC:.4f}")
        print(f"3. [传递性检验] A 蕴涵 C 能量 (未训练边，句子均已见): {e_AC:.4f}")
        print(f"4. [指定逆向边检验] B 蕴涵 A 能量 (目标 >= 1.0): {e_BA:.4f}")

        # 判定
        is_transitive = e_AB < 0.05 and e_BC < 0.05 and e_AC < 0.05
        is_asymmetric = e_BA > 0.8
        print(f"\n当前链 AB、BC、AC 是否均满足阈值（非全局证明）: {'通过' if is_transitive else '未通过'}")
        print(f"指定逆向边 B->A 是否被拒绝（非反对称律证明）: {'通过' if is_asymmetric else '未通过'}")
        return {"AB": e_AB, "BC": e_BC, "AC": e_AC, "BA": e_BA, "chain_threshold_check": is_transitive, "reverse_edge_rejected": is_asymmetric, "scope": "seen sentences, held-out edge only"}


toy_order_result = run_order_embedding_toy()
