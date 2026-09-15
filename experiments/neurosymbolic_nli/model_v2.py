"""Enhanced Neurosymbolic NLI Architecture (V2).

Provides:
1. Exact padding-aware sequence handling.
2. LayerNorm-stabilized non-negative cone projection.
3. Principled 3-way geometric relations:
   - Entailment: Cone Containment (low forward violation, asymmetric reverse margin)
   - Contradiction: Orthogonal Cones in non-negative orthant (low cosine overlap)
   - Neutral: Bounded Overlap (neither entails the other, non-orthogonal)
4. Low-rank geometric feature projection to eliminate parameter explosion / overfitting.
5. Support for both standard GRU and Attentive BiGRU encoders.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NeurosymbolicNLIv2(nn.Module):
    def __init__(self, embeddings, hidden_dim=128, order_dim=64, dropout=0.2,
                 encoder_type='gru', geo_bottleneck=8):
        super().__init__()
        self.encoder_type = encoder_type
        self.embedding = nn.Embedding.from_pretrained(
            embeddings.clone(), freeze=False, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        if encoder_type == 'gru':
            self.encoder = nn.GRU(embeddings.shape[1], hidden_dim, batch_first=True)
            enc_out_dim = hidden_dim
        elif encoder_type == 'bigru_attn':
            half_dim = hidden_dim // 2
            self.encoder = nn.GRU(embeddings.shape[1], half_dim, batch_first=True, bidirectional=True)
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
            enc_out_dim = hidden_dim
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Stabilized non-negative cone projection
        self.order_projection = nn.Sequential(
            nn.Linear(enc_out_dim, order_dim),
            nn.LayerNorm(order_dim),
            nn.Softplus()
        )

        # Compact geometric feature adapter (prevents classifier overfitting)
        self.geo_bottleneck = geo_bottleneck
        if geo_bottleneck > 0:
            self.geo_adapter = nn.Sequential(
                nn.Linear(4, geo_bottleneck),
                nn.Tanh()
            )
            cls_geo_dim = geo_bottleneck
        else:
            cls_geo_dim = 4

        self.classifier = nn.Sequential(
            nn.Linear(enc_out_dim * 4 + cls_geo_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )

    def encode(self, tokens, lengths):
        emb = self.dropout(self.embedding(tokens))
        cpu_lengths = lengths.detach().cpu()
        packed = pack_padded_sequence(emb, cpu_lengths, batch_first=True, enforce_sorted=False)

        if self.encoder_type == 'gru':
            _, hidden = self.encoder(packed)
            rep = hidden[-1]
        elif self.encoder_type == 'bigru_attn':
            out, _ = self.encoder(packed)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=tokens.shape[1])
            # Masked attention pooling
            scores = self.attn(out).squeeze(-1)
            mask = torch.arange(tokens.shape[1], device=tokens.device)[None, :] >= lengths.to(tokens.device)[:, None]
            scores = scores.masked_fill(mask, -1e9)
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
            rep = (out * weights).sum(dim=1)

        order = self.order_projection(rep)
        return rep, order

    @staticmethod
    def compute_geometry(u, v):
        """Computes 4 invariant geometric relations between non-negative vectors u and v:
        - fwd_energy: ||ReLU(v - u)||^2 / d  (0 if u >= v, i.e. u contains v)
        - rev_energy: ||ReLU(u - v)||^2 / d  (0 if v >= u)
        - cos_sim: (u . v) / (||u|| ||v||)   (0 if orthogonal / disjoint)
        - jaccard: sum(min(u,v)) / sum(max(u,v)) (intersection / union)
        """
        fwd_energy = torch.relu(v - u).square().mean(dim=-1, keepdim=True)
        rev_energy = torch.relu(u - v).square().mean(dim=-1, keepdim=True)
        
        u_norm = torch.norm(u, dim=-1, keepdim=True) + 1e-7
        v_norm = torch.norm(v, dim=-1, keepdim=True) + 1e-7
        cos_sim = (u * v).sum(dim=-1, keepdim=True) / (u_norm * v_norm)

        inter = torch.min(u, v).sum(dim=-1, keepdim=True)
        union = torch.max(u, v).sum(dim=-1, keepdim=True) + 1e-7
        jaccard = inter / union

        return torch.cat([fwd_energy, rev_energy, cos_sim, jaccard], dim=-1)

    def forward(self, premises, hypotheses, premise_lengths, hypothesis_lengths):
        n = premises.shape[0]
        hidden, order = self.encode(
            torch.cat((premises, hypotheses)),
            torch.cat((premise_lengths, hypothesis_lengths)))
        p, h = hidden[:n], hidden[n:]
        u, v = order[:n], order[n:]

        geo_metrics = self.compute_geometry(u, v)
        fwd_energy = geo_metrics[:, 0]
        rev_energy = geo_metrics[:, 1]
        cos_sim = geo_metrics[:, 2]
        jaccard = geo_metrics[:, 3]

        if self.geo_bottleneck > 0:
            geo_features = self.geo_adapter(geo_metrics)
        else:
            geo_features = geo_metrics

        neural_features = torch.cat((p, h, (p - h).abs(), p * h), dim=-1)
        features = torch.cat((neural_features, geo_features), dim=-1)
        logits = self.classifier(features)

        return logits, fwd_energy, rev_energy, cos_sim, jaccard


def v1_order_penalty(fwd_energy, labels, margin=0.2):
    """V1 penalty: Entailment vs non-entailment binary margin."""
    return torch.where(labels == 0, fwd_energy, torch.relu(margin - fwd_energy)).mean()


def v2_threeway_loss(fwd_energy, rev_energy, cos_sim, jaccard, labels,
                     margin_fwd=0.15, margin_rev=0.15):
    """Principled 3-way geometric objective:
    - Entailment (y=0): Cone containment (fwd_energy -> 0) + mild asymmetry (rev_energy >= margin_rev)
    - Contradiction (y=1): Orthogonal disjointness (cos_sim -> 0)
    - Neutral (y=2): Neither entails the other, safe margin from both containment cones
    """
    loss_ent = torch.where(labels == 0,
                           fwd_energy + 0.2 * torch.relu(margin_rev - rev_energy),
                           torch.zeros_like(fwd_energy))
    loss_contra = torch.where(labels == 1,
                              torch.relu(cos_sim - 0.05),
                              torch.zeros_like(cos_sim))
    loss_neut = torch.where(labels == 2,
                            0.5 * torch.relu(margin_fwd - fwd_energy) + 0.5 * torch.relu(margin_rev - rev_energy),
                            torch.zeros_like(fwd_energy))
    return (loss_ent + loss_contra + loss_neut).mean()
