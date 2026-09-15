"""Padding-aware extension of the notebook's GRU order embedding."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class NeuralOrderNLI(nn.Module):
    def __init__(self, embeddings, hidden_dim=128, order_dim=64, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embeddings.clone(), freeze=False, padding_idx=0)
        self.encoder = nn.GRU(embeddings.shape[1], hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.order_projection = nn.Sequential(
            nn.Linear(hidden_dim, order_dim), nn.Softplus())
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 2, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 3))

    def encode(self, tokens, lengths):
        embeddings = self.dropout(self.embedding(tokens))
        packed = pack_padded_sequence(
            embeddings, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        hidden = hidden[-1]
        return hidden, self.order_projection(hidden)

    @staticmethod
    def energy(premise, hypothesis):
        # Mean, rather than sum, makes the margin independent of order_dim.
        return torch.relu(hypothesis - premise).square().mean(dim=-1)

    def forward(self, premises, hypotheses, premise_lengths, hypothesis_lengths):
        n = premises.shape[0]
        hidden, order = self.encode(
            torch.cat((premises, hypotheses)),
            torch.cat((premise_lengths, hypothesis_lengths)))
        p, h = hidden[:n], hidden[n:]
        u, v = order[:n], order[n:]
        energy, reverse_energy = self.energy(u, v), self.energy(v, u)
        features = torch.cat((p, h, (p - h).abs(), p * h,
                              energy[:, None], reverse_energy[:, None]), dim=-1)
        return self.classifier(features), energy, reverse_energy


def order_penalty(energy, labels, margin=0.2):
    # Neutral remains a separate class in cross-entropy supervision.
    return torch.where(labels == 0, energy, torch.relu(margin - energy)).mean()
