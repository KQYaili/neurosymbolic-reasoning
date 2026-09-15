"""Contextual attend/compare/pool NLI with a sound, partial reverse-label loss.

Inspired by Parikh et al. (2016), https://aclanthology.org/D16-1244/.
These are probabilistic NLI models, not theorem provers or causal estimators.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AlignedNLI(nn.Module):
    def __init__(self, embeddings, hidden_dim=128, dropout=0.2):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError('hidden_dim must be even')
        self.embedding = nn.Embedding.from_pretrained(embeddings.clone(), freeze=False, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.GRU(embeddings.shape[1], hidden_dim // 2,
                              bidirectional=True, batch_first=True)
        self.attend = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.compare = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.ReLU(),
                                     nn.Dropout(dropout))
        # Mean and max pooling in each direction; the same classifier handles swaps.
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 8, hidden_dim), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hidden_dim, 3))

    def encode(self, tokens, lengths):
        if tokens.ndim != 2 or lengths.ndim != 1 or len(tokens) != len(lengths):
            raise ValueError('Expected tokens [batch,time] and lengths [batch]')
        if bool((lengths < 1).any()) or bool((lengths > tokens.shape[1]).any()):
            raise ValueError('Sequence lengths must be positive and within token width')
        packed = pack_padded_sequence(self.dropout(self.embedding(tokens)),
                                      lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=tokens.shape[1])
        mask = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < lengths.to(tokens.device)[:, None]
        return encoded, mask

    @staticmethod
    def pool(values, mask):
        mean = (values * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        maximum = values.masked_fill(~mask[..., None], torch.finfo(values.dtype).min).amax(1)
        return torch.cat((mean, maximum), -1)

    def forward(self, premises, hypotheses, premise_lengths, hypothesis_lengths,
                alignment='learned', return_attention=False):
        if alignment not in ('learned', 'uniform'):
            raise ValueError('Unknown alignment mode')
        # Separate widths are supported, including nonzero junk beyond valid lengths.
        p, pm = self.encode(premises, premise_lengths)
        h, hm = self.encode(hypotheses, hypothesis_lengths)
        fp, fh = self.attend(p), self.attend(h)
        scores = torch.bmm(fp, fh.transpose(1, 2)) / fp.shape[-1] ** 0.5
        if alignment == 'uniform':
            scores = scores * 0  # Same parameters and RNG path; ablate learned word matching.
        p_to_h = scores.masked_fill(~hm[:, None, :], torch.finfo(scores.dtype).min).softmax(-1)
        h_to_p = scores.transpose(1, 2).masked_fill(~pm[:, None, :], torch.finfo(scores.dtype).min).softmax(-1)
        hp = torch.bmm(p_to_h, h)
        ph = torch.bmm(h_to_p, p)
        cp = self.compare(torch.cat((p, hp, p - hp, p * hp), -1))
        ch = self.compare(torch.cat((h, ph, h - ph, h * ph), -1))
        rp, rh = self.pool(cp, pm), self.pool(ch, hm)
        forward = torch.cat((rp, rh, (rp-rh).abs(), rp*rh), -1)
        reverse = torch.cat((rh, rp, (rp-rh).abs(), rp*rh), -1)
        # Always compute both directions, even for CE-only arms: identical dropout draw count.
        logits, reverse_logits = self.classifier(torch.cat((forward, reverse))).chunk(2)
        if return_attention:
            return logits, reverse_logits, p_to_h, h_to_p
        return logits, reverse_logits


def reverse_partial_loss(reverse_logits, labels):
    """C(P,H) => C(H,P); E/N(P,H) => E or N(H,P).

    This follows from symmetric incompatibility for satisfiable propositions in
    a shared context. It does NOT assert that neutral is symmetric, that
    entailment is strictly asymmetric, or that reverse E/N labels are known.
    Stable -log of the probability mass assigned to the allowed reverse set.
    """
    logp = reverse_logits.log_softmax(-1)
    non_contradiction = torch.logsumexp(logp[:, [0, 2]], dim=-1)
    return -torch.where(labels == 1, logp[:, 1], non_contradiction).mean()


def contradiction_consistency(logits, reverse_logits):
    """Soft probability consistency; no promise of exact logical satisfaction."""
    return (logits.softmax(-1)[:, 1] - reverse_logits.softmax(-1)[:, 1]).square().mean()
