"""Load a final checkpoint and inspect NLI predictions without retraining."""
from pathlib import Path
import re

import torch

from .data import load_data
from .model import NeuralOrderNLI


def predict_pairs(run_dir, pairs, arm='neural', seed=17, device='cpu'):
    run_dir = Path(run_dir)
    cache = load_data(run_dir)
    checkpoint = torch.load(run_dir / 'checkpoints' / f'{arm}_seed{seed}_final.pt',
                            map_location='cpu', weights_only=False)
    model = NeuralOrderNLI(cache['embeddings'], **checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.to(device).eval()
    vocab = {token: index for index, token in enumerate(cache['vocab'])}

    def encode(texts):
        lines = [re.sub(r'[()]', '', text).split()[:50] for text in texts]
        ids = [[vocab.get(token, 1) for token in line] or [1] for line in lines]
        lengths = torch.tensor([len(line) for line in ids])
        tokens = torch.tensor([line + [0] * (50 - len(line)) for line in ids], device=device)
        return tokens, lengths

    if not pairs:
        return []
    premises, premise_lengths = encode([pair[0] for pair in pairs])
    hypotheses, hypothesis_lengths = encode([pair[1] for pair in pairs])
    with torch.inference_mode():
        logits, energy, reverse = model(premises, hypotheses, premise_lengths, hypothesis_lengths)
        probabilities = logits.softmax(-1).cpu().tolist()
    names = ['entailment', 'contradiction', 'neutral']
    return [{'premise': p, 'hypothesis': h, 'prediction': names[max(range(3), key=lambda i: prob[i])],
             'probabilities': dict(zip(names, prob)), 'forward_order_energy': float(e),
             'reverse_order_energy': float(r), 'arm': arm, 'seed': seed}
            for (p, h), prob, e, r in zip(pairs, probabilities, energy.cpu(), reverse.cpu())]
