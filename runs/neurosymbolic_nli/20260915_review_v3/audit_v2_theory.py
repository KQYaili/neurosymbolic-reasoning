"""Read-only checkpoint inspection and exact small counterexamples; no training."""
import hashlib
import itertools
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.neurosymbolic_nli.model_v2 import NeurosymbolicNLIv2, v2_threeway_loss


def main():
    run = ROOT / 'runs/neurosymbolic_nli/20260915_order_v2'
    checkpoints = []
    for path in sorted((run / 'checkpoints').glob('*_final.pt')):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        gamma = checkpoint['model_state']['order_projection.1.weight']
        checkpoints.append({
            'checkpoint': path.name,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'gamma_min': float(gamma.min()),
            'gamma_max': float(gamma.max()),
            'all_gamma_positive': bool((gamma > 0).all()),
        })
    assert len(checkpoints) == 15
    assert all(row['all_gamma_positive'] for row in checkpoints)

    # Six distinct normalized vectors all have sum zero. Positive shared affine
    # scales and strictly monotone Softplus preserve the absence of domination.
    z = torch.tensor(list(itertools.permutations([-1., 0., 1.])), dtype=torch.float64)
    u = torch.nn.functional.softplus(z * torch.tensor([.6, 1., 1.3]))
    dominated = (u[:, None, :] >= u[None, :, :]).all(-1)
    assert torch.equal(dominated, torch.eye(len(z), dtype=torch.bool))

    equivalent = NeurosymbolicNLIv2.compute_geometry(torch.ones(1, 3), torch.ones(1, 3))
    eq_loss = v2_threeway_loss(*equivalent.unbind(-1), torch.tensor([0]))
    assert eq_loss > 0

    # A broad premise and a strictly stronger hypothesis can be labelled neutral
    # forward while the reverse is entailment; here the reverse energy is zero.
    neutral = NeurosymbolicNLIv2.compute_geometry(torch.ones(1, 3), torch.full((1, 3), 2.))
    neutral_loss = v2_threeway_loss(*neutral.unbind(-1), torch.tensor([2]))
    assert neutral[0, 1] == 0 and neutral_loss > 0

    apex_a = torch.tensor([1., 0.])
    apex_b = torch.tensor([0., 1.])
    witness = torch.maximum(apex_a, apex_b)
    assert torch.dot(apex_a, apex_b) == 0
    assert (witness >= apex_a).all() and (witness >= apex_b).all()

    record = {
        'all_checks_passed': True,
        'scope': 'Counterexamples expose assumptions; they do not validate V2 natural-language accuracy.',
        'checkpoint_gamma': checkpoints,
        'checks': {
            'all_15_final_checkpoints_have_positive_layernorm_gamma': True,
            'six_distinct_zero_sum_vectors_form_antichain_after_positive_affine_softplus': True,
            'equivalent_entailment_receives_positive_auxiliary_penalty': True,
            'neutral_with_reverse_entailment_receives_positive_auxiliary_penalty': True,
            'orthogonal_apices_have_nonempty_upper_cone_intersection': True,
        },
        'equivalent_entailment_auxiliary_penalty': float(eq_loss),
        'neutral_with_reverse_entailment_auxiliary_penalty': float(neutral_loss),
        'orthogonal_apices_intersection_witness': witness.tolist(),
        'proof': 'Let z and z_prime be pre-affine LayerNorm outputs, each summing to zero. '
                 'With shared gamma_i > 0 and beta_i, Softplus(gamma*z+beta) >= '
                 'Softplus(gamma*z_prime+beta) implies z >= z_prime by strict monotonicity. '
                 'Equal sums force coordinate equality. Thus exact domination is equality '
                 'in real arithmetic. This argument does not assert a finite-precision '
                 'energy threshold is transitive.',
    }
    output = Path(__file__).with_name('v2_theory_audit.json')
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'all_checks_passed': True, 'checkpoints': len(checkpoints), 'output': str(output)}))


if __name__ == '__main__':
    main()
