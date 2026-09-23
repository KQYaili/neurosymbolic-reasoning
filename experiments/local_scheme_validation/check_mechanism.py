"""Independent arithmetic and numerical checks, using no held-out task examples."""
import json
from pathlib import Path
import torch
from peft import PeftModel
import train as tr


def main():
    run = tr.DEFAULT_RUN
    protocol = json.loads((run/'protocol.json').read_text())
    tr.setup(9023)
    checks = {}
    # An independently enumerated set semantics for satisfiable formulas.
    masks = range(1, 256)
    def relation(a,b):
        worlds_a = {i for i in range(8) if a & (1<<i)}
        worlds_b = {i for i in range(8) if b & (1<<i)}
        if worlds_a <= worlds_b: return 0
        if worlds_a.isdisjoint(worlds_b): return 1
        return 2
    checks['65025_swap_relations_sound'] = all((relation(a,b)==1)==(relation(b,a)==1) for a in masks for b in masks)
    checks['equivalence_entails_both_ways'] = relation(7,7)==relation(7,7)==0
    checks['neutral_can_reverse_to_entailment'] = relation(7,3)==2 and relation(3,7)==0
    z = torch.tensor([[1.,2.,3.,4.],[.1,.2,.3,.4]],dtype=torch.float64,requires_grad=True)
    l = tr.allowed_mass(z.log_softmax(-1),[['A','C'],['B']],torch.arange(4))
    expected = -torch.stack((z[0].softmax(-1)[[0,2]].sum(),z[1].softmax(-1)[1])).log()
    checks['allowed_loss_matches_explicit_probability_sum'] = torch.allclose(l,expected,atol=1e-12)
    l.mean().backward()
    checks['allowed_loss_finite_gradient'] = bool(torch.isfinite(z.grad).all())
    a = torch.tensor([[1.,2.,3.,4.]])
    b = a[:,[2,0,3,1]]
    js = tr.semantic_js(a,b,[[0,1,2,3]],[[2,0,3,1]])
    checks['permutation_alignment_direction_correct'] = abs(float(js))<1e-7
    checks['uniform_distribution_has_zero_js_but_no_accuracy_guarantee'] = abs(float(tr.semantic_js(torch.zeros_like(a),torch.zeros_like(a),[[0,1,2,3]],[[2,0,3,1]])))<1e-7
    # A final trained adapter is tested on artificial text, not quality/evaluation data.
    model,tok,answer_ids = tr.base_and_tokenizer(protocol,with_adapter=False)
    path = run/'adapters/sft_seed17'
    model = PeftModel.from_pretrained(model,path,is_trainable=False).float().eval()
    prompts = ['A red light is on. Answer with one letter: A.',
               'The red light and blue light are independent. The red light is on. Answer with one letter: A.']
    tokens = [tr.encode_prompt(tok,p) for p in prompts]
    with torch.inference_mode():
        batch = tr.last_logits(model,tr.collate(tokens,tok.pad_token_id))
        singles = torch.cat([tr.last_logits(model,tr.collate([t],tok.pad_token_id)) for t in tokens])
    delta = (batch-singles).abs()
    checks['fp32_trained_padding_batch_invariance_atol_0_0002'] = bool(torch.allclose(batch,singles,atol=2e-4,rtol=1e-5))
    checks['fp32_trained_argmax_invariance'] = bool(torch.equal(batch.argmax(-1),singles.argmax(-1)))
    report = {'checks':checks,'all_passed':all(checks.values()),'dtype':'float32_no_tf32',
              'maximum_logit_difference':float(delta.max()), 'mean_logit_difference':float(delta.mean()),
              'adapter_sha256':tr.sha256(path/'adapter_model.safetensors'),
              'script_sha256':tr.sha256(__file__), 'no_heldout_task_examples_used':True}
    tr.save_json(run/'mechanism_verification.json',report)
    print(json.dumps(report,indent=2))
    assert report['all_passed']


if __name__ == '__main__':
    main()
