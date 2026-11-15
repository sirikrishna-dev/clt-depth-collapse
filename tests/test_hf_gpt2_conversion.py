"""The GPT-2 weight conversion must reproduce Hugging Face's own forward pass exactly;
otherwise every real-data number is computed on the wrong model. Random weights, no download."""
import pytest
import torch

transformers = pytest.importorskip("transformers")


def test_generic_engine_matches_hf_gpt2():
    from cdc.realdata.generic_tf import from_hf_gpt2_model
    from cdc.realdata.lm_twohop import HFMLPIO
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=2, n_embd=64, n_head=4, vocab_size=101, n_positions=32,
                                  resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    hf = transformers.GPT2LMHeadModel(cfg).eval()
    gm = from_hf_gpt2_model(hf)
    toks = torch.randint(0, 101, (12,))
    with torch.no_grad():
        ref = hf(toks[None]).logits[0]
    logits, c = gm.run(toks)
    assert torch.allclose(ref, logits, atol=1e-4), float((ref - logits).abs().max())
    ins, outs = HFMLPIO(hf)(toks[None])
    for l in range(2):
        assert torch.allclose(ins[l], c["mlp_in"][l], atol=1e-4)
        assert torch.allclose(outs[l], c["mlp_out"][l], atol=1e-4)
