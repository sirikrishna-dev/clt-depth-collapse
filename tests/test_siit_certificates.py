from cdc.siit import certify


def test_tiny_model_certifies(tiny_d2):
    model, data, _ = tiny_d2
    c = certify(model, data, n_rep=1)
    assert c["clean_bit_acc"] > 0.98
    assert c["iia_h1_cf_flip"] > 0.95
    assert c["strict_min"] > 0.95
    assert "bypass" in c and c["bypass"]["h1"]["n_informative_bits"] > 0
