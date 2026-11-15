import numpy as np

from cdc.tasks import make_dataset, high_level, counterfactual_pairs, op_name


def test_equations_final_xor():
    d2 = make_dataset(2)
    a, b, c = d2.x[:, 0], d2.x[:, 1], d2.x[:, 2]
    assert np.array_equal(d2.H[:, 0], a & b)
    assert np.array_equal(d2.y, (a & b) ^ c)
    for D in range(1, 6):
        assert op_name(D, D) == "XOR"          # last hop is never linearly separable
    d3 = make_dataset(3)
    a, b, c, d = (d3.x[:, i] for i in range(4))
    assert np.array_equal(d3.y, (((a ^ b) & c) ^ d))


def test_every_bit_every_depth():
    for D in range(1, 6):
        ds = make_dataset(D)
        h = (ds.x[:, 0] & ds.x[:, 1]) if op_name(1, D) == "AND" else (ds.x[:, 0] ^ ds.x[:, 1])
        assert np.array_equal(ds.H[:, 0], h)
        for k in range(2, D + 1):
            h = (h & ds.x[:, k]) if op_name(k, D) == "AND" else (h ^ ds.x[:, k])
            assert np.array_equal(ds.H[:, k - 1], h)
        assert len(ds.x) == 7000 and (ds.split == 0).sum() == 5000 and (ds.split == 2).sum() == 1000


def test_splits_disjoint_and_deterministic():
    a, b = make_dataset(3), make_dataset(3)
    assert np.array_equal(a.x, b.x)


def test_confound_free_pairs_keep_side_inputs():
    ds = make_dataset(3)
    rng = np.random.default_rng(0)
    for hop in (1, 2):
        xb, xs, y_star, y_b = counterfactual_pairs(ds.x[:500], 3, hop, rng, confound_free=True)
        assert np.array_equal(xb[:, hop + 1:], xs[:, hop + 1:])
        # y_star must equal the HL model with h_hop from the source
        H_src = high_level(xs, 3)
        assert np.array_equal(y_star, high_level(xb, 3, {hop: H_src[:, hop - 1]})[:, -1])
