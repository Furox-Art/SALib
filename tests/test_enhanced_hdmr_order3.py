import numpy as np

from SALib.analyze.enhanced_hdmr import analyze


def test_enhanced_hdmr_max_order_three_returns_finite_totals():
    """max_order=3 used to crash while assembling total-order indices.

    nc3 is a count. The third-order term index array is gamma, matching
    how second-order totals use beta. See SALib/SALib#667.
    """
    rng = np.random.default_rng(0)
    dimension = 4
    sample_count = 300
    inputs = rng.random((sample_count, dimension))
    outputs = inputs[:, 0] + inputs[:, 1] * inputs[:, 2] + 0.1 * inputs[:, 3]
    problem = {
        "num_vars": dimension,
        "names": [f"x{i}" for i in range(dimension)],
        "bounds": [[0.0, 1.0]] * dimension,
    }

    result = analyze(
        problem,
        inputs,
        outputs,
        max_order=3,
        poly_order=2,
        bootstrap=1,
        seed=1,
    )

    totals = np.asarray(result["ST"][:dimension], dtype=float)
    assert totals.shape == (dimension,)
    assert np.isfinite(totals).all()
