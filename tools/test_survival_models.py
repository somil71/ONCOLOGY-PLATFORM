"""Meaningful regressions for research survival models; synthetic fixtures only."""

import numpy as np
import torch
import pandas as pd
from src.survival_models import (
    RegularizedCox,
    HazardFusion,
    discrete_hazard_nll,
    predict_cox_bundle,
)
from tools.run_survival_experiments import fit_transform, uncertainty_if_supported
from src.survival_transport import rank_expression, shared_gene_symbols


def main():
    torch.set_num_threads(2)
    # Explicit tied-time Breslow reference with one predictor.
    x = np.array([[0.0], [1.0], [-1.0], [2.0], [0.5], [1.5]])
    time = np.array([5.0, 5.0, 3.0, 2.0, 7.0, 9.0])
    event = np.array([1, 0, 1, 1, 0, 1])
    m = RegularizedCox(0.1).fit(x, time, event)
    perm = np.array([5, 2, 0, 4, 1, 3])
    mp = RegularizedCox(0.1).fit(x[perm], time[perm], event[perm])
    np.testing.assert_allclose(m.coef_, mp.coef_, atol=1e-7)
    bundle = {
        "columns": ["x"],
        "preprocessing": {"median": [0.0], "mean": [0.0], "scale": [1.0]},
        "coefficients": m.coef_.tolist(),
        "penalty": 0.1,
        "event_times": m.event_times_.tolist(),
        "baseline_cumulative_hazard": m.baseline_cumulative_hazard_.tolist(),
    }
    br, bs = predict_cox_bundle(bundle, pd.DataFrame({"x": x[:, 0]}), [1, 3, 5, 10])
    bad_bundle = {**bundle, "event_times": [float("nan")] * len(bundle["event_times"])}
    try:
        predict_cox_bundle(bad_bundle, pd.DataFrame({"x": x[:, 0]}), [1, 3])
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite frozen hazard grid accepted")
    np.testing.assert_allclose(br, m.predict_risk(x))
    np.testing.assert_allclose(bs, m.predict_survival(x, [1, 3, 5, 10]))
    s = m.predict_survival(x, [1, 3, 5, 10])
    assert ((s >= 0) & (s <= 1)).all() and (np.diff(s, axis=1) <= 0).all()
    try:
        RegularizedCox().fit(x, time, np.zeros_like(event))
    except ValueError:
        pass
    else:
        raise AssertionError("Zero-event Cox fitting accepted")
    # Hazards .2,.3,.4: event in second bin -> .8*.3; censor at
    # first endpoint -> .8; censor before first end -> contribution 1.
    logits = torch.logit(torch.tensor([[0.2, 0.3, 0.4]] * 3, requires_grad=True))
    logits.retain_grad()
    losses = discrete_hazard_nll(
        logits,
        torch.tensor([15.0, 10.0, 5.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        [10.0, 20.0],
        reduction="none",
    )
    torch.testing.assert_close(losses, -torch.log(torch.tensor([0.8 * 0.3, 0.8, 1.0])))
    losses.mean().backward()
    assert torch.isfinite(logits.grad).all()
    # An event exactly at boundary belongs to its ending bin.
    boundary_loss = discrete_hazard_nll(
        logits[:1], torch.tensor([10.0]), torch.tensor([1.0]), [10.0, 20.0]
    )
    torch.testing.assert_close(boundary_loss, -torch.log(torch.tensor(0.2)))
    torch.manual_seed(42)
    for fusion in ["attention", "mean", "concat"]:
        model = HazardFusion({"cli": 2, "rna": 3}, [10.0, 20.0], fusion=fusion).eval()
        features = {"cli": torch.randn(3, 2), "rna": torch.randn(3, 3)}
        mask = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.bool)
        first = model(features, mask)
        changed = {k: v.clone() for k, v in features.items()}
        changed["rna"][0] = 1e8
        changed["cli"][2] = -1e8
        torch.testing.assert_close(first, model(changed, mask))
        survival = model.predict_survival(features, mask, [5, 10, 20, 25])
        assert (np.diff(survival, axis=1) <= 1e-7).all()
        try:
            model(features, torch.zeros_like(mask))
        except ValueError:
            pass
        else:
            raise AssertionError("All-missing input accepted")
    train = np.array([[10.0, np.nan], [20.0, 2.0], [30.0, 4.0]])
    _, contract = fit_transform(train, np.array([[999999.0, 20.0]]))
    assert contract["median"] == [20.0, 3.0]
    assert contract["mean"] == [20.0, 3.0], (
        "Held-out outlier affected training preprocessing"
    )
    symbols = ["B", "A", "A", "C"]
    values = np.array([[2.0, 0.0, 2.0, 4.0], [1.0, np.nan, np.nan, 3.0]])
    contract = shared_gene_symbols(symbols, ["C", "B", "A", "D"])
    assert contract == ["A", "B", "C"]
    rank = rank_expression(values, symbols, contract)
    np.testing.assert_allclose(rank[0], [1 / 6, 1 / 2, 5 / 6])
    np.testing.assert_array_equal(
        rank, rank_expression(values * 5 + 10, symbols, contract)
    )
    np.testing.assert_array_equal(
        rank,
        rank_expression(
            values[:, [3, 1, 0, 2]], np.array(symbols)[[3, 1, 0, 2]], contract
        ),
    )
    unsupported = uncertainty_if_supported(
        np.array([100.0, 200.0]),
        np.array([0, 0]),
        np.array([0.9, 0.8]),
        np.array([0.1, 0.2]),
        np.array([100.0, 200.0]),
        np.array([0, 0]),
        50.0,
    )
    assert unsupported["status"] == "not_evaluated" and "reason" in unsupported
    guarded = HazardFusion({"cli": 2}, [10.0, 20.0])
    clean_features = {"cli": torch.ones(2, 2)}

    def rejected(call):
        try:
            call()
        except ValueError:
            return
        raise AssertionError("Invalid research input accepted")

    for mask in (
        torch.full((2, 1), 0.5),
        torch.full((2, 1), float("nan")),
        torch.ones(2, 2),
    ):
        rejected(lambda mask=mask: guarded(clean_features, mask))
    rejected(
        lambda: guarded({"cli": torch.full((2, 2), float("nan"))}, torch.ones(2, 1))
    )
    rejected(lambda: guarded({"cli": torch.ones(2, 3)}, torch.ones(2, 1)))
    rejected(lambda: guarded({"wrong": torch.ones(2, 2)}, torch.ones(2, 1)))
    rejected(lambda: HazardFusion({"cli": 2}, [10.0, 20.0], hidden=10))
    rejected(
        lambda: discrete_hazard_nll(
            torch.ones(2, 3), torch.ones(3), torch.ones(2), [10.0, 20.0]
        )
    )
    rejected(
        lambda: discrete_hazard_nll(
            torch.full((2, 3), float("nan")), torch.ones(2), torch.ones(2), [10.0, 20.0]
        )
    )
    rejected(
        lambda: discrete_hazard_nll(
            torch.ones(2, 3), torch.ones(2, 1), torch.ones(2), [10.0, 20.0]
        )
    )
    print(
        "PASS: tied Cox permutation, censor likelihood analytic probabilities, boundary convention, finite gradients, missing-input invariance, allmissing rejection and train-only preprocessing."
    )


if __name__ == "__main__":
    main()
