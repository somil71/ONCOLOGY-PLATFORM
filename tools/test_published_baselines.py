"""Source-parity and hand-computed survival checks for the author-model adapter."""

import sys

import numpy as np
import torch

from src.published_baselines import AuthorMultiSurvRNA, load_official_multisurv_classes


def main():
    torch.set_num_threads(2)
    before = {
        name: sys.modules.get(name)
        for name in ("attention", "embrace_net", "sub_models", "multisurv", "loss")
    }
    author_model, _ = load_official_multisurv_classes()
    assert all(sys.modules.get(name) is value for name, value in before.items())
    torch.manual_seed(42)
    official = author_model(
        ["mRNA"], fusion_method=None, n_output_intervals=30, device="cpu"
    )
    adapted = AuthorMultiSurvRNA()
    adapted.author_model.load_state_dict(official.state_dict())
    x = torch.randn(4, 1000)
    official.eval()
    adapted.eval()
    with torch.no_grad():
        np.testing.assert_array_equal(
            official({"mRNA": x})[1].numpy(), adapted(x).numpy()
        )
    # Two interval labels: event15 -> S1*(1-S2); censor15 receives
    # midpoint-credit for both intervals; censor4 receives no survival credit.
    tiny = AuthorMultiSurvRNA([0, 10, 20])
    probability = torch.tensor([[0.8, 0.7]] * 4, requires_grad=True)
    loss = tiny.loss_from_days(probability, [15, 15, 4, 10], [1, 0, 0, 1])
    # Upstream reduction divides the summed NLL by patients*2*intervals.
    expected = (
        -torch.log(torch.tensor([0.8 * 0.3, 0.8 * 0.7, 1.0, 0.8 * 0.3])).sum() / 16
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(probability.grad).all()
    survival = adapted.predict_survival(x, [0, 365, 1095, 1825])
    assert (survival[:, 0] == 1).all() and (np.diff(survival, axis=1) <= 0).all()
    for wrong in (torch.randn(3, 256), torch.full((3, 1000), float("nan"))):
        try:
            adapted(wrong)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid author model input accepted")
    try:
        tiny.loss_from_days(probability[:1], [20], [1])
    except ValueError:
        pass
    else:
        raise AssertionError("Death without an upstream event interval accepted")
    print(
        "PASS: pinned author-source parity, import isolation, exact midpoint-censor/boundary loss arithmetic, gradients, survival monotonicity and invalid-input/tail rejection; synthetic fixtures only."
    )


if __name__ == "__main__":
    main()
