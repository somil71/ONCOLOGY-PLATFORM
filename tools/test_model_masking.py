"""Functional masking checks for real model code; fixtures are synthetic, not results."""

import torch
from src.architecture import (
    GatedAttentionMIL,
    VolumetricSliceTransformer,
    MultiModalCoAttentionTransformer,
    CancerFusionResearchPlatform,
    modality_orthogonality_loss,
)


def main():
    torch.manual_seed(31)
    torch.set_num_threads(2)
    for encoder, dimension in (
        (GatedAttentionMIL(), 768),
        (VolumetricSliceTransformer(max_slices=5), 512),
    ):
        encoder.eval()
        features = torch.randn(2, 5, dimension)
        valid = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
        changed = features.clone()
        changed[~valid] = float("nan")
        with torch.no_grad():
            output = encoder(features, valid)
            modified = encoder(changed, valid)
        for a, b in zip(output, modified):
            torch.testing.assert_close(a, b)
        assert output[-1][~valid].eq(0).all()
        assert output[0][1].eq(0).all()
        assert torch.isfinite(output[0]).all()
    fusion = MultiModalCoAttentionTransformer().eval()
    tokens = torch.randn(2, 4, 64)
    mask = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 1.0]])
    with torch.no_grad():
        baseline = fusion(tokens, mask)
        changed = tokens.masked_fill(~mask.bool().unsqueeze(-1), float("nan"))
        modified = fusion(changed, mask)
    for a, b in zip(baseline, modified):
        torch.testing.assert_close(a, b)
    assert baseline[1][~mask.bool()].eq(0).all()
    torch.testing.assert_close(baseline[1].sum(1), torch.ones(2))
    for invalid in (torch.zeros(2, 4), torch.ones(2, 4) * 0.5):
        try:
            fusion(tokens, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid availability was accepted")
    single = torch.tensor([[0, 0, 0, 1.0]])
    assert modality_orthogonality_loss(tokens[:1], single).item() == 0
    all_present = torch.ones(2, 4)
    loss = modality_orthogonality_loss(tokens, all_present)
    assert torch.isfinite(loss)
    model = CancerFusionResearchPlatform(rna_dim=3, cli_dim=2, max_slices=5).eval()
    inputs = [
        torch.randn(2, 5, 768),
        torch.randn(2, 5, 512),
        torch.randn(2, 3),
        torch.randn(2, 2),
    ]
    with torch.no_grad():
        a = model(*inputs, mask)
        modified = [
            x.masked_fill(
                ~mask[:, i].bool().reshape((-1,) + (1,) * (x.ndim - 1)), float("nan")
            )
            for i, x in enumerate(inputs)
        ]
        b = model(*modified, mask)
    for i in (0, 1, 2, 4):
        torch.testing.assert_close(a[i], b[i])
    state = model.state_dict()
    clone = CancerFusionResearchPlatform(rna_dim=3, cli_dim=2, max_slices=5)
    clone.load_state_dict(state, strict=True)
    print(
        "PASS: padded-item invariance, empty-bag safety, missing-branch invariance, binary/all-missing rejection, mask-aware regularization, strict checkpoint shape compatibility."
    )


if __name__ == "__main__":
    main()
