"""Regressions for tied survival times and objective parity across entry points."""

import ast
import json
from pathlib import Path

import torch
import pandas as pd

from src.clinical_contract import select_reviewed_clinical_features

from src.architecture import continuous_cox_loss, evidence_strength_penalty, evidential_kl_regularization


def main():
    root = Path(__file__).resolve().parents[1]
    clinical = pd.DataFrame({"age": [50.0], "last_contact_days_to": [1200.0]})
    assert list(select_reviewed_clinical_features(clinical, ["age"]).columns) == ["age"]
    for invalid in ([], ["last_contact_days_to"], ["age", "age"], ["missing"]):
        try:
            select_reviewed_clinical_features(clinical, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe clinical contract accepted: {invalid}")
    times = torch.tensor([5.0, 5.0, 3.0, 2.0])
    events = torch.tensor([1.0, 0.0, 1.0, 0.0])
    risks = torch.tensor([0.2, 0.3, 0.4, 0.1], requires_grad=True)
    # Both patients at day 5 share the 0.5 denominator. At day 3 it is 0.9.
    expected = -(torch.log(torch.tensor(0.2 / 0.5)) + torch.log(torch.tensor(0.4 / 0.9))) / 2
    loss = continuous_cox_loss(risks, times, events)
    torch.testing.assert_close(loss, expected)
    permutation = torch.tensor([1, 0, 3, 2])
    torch.testing.assert_close(loss, continuous_cox_loss(risks[permutation], times[permutation], events[permutation]))
    loss.backward()
    assert torch.isfinite(risks.grad).all()
    assert continuous_cox_loss(risks, times, torch.zeros_like(events)).item() == 0

    alpha = torch.tensor([[1.0, 1.0], [3.0, 5.0]], requires_grad=True)
    expected_penalty = torch.tensor((0.0 + 6.0 / 8.0) / 2)
    torch.testing.assert_close(evidence_strength_penalty(alpha), expected_penalty)
    torch.testing.assert_close(evidential_kl_regularization(alpha, events), expected_penalty)

    notebook = root / "CancerFusion_Phase2_PanCancer.ipynb"
    assert notebook.read_bytes() == (root / "notebooks" / notebook.name).read_bytes()
    document = json.loads(notebook.read_bytes())
    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] == "code":
            code = "\n".join(line for line in "".join(cell["source"]).splitlines() if not line.startswith("!"))
            compile(code, f"cell {index}", "exec")
    guard_source = ast.parse((root / "src" / "clinical_contract.py").read_text(encoding="utf-8"))
    guard_notebook = ast.parse("".join(document["cells"][6]["source"]))
    def guard(module):
        return next(ast.dump(node, include_attributes=False) for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == "select_reviewed_clinical_features")
    assert guard(guard_source) == guard(guard_notebook)
    assert "".join(document["cells"][5]["source"]) == (root / "src" / "architecture.py").read_text(encoding="utf-8-sig")
    imaging = (root / "src" / "imaging_manifest.py").read_text(encoding="utf-8-sig")
    assert "".join(document["cells"][4]["source"]).startswith(imaging)
    source_ast = ast.parse((root / "src" / "architecture.py").read_text(encoding="utf-8"))
    notebook_ast = ast.parse("".join(document["cells"][5]["source"]))
    names = {"continuous_cox_loss", "evidence_strength_penalty", "evidential_kl_regularization"}
    def objectives(module):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names}
    assert objectives(source_ast) == objectives(notebook_ast)
    print("PASS: clinical leakage rejection, manual Breslow ties, row-order invariance, finite gradients, censoring, legacy penalty compatibility and notebook/source parity.")


if __name__ == "__main__":
    main()
