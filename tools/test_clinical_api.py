"""API/offline roundtrip using the actually trained clean clinical bundle."""

import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from src.api import ROOT, create_app
from src.survival_models import predict_cox_bundle


def main():
    with (
        tempfile.TemporaryDirectory() as folder,
        TestClient(create_app(database_path=Path(folder) / "cases.sqlite3")) as client,
    ):
        for features in (
            {"age_years": 55, "pathologic_stage": 2},
            {"age_years": 65, "pathologic_stage": None},
        ):
            relative = (
                "RUN REPORT/clean-clinical-validated-20261117/clinical_cox_bundle.json"
                if features["pathologic_stage"] is not None
                else "RUN REPORT/clean-age-only-20261118/clinical_cox_bundle.json"
            )
            path = ROOT / relative
            bundle = json.loads(path.read_bytes())
            response = client.post("/api/survival/clinical/predict", json=features)
            assert response.status_code == 200, response.text
            data = response.json()
            risk, survival = predict_cox_bundle(
                bundle, pd.DataFrame([features]), [365, 1095, 1825]
            )
            np.testing.assert_allclose(data["log_relative_hazard"], risk[0])
            np.testing.assert_allclose(
                [r["model_estimated_survival"] for r in data["survival_estimates"]],
                survival[0],
            )
            assert data["clinical_validated"] is False and data["uncertainty"] is None
            assert data["model_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert all(a >= b for a, b in zip(survival[0], survival[0][1:]))
        for invalid in (
            {"age_years": -1},
            {"age_years": 45, "pathologic_stage": 5},
            {"age_years": 55, "event": 1},
        ):
            assert (
                client.post("/api/survival/clinical/predict", json=invalid).status_code
                == 422
            )
        assert (
            client.post(
                "/api/survival/clinical/predict",
                content="{}",
                headers={"Content-Type": "text/plain"},
            ).status_code
            == 415
        )
    print(
        "PASS: clean age-only/stage-sensitive API equals offline models, monotonic curves, provenance and invalid-input rejection."
    )


if __name__ == "__main__":
    main()
