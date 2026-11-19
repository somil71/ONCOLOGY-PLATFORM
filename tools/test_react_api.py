"""API integration checks with the actual Phase 2 checkpoint and a temporary case database."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from src.api import ROOT, create_app


def main():
    checkpoint = ROOT / "RUN REPORT" / "CancerFusion_Phase2_Weights.pth"
    with TemporaryDirectory() as temp:
        directory = Path(temp)
        database = directory / "cases.sqlite3"
        build = directory / "dist"
        build.mkdir()
        (build / "index.html").write_text("<h1>CancerFusion test build</h1>", encoding="utf-8")
        options = {"database_path": database, "checkpoint_path": checkpoint, "web_root": build}
        with TestClient(create_app(**options)) as client:
            assert client.get("/api/health").json() == {"status": "ok"}
            assert "CancerFusion test build" in client.get("/").text
            assert client.get("/workspace/case", headers={"Accept": "text/html"}).status_code == 200
            assert client.get("/api/unknown", headers={"Accept": "text/html"}).status_code == 404
            assert client.get("/assets/missing.js", headers={"Accept": "text/html"}).status_code == 404
            model = client.get("/api/model").json()
            assert model["available"] and model["clinical_validated"] is False
            assert model["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            assert client.get("/api/cases").json() == []
            assert client.post("/api/cases", json={"code": "BRCA-001", "source": "upload"}).status_code == 422
            assert client.post("/api/cases", json={
                "code": "BRCA-001", "source": "upload", "features": {"cli": [1]},
            }).status_code == 422
            assert client.post("/api/cases", content=b"{}", headers={"Content-Type": "text/plain"}).status_code == 415
            assert client.post("/api/cases", content=b" " * 2_002_000,
                               headers={"Content-Type": "application/json"}).status_code == 413

            demo = client.post("/api/cases", json={"code": "BRCA-001", "source": "demo"})
            assert demo.status_code == 201, demo.text
            demo = demo.json()
            assert demo["source"] == "synthetic demonstration"
            assert demo["available_modalities"] == ["wsi", "rad", "rna", "cli"]
            assert "features" not in demo
            assert client.post("/api/cases", json={"code": "brca-001", "source": "demo"}).status_code == 409
            demo_path = f"/api/cases/{demo['id']}"
            selection = {"selected_modalities": ["wsi", "rna", "cli"]}
            result = client.post(f"{demo_path}/runs", json=selection)
            assert result.status_code == 200, result.text
            run = result.json()
            assert run["checkpoint"] == model["checkpoint"]
            assert run["outputs"][0]["weights"]["rad"] > 0
            assert run["outputs"][1]["weights"]["rad"] == 0
            assert client.post(f"{demo_path}/runs", json=selection).json()["run_id"] == run["run_id"]
            assert client.post(f"{demo_path}/runs", json={"selected_modalities": []}).status_code == 422
            assert len(client.get(f"{demo_path}/runs").json()) == 1

            report_path = f"{demo_path}/runs/{run['run_id']}/report"
            markdown = client.get(report_path)
            assert "BRCA-001" in markdown.text and "attachment" in markdown.headers["content-disposition"]
            assert client.get(f"{report_path}?format=json").json() == run

            uploaded = client.post("/api/cases", json={
                "code": "BRCA-002", "source": "upload",
                "features": {"cli": [0] * model["feature_shapes"]["cli"][0]},
            })
            assert uploaded.status_code == 201, uploaded.text
            uploaded = uploaded.json()
            upload_path = f"/api/cases/{uploaded['id']}"
            assert uploaded["available_modalities"] == ["cli"]
            assert client.post(f"{upload_path}/runs", json={"selected_modalities": ["rad"]}).status_code == 422
            assert client.get(f"{upload_path}/runs/{run['run_id']}/report").status_code == 404
            assert client.get("/api/cases/unknown").status_code == 404

            figures = client.get("/api/figures").json()
            assert figures and all(client.get(item["url"]).headers["content-type"] == "image/png" for item in figures)
            assert client.get("/api/figures/unknown").status_code == 404
        with TestClient(create_app(**options)) as reopened:
            assert reopened.get(demo_path).json()["runs"] == [run]
            assert reopened.get("/api/cases").json()[1]["run_count"] == 1
        with TestClient(create_app(database_path=database, checkpoint_path=directory / "missing.pth")) as missing:
            assert missing.get("/api/model").json()["available"] is False
            assert missing.get(demo_path).json()["runs"] == [run]
            assert missing.post(f"{demo_path}/runs", json=selection).status_code == 503
    print("PASS: frontend serving, model contract, bounded input validation, persistent cases, actual inference, reports and research figures.")


if __name__ == "__main__":
    main()
