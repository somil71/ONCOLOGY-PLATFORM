from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from streamlit.testing.v1 import AppTest

from src.architecture import CancerFusionResearchPlatform
from src.case_inference import demo_features, feature_shapes, read_case_features, run_case
from src.case_store import CaseStore, report_markdown


ROOT = Path(__file__).resolve().parents[1]


def assert_clean(app):
    assert not app.exception, app.exception


def verify_storage_and_inputs(database):
    state = torch.load(ROOT / "RUN REPORT" / "CancerFusion_Phase2_Weights.pth", map_location="cpu", weights_only=True)
    model = CancerFusionResearchPlatform(
        rna_dim=state["rna_encoder.encoder.0.weight"].shape[1],
        cli_dim=state["cli_encoder.net.0.weight"].shape[1],
        max_slices=state["rad_encoder.pos_embedding"].shape[1],
    ).eval()
    model.load_state_dict(state)
    shapes = feature_shapes(model)
    features = read_case_features(json.dumps({"cli": [0.0] * shapes["cli"][0]}).encode(), shapes)
    risk, uncertainty, weights = run_case(model, shapes, features, ["cli"])
    assert weights == [[0.0, 0.0, 0.0, 1.0]] * 2
    for payload in (b'{"cli":[0]}', b'{}', b'{"unexpected":[1]}'):
        try:
            read_case_features(payload, shapes)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted an invalid feature upload")
    try:
        run_case(model, shapes, features, [])
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted an all-missing inference request")

    store = CaseStore(database)
    case_id = store.create_case("BRCA-TEST", "uploaded features", features)
    outputs = [
        {"condition": condition, "risk": risk[i], "uncertainty": uncertainty[i], "weights": dict(zip(shapes, weights[i]))}
        for i, condition in enumerate(("All supplied inputs", "Selected inputs"))
    ]
    def save(_):
        return store.save_run(case_id, {"sha256": "test-checkpoint"}, ["cli"], outputs)
    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(save, range(4)))
    assert len({report["run_id"] for report in reports}) == 1
    assert len(CaseStore(database).list_runs(case_id)) == 1
    assert "uploaded features" in report_markdown(reports[0])
    try:
        store.create_case("BRCA-TEST", "uploaded features", features)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate case code accepted")
    assert len(store.list_cases()) == 1
    assert json.loads(store.get_case(case_id)["features"])["cli"] == [0.0] * shapes["cli"][0]


def main():
    torch.set_num_threads(2)
    with TemporaryDirectory() as folder:
        verify_storage_and_inputs(Path(folder) / "storage.sqlite3")
        os.environ["CANCERFUSION_DB"] = str(Path(folder) / "app.sqlite3")
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        assert_clean(app)
        for slug, heading in (("platform", "Platform"), ("research", "Research"), ("figures", "Figures"), ("workspace", "Case workspace")):
            app.query_params["page"] = slug
            app.run()
            assert_clean(app)
            assert any(f"<h1>{heading}</h1>" in item.value for item in app.markdown)
        app.text_input[0].set_value("DEMO-TEST")
        next(button for button in app.button if button.label == "Save case").click().run()
        assert_clean(app)
        assert app.selectbox[0].format_func(app.selectbox[0].value).startswith("DEMO-TEST")
        app.checkbox[1].uncheck().run()
        next(button for button in app.button if button.label == "Run and record comparison").click().run()
        assert_clean(app)
        assert len(CaseStore(os.environ["CANCERFUSION_DB"]).list_cases()) == 1
        runs = CaseStore(os.environ["CANCERFUSION_DB"]).list_runs(app.session_state["saved_case"])
        assert len(runs) == 1
        assert runs[0]["outputs"][1]["weights"]["rad"] == 0.0
        app.run()
        assert_clean(app)
        assert len(app.get("download_button")) == 2
        # A new browser session can recover the same case and result.
        reopened = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        reopened.query_params["page"] = "workspace"
        reopened.run()
        assert_clean(reopened)
        assert len(reopened.get("download_button")) == 2
        os.environ.pop("CANCERFUSION_DB")
    print("PASS: pages, case creation, persisted results, missing RAD, exports, input validation and concurrent retry deduplication.")


if __name__ == "__main__":
    main()
