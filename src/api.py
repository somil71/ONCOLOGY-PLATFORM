"""Local HTTP API for the CancerFusion AI React research workspace."""

from functools import lru_cache
import hashlib
from io import BytesIO
import json
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.exceptions import HTTPException as StaticFileError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.case_inference import demo_features, feature_shapes, read_case_features, run_case
from src.case_store import CaseStore, report_markdown


ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("wsi", "rad", "rna", "cli")
Modality = Literal["wsi", "rad", "rna", "cli"]
LOGGER = logging.getLogger(__name__)
RESEARCH_FIGURES = {
    "architecture": (
        "Phase 2 prototype architecture", "Four feature branches feed masked attention. This diagram describes the checkpoint design, not an evaluated four-input survival model.",
        "Prototype design", "IEEE_Paper_Handoff/figures/architecture/Figure_A1_Phase2_Architecture_and_Modality_Encoders.png",
    ),
    "kirc-fold-results": (
        "KIRC clinical/RNA fold results", "Five-fold retrospective comparison using clinical and RNA inputs only. Error bars are conditional on saved fitted models.",
        "Evaluated clinical/RNA survival study", "web/public/research/kirc-clinical-rna-fold-results.png",
    ),
    "kirc-paired-differences": (
        "KIRC paired model differences", "Clinical/RNA comparisons on the locked patient folds. The plotted intervals include no established fusion gain over the age-only Cox reference.",
        "Evaluated clinical/RNA survival study", "web/public/research/kirc-clinical-rna-paired-differences.png",
    ),
    "uncertainty-missing-input": (
        "Disagreement under RNA withholding", "In KIRC, RNA withholding worsened five-year Brier error while lowering ensemble disagreement. Disagreement is not calibrated confidence.",
        "Exploratory uncertainty analysis", "web/public/research/uncertainty-missing-input-summary.png",
    ),
}


class CreateCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(max_length=40)
    source: Literal["demo", "upload"]
    features: dict | None = None


class CreateRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_modalities: list[Modality] = Field(min_length=1, max_length=len(MODALITIES))


class ClinicalSurvivalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age_years: float = Field(ge=18, le=120, allow_inf_nan=False)
    pathologic_stage: Literal[1, 2, 3, 4] | None = None


class ReactFiles(StaticFiles):
    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except StaticFileError as exc:
            # A client-side navigation might use a path. Missing assets and API URLs still return 404.
            first_segment = path.replace("\\", "/").partition("/")[0]
            if exc.status_code != 404 or first_segment in {"api", "assets"} or Path(path).suffix:
                raise
            request = Request(scope)
            if "text/html" not in request.headers.get("accept", ""):
                raise
            return await super().get_response("index.html", scope)


async def read_json_request(request: Request, schema, limit: int):
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise HTTPException(415, "Send an application/json request.")
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > limit:
            raise HTTPException(413, "The JSON request is too large.")
        data.extend(chunk)
    try:
        return schema.model_validate_json(data)
    except ValidationError as exc:
        # Omit invalid input values from responses because they can contain case data.
        errors = exc.errors(include_input=False, include_context=False, include_url=False)
        raise HTTPException(422, errors) from exc


@lru_cache(maxsize=1)
def load_checkpoint(path: Path, modified_ns: int, file_size: int):
    import torch

    from src.architecture import CancerFusionResearchPlatform

    cpu_threads = int(os.environ.get("CANCERFUSION_CPU_THREADS", min(4, os.cpu_count() or 1)))
    if not 1 <= cpu_threads <= 32:
        raise ValueError("CANCERFUSION_CPU_THREADS must lie in [1, 32].")
    torch.set_num_threads(cpu_threads)

    data = path.read_bytes()
    state = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
    model = CancerFusionResearchPlatform(
        rna_dim=state["rna_encoder.encoder.0.weight"].shape[1],
        cli_dim=state["cli_encoder.net.0.weight"].shape[1],
        embed_dim=state["fusion.modality_embeddings"].shape[-1],
        max_slices=state["rad_encoder.pos_embedding"].shape[1],
    )
    model.load_state_dict(state, strict=True)
    return model.eval(), feature_shapes(model), {
        "name": path.name, "sha256": hashlib.sha256(data).hexdigest(),
        "implementation_sha256": hashlib.sha256((ROOT / "src" / "architecture.py").read_bytes()).hexdigest(),
        "implementation_version": "masked-image-bags-v2",
    }


def create_app(*, database_path=None, checkpoint_path=None, web_root=None, project_root=ROOT):
    project_root = Path(project_root)
    database = Path(database_path or os.environ.get("CANCERFUSION_DB", project_root / "runtime" / "cases.sqlite3"))
    checkpoint = Path(checkpoint_path or os.environ.get(
        "CANCERFUSION_CHECKPOINT", project_root / "RUN REPORT" / "CancerFusion_Phase2_Weights.pth"
    ))
    static_root = Path(web_root) if web_root is not None else project_root / "web" / "dist"
    application = FastAPI(title="CancerFusion AI Research API")
    application.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"]
    )

    @lru_cache(maxsize=1)
    def store():
        return CaseStore(database)

    def model_resources():
        try:
            metadata = checkpoint.stat()
            return load_checkpoint(checkpoint, metadata.st_mtime_ns, metadata.st_size)
        except FileNotFoundError as exc:
            raise HTTPException(503, "The Phase 2 checkpoint is unavailable.") from exc
        except Exception as exc:
            LOGGER.exception("Could not load the configured Phase 2 checkpoint")
            raise HTTPException(503, "The Phase 2 checkpoint could not be loaded.") from exc

    def case_record(case_id):
        try:
            return store().get_case(case_id)
        except ValueError as exc:
            raise HTTPException(404, "Research case not found.") from exc

    def case_detail(case):
        features = json.loads(case["features"])
        return {
            "id": case["id"], "code": case["code"], "source": case["source"],
            "created_at": case["created_at"], "input_sha256": case["input_sha256"],
            "available_modalities": [key for key in MODALITIES if key in features],
            "runs": store().list_runs(case["id"]),
        }

    def add_case(request):
        _, shapes, _ = model_resources()
        if request.source == "demo":
            if request.features is not None:
                raise HTTPException(422, "Demonstration cases use built-in inputs.")
            features = demo_features(shapes)
            source = "synthetic demonstration"
        else:
            if request.features is None:
                raise HTTPException(422, "Choose a preprocessed feature JSON file to upload.")
            try:
                encoded = json.dumps(request.features, separators=(",", ":")).encode()
                features = read_case_features(encoded, shapes)
            except (ValueError, RecursionError) as exc:
                raise HTTPException(422, str(exc)) from exc
            source = "uploaded features"

        try:
            case_id = store().create_case(request.code, source, features)
        except ValueError as exc:
            status = 409 if str(exc).startswith("This case code already exists.") else 422
            raise HTTPException(status, str(exc)) from exc
        return case_detail(case_record(case_id))

    def infer(case_id, request):
        case = case_record(case_id)
        model, shapes, identity = model_resources()
        selected = request.selected_modalities
        try:
            features = read_case_features(case["features"].encode(), shapes)
            risk, uncertainty, weights = run_case(model, shapes, features, selected)
            outputs = [
                {"condition": condition, "risk": risk[index], "uncertainty": uncertainty[index],
                 "weights": dict(zip(shapes, weights[index]))}
                for index, condition in enumerate(("All supplied inputs", "Selected inputs"))
            ]
            return store().save_run(case_id, identity, selected, outputs)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    def figure_path(figure_id):
        relative_path = RESEARCH_FIGURES[figure_id][-1]
        path = project_root / relative_path
        return path if path.is_file() else None

    @application.get("/api/health")
    def health():
        return {"status": "ok"}

    @application.post("/api/survival/clinical/predict")
    async def predict_clinical_survival(request: Request):
        data = await read_json_request(request, ClinicalSurvivalRequest, 2048)

        def predict():
            import pandas as pd
            from src.survival_models import predict_cox_bundle

            age_only = data.pathologic_stage is None
            run_name = "clean-age-only-20261118" if age_only else "clean-clinical-validated-20261117"
            bundle_name = "clinical_cox_bundle.json"
            bundle_path = project_root / "RUN REPORT" / run_name / bundle_name
            try:
                contents = bundle_path.read_bytes()
                bundle = json.loads(contents)
            except (OSError, ValueError) as exc:
                raise HTTPException(503, "The clean clinical baseline bundle is unavailable.") from exc
            frame = pd.DataFrame([{"age_years": data.age_years, "pathologic_stage": data.pathologic_stage}])
            horizons = [365.0, 1095.0, 1825.0]
            risk, survival = predict_cox_bundle(bundle, frame, horizons)
            return {
                "model": "TCGA-BRCA age-only Cox baseline" if age_only else "TCGA-BRCA exploratory age and pathologic-stage Cox baseline",
                "model_sha256": hashlib.sha256(contents).hexdigest(),
                "implementation_sha256": hashlib.sha256((ROOT / "src" / "survival_models.py").read_bytes()).hexdigest(),
                "inputs": data.model_dump(),
                "imputed_features": [],
                "log_relative_hazard": float(risk[0]),
                "survival_estimates": [
                    {"days": horizon, "model_estimated_survival": float(probability)}
                    for horizon, probability in zip(horizons, survival[0])
                ],
                "scope": "Research demonstration of the frozen clinical baseline; not a four-modality prediction or validated individual prognosis.",
                "clinical_validated": False,
                "time_origin_status": "Diagnosis-time age baseline." if age_only else "Retrospective stage sensitivity: staging-completion date versus diagnosis-origin OS requires review.",
                "uncertainty": None,
                "uncertainty_status": "No calibrated individual uncertainty estimate is available for this baseline.",
            }

        return await run_in_threadpool(predict)

    @application.get("/api/model")
    def model_info():
        try:
            _, shapes, identity = model_resources()
        except HTTPException as exc:
            return {"available": False, "checkpoint": None, "feature_shapes": {}, "clinical_validated": False,
                    "detail": exc.detail}
        return {"available": True, "checkpoint": identity, "feature_shapes": shapes, "clinical_validated": False}

    @application.get("/api/cases")
    def list_cases():
        return store().list_cases()

    @application.post("/api/cases", status_code=201)
    async def register_case(request: Request):
        data = await read_json_request(request, CreateCase, 2_001_024)
        return await run_in_threadpool(add_case, data)

    @application.get("/api/cases/{case_id}")
    def get_case(case_id: str):
        return case_detail(case_record(case_id))

    @application.get("/api/cases/{case_id}/runs")
    def list_runs(case_id: str):
        case_record(case_id)
        return store().list_runs(case_id)

    @application.post("/api/cases/{case_id}/runs")
    async def run_comparison(case_id: str, request: Request):
        data = await read_json_request(request, CreateRun, 2048)
        return await run_in_threadpool(infer, case_id, data)

    @application.get("/api/cases/{case_id}/runs/{run_id}/report")
    def download_report(case_id: str, run_id: str, format: Literal["markdown", "json"] = "markdown"):
        case_record(case_id)
        run = next((item for item in store().list_runs(case_id) if item["run_id"] == run_id), None)
        if run is None:
            raise HTTPException(404, "Recorded run not found for this case.")
        stem = f"{run['case_code']}-{run_id[:8]}"
        if format == "json":
            return JSONResponse(run, headers={"Content-Disposition": f'attachment; filename="{stem}.json"'})
        return Response(report_markdown(run), media_type="text/markdown",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.md"'})

    @application.get("/api/figures")
    def list_figures():
        return [
            {"id": figure_id, "title": title, "caption": caption, "category": category,
             "url": f"/api/figures/{figure_id}"}
            for figure_id, (title, caption, category, _) in RESEARCH_FIGURES.items()
            if figure_path(figure_id) is not None
        ]

    @application.get("/api/figures/{figure_id}")
    def get_figure(figure_id: str):
        path = figure_path(figure_id) if figure_id in RESEARCH_FIGURES else None
        if path is None:
            raise HTTPException(404, "Research figure not found.")
        return FileResponse(path, media_type="image/png")

    @application.get("/api/research/handoff")
    def download_handoff():
        path = project_root / "docs" / "FINAL_EVIDENCE_AND_SCOPE_2026-11-19.md"
        if not path.is_file():
            raise HTTPException(404, "The research handoff is unavailable.")
        return FileResponse(
            path,
            filename="CancerFusion_Final_Evidence_and_Scope_2026-11-19.md",
            media_type="text/markdown",
        )

    if (static_root / "index.html").is_file():
        application.mount("/", ReactFiles(directory=static_root, html=True), name="website")

    return application


app = create_app()
