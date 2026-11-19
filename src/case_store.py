"""Local, durable research cases and immutable model-run records."""

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class CaseStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY,
                    code TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    features TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(id),
                    created_at TEXT NOT NULL,
                    signature TEXT NOT NULL UNIQUE,
                    report TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS runs_by_case ON runs(case_id, created_at);
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def create_case(self, code, source, features):
        code = code.strip().upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{2,39}", code):
            raise ValueError("Use a case code of 3–40 letters, numbers, hyphens or underscores.")
        if source not in ("synthetic demonstration", "uploaded features"):
            raise ValueError("Unknown input source.")
        payload = canonical_json({key: tensor.tolist() for key, tensor in features.items()})
        case_id = uuid.uuid4().hex
        with closing(self.connect()) as db, db:
            try:
                db.execute("INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?)", (
                    case_id, code, source, datetime.now(timezone.utc).isoformat(),
                    hashlib.sha256(payload.encode()).hexdigest(), payload,
                ))
            except sqlite3.IntegrityError as exc:
                raise ValueError("This case code already exists. Open the saved case or choose a new code.") from exc
        return case_id

    def list_cases(self):
        with closing(self.connect()) as db:
            rows = db.execute("""
                SELECT c.id, c.code, c.source, c.created_at,
                       COUNT(r.id) AS run_count
                FROM cases c LEFT JOIN runs r ON c.id = r.case_id
                GROUP BY c.id ORDER BY c.created_at DESC
            """).fetchall()
        return [dict(row) for row in rows]

    def get_case(self, case_id):
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        if row is None:
            raise ValueError("The requested case does not exist.")
        return dict(row)

    def save_run(self, case_id, checkpoint, selected, outputs):
        case = self.get_case(case_id)
        selected = sorted(set(selected))
        if not selected or set(selected) - json.loads(case["features"]).keys():
            raise ValueError("Select at least one available modality.")
        identity = {
            "case_id": case_id, "input_sha256": case["input_sha256"],
            "checkpoint_sha256": checkpoint["sha256"], "selected": selected,
            "inference_version": 1,
        }
        signature = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
        report = {
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "case_code": case["code"], "input_source": case["source"],
            "input_sha256": case["input_sha256"],
            "available_modalities": list(json.loads(case["features"])),
            "selected_modalities": selected,
            "checkpoint": checkpoint, "outputs": outputs,
            "interpretation": "Exploratory research scores. Clinical outcome performance is unvalidated.",
        }
        payload = canonical_json(report)
        with closing(self.connect()) as db, db:
            db.execute("INSERT OR IGNORE INTO runs VALUES (?, ?, ?, ?, ?)", (
                report["run_id"], case_id, report["created_at"], signature, payload,
            ))
            # A retried submission returns the original immutable run.
            saved = db.execute("SELECT report FROM runs WHERE signature = ?", (signature,)).fetchone()
        return json.loads(saved["report"])

    def list_runs(self, case_id):
        with closing(self.connect()) as db:
            rows = db.execute(
                "SELECT report FROM runs WHERE case_id = ? ORDER BY created_at DESC", (case_id,)
            ).fetchall()
        return [json.loads(row["report"]) for row in rows]


def report_markdown(report):
    lines = [
        "# CancerFusion AI research report", "",
        f"Case **{report['case_code']}**", "",
        f"Input source — {report['input_source']}", "",
        f"Recorded at {report['created_at']} (UTC)", "",
        report["interpretation"], "",
        "This Phase 2 checkpoint report shows raw research outputs. It is not a validated survival prediction, calibrated confidence estimate, diagnosis, or treatment recommendation.", "",
        "| Condition | Relative risk score | Raw evidence score (not calibrated confidence) |",
        "| --- | ---: | ---: |",
    ]
    for result in report["outputs"]:
        lines.append(f"| {result['condition']} | {result['risk']:.6f} | {result['uncertainty']:.6f} |")
    lines += ["", "| Modality | Supplied | Selected | Baseline weight | Selected weight |",
              "| --- | --- | --- | ---: | ---: |"]
    for key in ("wsi", "rad", "rna", "cli"):
        lines.append(
            f"| {key.upper()} | {key in report['available_modalities']} | {key in report['selected_modalities']} "
            f"| {report['outputs'][0]['weights'][key]:.4f} | {report['outputs'][1]['weights'][key]:.4f} |"
        )
    lines += ["", f"Checkpoint SHA-256 — {report['checkpoint']['sha256']}", "",
              f"Input SHA-256 — {report['input_sha256']}", "", f"Run ID — {report['run_id']}"]
    return "\n".join(lines) + "\n"
