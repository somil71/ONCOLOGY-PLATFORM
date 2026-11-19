"""Strict adapters for pinned, unmodified published-method source.

MultiSurv's author mRNA model and loss are retained under their MIT license.
This shared-cohort adapter is not a reproduction of the pan-cancer paper, its
preprocessing, WSI/clinical pathways or a four-input MultiSurv model.
"""

from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import sys
import threading

import numpy as np
import torch
from torch import nn


_VENDOR = Path(__file__).resolve().parents[1] / "third_party/multisurv"
_COMMIT = "deb35e12502f61a59726bfb543210735406d5959"
_IMPORT_LOCK = threading.Lock()


def multisurv_source_provenance():
    manifest_path = _VENDOR / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["commit"] != _COMMIT
        or manifest["license"] != "MIT; original LICENSE retained"
    ):
        raise ValueError("Unexpected official MultiSurv revision/license record")
    for record in manifest["files"]:
        path = (_VENDOR / record["path"]).resolve()
        if not path.is_relative_to(_VENDOR.resolve()):
            raise ValueError("Invalid vendor source path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Modified official MultiSurv source: {record['path']}")
    return {
        "repository": manifest["repository"],
        "commit": manifest["commit"],
        "license": manifest["license"],
        "source_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "source_files_sha256": {x["path"]: x["sha256"] for x in manifest["files"]},
    }


@lru_cache(maxsize=1)
def load_official_multisurv_classes():
    """Load original flat imports without leaving collisions in host sys.modules."""
    multisurv_source_provenance()
    names = ["attention", "embrace_net", "sub_models", "multisurv", "loss"]
    with _IMPORT_LOCK:
        previous = {name: sys.modules.get(name) for name in names}
        old_path = list(sys.path)
        try:
            for name in names:
                sys.modules.pop(name, None)
            sys.path.insert(0, str(_VENDOR / "src"))
            model = importlib.import_module("multisurv")
            criterion = importlib.import_module("loss")
            return model.MultiSurv, criterion.Loss
        finally:
            sys.path[:] = old_path
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


class AuthorMultiSurvRNA(nn.Module):
    """Exact author mRNA-only network/loss with explicit days-based input wrapper.

    Requires the author's 1,000-feature dimension. Model architecture and loss are
    unmodified. Shared-cohort logTPM selection/scaling, new folds and training policy
    are adaptations and must be disclosed with any experiment using this adapter.
    """

    def __init__(self, breaks_days=None):
        super().__init__()
        breaks = (
            np.arange(31, dtype=float) * 365
            if breaks_days is None
            else np.asarray(breaks_days, float)
        )
        if (
            breaks.ndim != 1
            or len(breaks) < 2
            or breaks[0] != 0
            or not np.isfinite(breaks).all()
            or not (np.diff(breaks) > 0).all()
        ):
            raise ValueError(
                "Finite increasing interval endpoints beginning at zero required"
            )
        model, loss = load_official_multisurv_classes()
        self.author_model = model(
            data_modalities=["mRNA"],
            fusion_method=None,
            n_output_intervals=len(breaks) - 1,
            device="cpu",
        )
        self.author_criterion = loss()
        self.register_buffer("breaks_days", torch.tensor(breaks, dtype=torch.float32))
        self.provenance = multisurv_source_provenance()

    def forward(self, rna):
        if (
            not torch.is_tensor(rna)
            or rna.ndim != 2
            or rna.shape[1] != 1000
            or len(rna) == 0
        ):
            raise ValueError(
                "Author mRNA branch requires nonempty [patients,1000] real RNA features"
            )
        if not rna.is_floating_point() or not torch.isfinite(rna).all():
            raise ValueError("Finite floating RNA features required")
        return self.author_model({"mRNA": rna})[1]

    def loss_from_days(self, conditional_survival, time_days, event):
        """Original midpoint censor-credit likelihood, not the local hazard loss.

        The upstream labels use years and round back to days. Deaths at an interval
        boundary fall in the following interval in the upstream implementation.
        """
        time_days = torch.as_tensor(time_days).detach().cpu().float()
        event = torch.as_tensor(event).detach().cpu()
        if time_days.ndim != 1 or event.shape != time_days.shape or not len(time_days):
            raise ValueError("Aligned nonempty survival outcomes required")
        if (
            not torch.isfinite(time_days).all()
            or not (time_days > 0).all()
            or not ((event == 0) | (event == 1)).all()
        ):
            raise ValueError(
                "Positive observed durations and binary recorded events required"
            )
        if (
            conditional_survival.shape != (len(time_days), len(self.breaks_days) - 1)
            or not torch.isfinite(conditional_survival).all()
        ):
            raise ValueError(
                "Aligned finite conditional-survival probabilities required"
            )
        if not ((conditional_survival >= 0) & (conditional_survival <= 1)).all():
            raise ValueError("Conditional-survival probabilities outside[0,1]")
        if (time_days[event.bool()] >= float(self.breaks_days[-1])).any():
            raise ValueError(
                "An observed death lacks an upstream event interval; extend declared breaks before fitting"
            )
        return self.author_criterion(
            conditional_survival,
            times=time_days / 365.0,
            events=event,
            breaks=self.breaks_days.detach().cpu(),
            device=conditional_survival.device,
        )

    def predict_survival(self, rna, horizons_days):
        horizons = np.asarray(horizons_days, float)
        breaks = self.breaks_days.detach().cpu().numpy()
        if (
            horizons.ndim != 1
            or not np.isfinite(horizons).all()
            or (horizons < 0).any()
            or (horizons > breaks[-1]).any()
        ):
            raise ValueError("Supported finite prediction horizons required")
        self.eval()
        with torch.no_grad():
            cumulative = self(rna).cumprod(1)
            completed = np.searchsorted(breaks[1:], horizons, side="right")
            result = torch.ones((len(rna), len(horizons)), device=rna.device)
            for column, count in enumerate(completed):
                if count:
                    result[:, column] = cumulative[:, count - 1]
            return result.cpu().numpy()


MULTISURV_ADAPTATION_NOTICE = {
    "method_label": "MultiSurv-author-mRNA shared-cohort adaptation",
    "retained": "Unmodified pinned author mRNA-only network,30annual conditional-survival outputs and original midpoint-censor likelihood",
    "adapted": "Cancer-specific cohort/folds, log1pTPM training-only1000gene selection/scaling, optimization/early-stopping policy and fixed-boundary prediction export",
    "not_reproduced": "Original pan-cancer study, official nine-category clinical pathway, raw-WSI ResNeXt pathway, imaging preprocessing or four-input/CT architecture",
    "comparison_requirement": "Same eligible patients/outer folds; report feature-budget/training adaptations and compare a1000gene unimodal reference where making architectural claims",
}
