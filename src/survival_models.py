"""Survival research models with full-risk-set Cox and censored discrete hazards.

Discrete hazard likelihood follows Kvamme & Borgan (2021), DOI
10.1007/s10985-021-09532-6. Neural embeddings are learned representations,
not biological concepts or validated uncertainty estimates.
"""

from copy import deepcopy
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import torch
from torch import nn
from torch.nn import functional as F


def validate_outcomes(time, event):
    time, event = np.asarray(time, float), np.asarray(event, float)
    if time.ndim != 1 or event.shape != time.shape or not len(time):
        raise ValueError("Outcome vectors must be nonempty and aligned.")
    if (
        not np.isfinite(time).all()
        or not (time > 0).all()
        or not np.isin(event, [0, 1]).all()
    ):
        raise ValueError(
            "Require finite positive durations and observed binary event indicators."
        )
    return time, event


class RegularizedCox:
    """Full-risk-set Breslow Cox with L2 penalty; no future-outcome predictors."""

    def __init__(self, penalty=0.01):
        if penalty < 0:
            raise ValueError("penalty must be nonnegative")
        self.penalty = float(penalty)

    def fit(self, x, time, event):
        x = np.asarray(x, float)
        time, event = validate_outcomes(time, event)
        if x.ndim != 2 or len(x) != len(time) or not np.isfinite(x).all():
            raise ValueError("Finite 2D aligned covariates required.")
        if not event.sum():
            raise ValueError("Cox fitting requires an observed event.")
        groups = [
            (
                x[(time == t) & (event == 1)].sum(axis=0),
                int(((time == t) & (event == 1)).sum()),
                x[time >= t],
            )
            for t in np.unique(time[event == 1])
        ]

        def objective(beta):
            loss, grad = 0.0, np.zeros_like(beta)
            for death_x, deaths, risk_x in groups:
                z = risk_x @ beta
                logden = logsumexp(z)
                loss += deaths * logden - death_x @ beta
                grad += deaths * (np.exp(z - logden) @ risk_x) - death_x
            return loss / event.sum() + self.penalty * np.square(
                beta
            ).sum() / 2, grad / event.sum() + self.penalty * beta

        result = minimize(
            objective,
            np.zeros(x.shape[1]),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 1000, "ftol": 1e-11, "gtol": 1e-7},
        )
        if not result.success:
            raise RuntimeError(f"Cox optimizer failed: {result.message}")
        self.coef_ = result.x
        self.optimizer_ = {
            "success": bool(result.success),
            "iterations": int(result.nit),
            "loss": float(result.fun),
        }
        self.event_times_ = np.unique(time[event == 1])
        eta = x @ self.coef_
        self.baseline_cumulative_hazard_ = np.cumsum(
            [
                np.exp(
                    np.log(((time == t) & (event == 1)).sum())
                    - logsumexp(eta[time >= t])
                )
                for t in self.event_times_
            ]
        )
        return self

    def predict_risk(self, x):
        return np.asarray(x, float) @ self.coef_

    def predict_survival(self, x, horizons):
        indices = np.searchsorted(self.event_times_, horizons, side="right") - 1
        baseline = np.where(
            indices >= 0, self.baseline_cumulative_hazard_[np.maximum(indices, 0)], 0
        )
        return np.exp(
            -np.exp(np.clip(self.predict_risk(x), -700, 700))[:, None]
            * baseline[None, :]
        )


def discrete_hazard_nll(logits, time, event, boundaries, reduction="mean"):
    """Censoring contributes only completed intervals; event contributes its bin.

    Boundaries are finite interval ends; final logit represents the open tail.
    An event at a boundary is in the ending interval. A censoring observation at
    that boundary contributes its completed-interval survival probability.
    """
    if not all(torch.is_tensor(v) for v in (logits, time, event)):
        raise ValueError("Hazards and outcomes must be tensors")
    if (
        logits.ndim != 2
        or not logits.is_floating_point()
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("Finite floating-point [patients,intervals] logits required")
    if (
        time.ndim != 1
        or event.ndim != 1
        or time.shape != event.shape
        or len(time) != len(logits)
        or not len(time)
    ):
        raise ValueError("One aligned time and event per patient required")
    if time.device != logits.device or event.device != logits.device:
        raise ValueError("Hazards and outcomes must share a device")
    boundaries = torch.as_tensor(boundaries, dtype=time.dtype, device=time.device)
    if boundaries.ndim != 1:
        raise ValueError("One-dimensional interval boundaries required")
    if logits.ndim != 2 or logits.shape[1] != len(boundaries) + 1:
        raise ValueError("Expected one hazard logit per boundary plus a final tail.")
    if len(boundaries) and (
        not torch.isfinite(boundaries).all()
        or not (boundaries > 0).all()
        or not (boundaries[1:] > boundaries[:-1]).all()
    ):
        raise ValueError("Boundaries must be finite, positive and increasing.")
    if (
        not torch.isfinite(time).all()
        or not (time > 0).all()
        or not ((event == 0) | (event == 1)).all()
    ):
        raise ValueError("Invalid survival outcomes.")
    event_bin = torch.searchsorted(boundaries, time.contiguous(), right=False)
    completed = torch.searchsorted(boundaries, time.contiguous(), right=True)
    indices = torch.arange(logits.shape[1], device=logits.device)[None, :]
    survival_count = torch.where(event.bool(), event_bin, completed)
    losses = (F.softplus(logits) * (indices < survival_count[:, None])).sum(dim=1)
    losses = losses + event * F.softplus(
        -logits.gather(1, event_bin[:, None]).squeeze(1)
    )
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction must be mean or none")


class HazardFusion(nn.Module):
    """Masked attention/mean/concatenation fusion for patient feature matrices."""

    def __init__(
        self, dimensions, boundaries, fusion="attention", hidden=32, use_mask=True
    ):
        super().__init__()
        if not dimensions or any(
            not isinstance(d, int) or d < 1 for d in dimensions.values()
        ):
            raise ValueError("Nonempty positive integer modality dimensions required")
        if not isinstance(hidden, int) or hidden < 4 or hidden % 4:
            raise ValueError("Hidden width must be a positive multiple of four heads")
        boundary_array = np.asarray(boundaries, dtype=float)
        if (
            boundary_array.ndim != 1
            or not np.isfinite(boundary_array).all()
            or (boundary_array <= 0).any()
            or (np.diff(boundary_array) <= 0).any()
        ):
            raise ValueError("Finite positive increasing interval boundaries required")
        self.dimensions = dict(dimensions)
        self.names = list(dimensions)
        self.boundaries = list(boundaries)
        if fusion not in ("attention", "mean", "concat"):
            raise ValueError("Unknown fusion")
        self.fusion, self.use_mask = fusion, use_mask
        self.encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU()
                )
                for name, dim in dimensions.items()
            }
        )
        self.type_embeddings = nn.Parameter(
            torch.randn(1, len(dimensions), hidden) * 0.02
        )
        self.attention = nn.MultiheadAttention(hidden, 4, dropout=0, batch_first=True)
        width = (
            hidden * len(dimensions) + len(dimensions) if fusion == "concat" else hidden
        )
        self.head = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, len(boundaries) + 1)
        )

    def forward(self, features, available):
        if set(features) != set(self.names) or not all(
            torch.is_tensor(v) and v.ndim == 2 for v in features.values()
        ):
            raise ValueError(
                "Exactly the configured modality feature matrices required"
            )
        batch = len(features[self.names[0]])
        if (
            not batch
            or not torch.is_tensor(available)
            or available.shape != (batch, len(self.names))
        ):
            raise ValueError("Aligned [patients,modalities] availability mask required")
        if (
            not torch.isfinite(available).all()
            or not ((available == 0) | (available == 1)).all()
        ):
            raise ValueError("Availability must contain only finite binary values")
        available = available.bool()
        if not available.any(dim=1).all():
            raise ValueError("At least one real input per patient required")
        for i, name in enumerate(self.names):
            values = features[name]
            if (
                values.shape != (batch, self.dimensions[name])
                or not values.is_floating_point()
            ):
                raise ValueError(f"Invalid {name} feature shape or dtype")
            if (
                values.device != self.type_embeddings.device
                or available.device != values.device
                or values.dtype != self.type_embeddings.dtype
            ):
                raise ValueError(
                    "Features, masks and model must have compatible device/dtype"
                )
            if not torch.isfinite(values[available[:, i]]).all():
                raise ValueError(f"Nonfinite available {name} features")
        # Clear unavailable raw features before encoding: withheld values cannot leak.
        tokens = torch.stack(
            [
                self.encoders[name](
                    torch.where(
                        available[:, i : i + 1],
                        features[name],
                        torch.zeros_like(features[name]),
                    )
                )
                for i, name in enumerate(self.names)
            ],
            dim=1,
        )
        tokens = tokens + self.type_embeddings
        if self.fusion == "concat":
            pooled = torch.cat(
                [
                    (tokens * available[:, :, None]).flatten(1),
                    available.to(tokens.dtype),
                ],
                dim=1,
            )
        else:
            active = available if self.use_mask else torch.ones_like(available)
            if self.fusion == "attention":
                tokens, _ = self.attention(
                    tokens, tokens, tokens, key_padding_mask=~active, need_weights=False
                )
            pooled = (tokens * active[:, :, None]).sum(1) / active.sum(1, keepdim=True)
        return self.head(pooled)

    def predict_survival(self, features, available, horizons):
        self.eval()
        with torch.no_grad():
            logits = self(features, available)
            survival = torch.sigmoid(-logits).cumprod(dim=1)
            # Step survival changes only at completed finite intervals.
            count = np.searchsorted(self.boundaries, horizons, side="right")
            result = torch.ones((len(logits), len(horizons)), device=logits.device)
            for i, n in enumerate(count):
                if n:
                    result[:, i] = survival[:, n - 1]
            return result.cpu().numpy()


def train_hazard(
    model,
    train_features,
    train_mask,
    train_time,
    train_event,
    val_features,
    val_mask,
    val_time,
    val_event,
    epochs=300,
    patience=35,
    learning_rate=0.003,
    modality_dropout=0.0,
    seed=42,
):
    """Full-batch small-cohort trainer; selection uses validation likelihood only."""
    if not 0 <= modality_dropout < 1:
        raise ValueError("Invalid modality dropout")
    if (
        not isinstance(epochs, int)
        or epochs < 1
        or not isinstance(patience, int)
        or patience < 1
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Positive epochs, patience and learning rate required")
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        epoch_wall_start = time.perf_counter()
        epoch_cpu_start = time.process_time()
        model.train()
        mask = train_mask.clone()
        if modality_dropout:
            mask &= torch.rand(mask.shape, device=mask.device) >= modality_dropout
            empty = ~mask.any(1)
            if empty.any():
                restored = torch.multinomial(
                    train_mask[empty].to(torch.float32), 1
                ).squeeze(1)
                rows = empty.nonzero(as_tuple=False).squeeze(1)
                mask[rows, restored] = True
        optimizer.zero_grad()
        loss = discrete_hazard_nll(
            model(train_features, mask), train_time, train_event, model.boundaries
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = discrete_hazard_nll(
                model(val_features, val_mask), val_time, val_event, model.boundaries
            ).item()
        history.append(
            {
                "epoch": epoch + 1,
                "train_nll": float(loss),
                "validation_nll": val_loss,
                "epoch_wall_seconds": time.perf_counter() - epoch_wall_start,
                "epoch_cpu_process_seconds": time.process_time() - epoch_cpu_start,
            }
        )
        if val_loss < best - 1e-6:
            best, best_state, stale = val_loss, deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No valid checkpoint")
    model.load_state_dict(best_state)
    return history


def predict_cox_bundle(bundle, frame, horizons):
    """Apply a frozen JSON Cox bundle without fitting or changing feature order.

    ``frame`` is a pandas frame. This helper shares the offline transform with
    API consumers; it does not validate clinical use or absolute calibration.
    """
    columns = bundle["columns"]
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Invalid bundled feature order")
    values = frame.loc[:, columns].to_numpy(dtype=float)
    preprocessing = bundle["preprocessing"]
    median = np.asarray(preprocessing["median"], dtype=float)
    mean = np.asarray(preprocessing["mean"], dtype=float)
    scale = np.asarray(preprocessing["scale"], dtype=float)
    coefficients = np.asarray(bundle["coefficients"], dtype=float)
    if any(v.shape != (len(columns),) for v in (median, mean, scale, coefficients)):
        raise ValueError("Bundled preprocessing dimensions disagree with feature order")
    if (
        not all(np.isfinite(v).all() for v in (median, mean, scale, coefficients))
        or (scale <= 0).any()
        or np.isinf(values).any()
    ):
        raise ValueError("Invalid preprocessing values")
    x = (np.where(np.isfinite(values), values, median) - mean) / scale
    x = x.astype(preprocessing.get("output_dtype", "float32"))
    model = RegularizedCox(bundle["penalty"])
    model.coef_ = coefficients
    model.event_times_ = np.asarray(bundle["event_times"], dtype=float)
    model.baseline_cumulative_hazard_ = np.asarray(
        bundle["baseline_cumulative_hazard"], dtype=float
    )
    if (
        model.event_times_.shape != model.baseline_cumulative_hazard_.shape
        or not len(model.event_times_)
        or not np.isfinite(model.event_times_).all()
        or not np.isfinite(model.baseline_cumulative_hazard_).all()
        or (model.event_times_ <= 0).any()
        or (model.baseline_cumulative_hazard_ < 0).any()
        or (np.diff(model.event_times_) <= 0).any()
        or (np.diff(model.baseline_cumulative_hazard_) < 0).any()
    ):
        raise ValueError("Invalid baseline hazard grid")
    horizons = np.asarray(horizons, dtype=float)
    if (
        horizons.ndim != 1
        or not len(horizons)
        or not np.isfinite(horizons).all()
        or (horizons <= 0).any()
    ):
        raise ValueError("Positive finite evaluation horizons required")
    return model.predict_risk(x), model.predict_survival(x, horizons)
