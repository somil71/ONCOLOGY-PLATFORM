"""Outcome-independent shared-gene rank representation for transport experiments.

This is a distinct exploratory representation, not a claim that microarray and
RNA-seq expression values are directly interchangeable. Metadata gene-symbol
intersection is fixed before outcome-based selection. Duplicate assay entries
are averaged within each patient; missing entries use the patient's median.
"""

from collections import defaultdict
import numpy as np
from scipy.stats import rankdata


def shared_gene_symbols(source_symbols, target_symbols):
    def valid(symbol):
        name = str(symbol).strip()
        return bool(name) and name.casefold() not in ("nan", "none", "na")

    return sorted(
        {str(s).strip() for s in source_symbols if valid(s)}
        & {str(s).strip() for s in target_symbols if valid(s)}
    )


def rank_expression(values, gene_symbols, shared_symbols):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(gene_symbols):
        raise ValueError("Expression dimensions disagree with gene symbols")
    if not shared_symbols or len(set(shared_symbols)) != len(shared_symbols):
        raise ValueError("Unique nonempty shared-gene contract required")
    groups = defaultdict(list)
    for i, symbol in enumerate(gene_symbols):
        groups[str(symbol).strip()].append(i)
    missing = set(shared_symbols) - set(groups)
    if missing:
        raise ValueError(f"Shared genes absent from assay: {sorted(missing)[:10]}")
    matrix = np.empty((len(values), len(shared_symbols)), dtype=np.float32)
    for j, symbol in enumerate(shared_symbols):
        selected = values[:, groups[symbol]]
        count = np.isfinite(selected).sum(1)
        total = np.where(np.isfinite(selected), selected, 0).sum(1)
        matrix[:, j] = np.divide(
            total, count, out=np.full(len(values), np.nan), where=count > 0
        )
    if (~np.isfinite(matrix).any(1)).any():
        raise ValueError("Patient has no usable shared expression")
    median = np.nanmedian(matrix, axis=1)
    matrix = np.where(np.isfinite(matrix), matrix, median[:, None])
    return (
        (rankdata(matrix, axis=1, method="average") - 0.5) / len(shared_symbols)
    ).astype("float32")
