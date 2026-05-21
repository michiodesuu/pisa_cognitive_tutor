"""
reliability.py
───────────────
Inter-rater reliability metrics for the cognitive assessment ensemble.

Implements:
  - Fleiss' Kappa (categorical agreement across K raters, N items)
  - Krippendorff's Alpha (ordinal, handles missing data)
  - Gwet's AC2 (ordinal, quadratic weights — robust to Kappa Paradox / skewed distributions)
  - Per-dimension breakdown
  - Automatic weight update for MajorityVoter

References:
  Fleiss (1971), Krippendorff (2011), Gwet (2014) Handbook of Inter-Rater Reliability
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DIMENSIONS = ["C1","C2","C3","C4","C5","C6","C7","C8"]


# ── Fleiss' Kappa ─────────────────────────────────────────────────────────────

def fleiss_kappa(
    rating_matrix: np.ndarray,
    categories: Optional[List] = None,
) -> float:
    """
    Compute Fleiss' Kappa.

    Parameters
    ----------
    rating_matrix : np.ndarray, shape (N, K)
        N items × K raters.  Values are category labels.
        Missing values should be represented as NaN or None.
    categories : list, optional
        Explicit list of category values.  Auto-detected if None.

    Returns
    -------
    float : kappa ∈ [-1, 1]
    """
    # Convert to category counts matrix (N × C)
    if categories is None:
        flat = rating_matrix.flatten()
        categories = sorted(set(v for v in flat if not (isinstance(v, float) and np.isnan(v))))

    N, K = rating_matrix.shape
    C = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    n_ij = np.zeros((N, C), dtype=float)
    for i in range(N):
        for j in range(K):
            v = rating_matrix[i, j]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                if v in cat_idx:
                    n_ij[i, cat_idx[v]] += 1

    raters_per_item = n_ij.sum(axis=1)
    valid = raters_per_item > 0
    if valid.sum() == 0:
        return float("nan")

    n_ij = n_ij[valid]
    raters_per_item = raters_per_item[valid]
    N_valid = int(valid.sum())

    # P_i: proportion of agreeing pairs for item i
    P_i = (n_ij ** 2).sum(axis=1) - raters_per_item
    P_i /= raters_per_item * (raters_per_item - 1)
    P_bar = P_i.mean()

    # P_j: overall proportion assigned to category j
    p_j = n_ij.sum(axis=0) / n_ij.sum()
    P_e = (p_j ** 2).sum()

    if abs(1 - P_e) < 1e-10:
        return float("nan")

    kappa = (P_bar - P_e) / (1 - P_e)
    return float(kappa)


# ── Krippendorff's Alpha ──────────────────────────────────────────────────────

def krippendorff_alpha(
    data: np.ndarray,
    level_of_measurement: str = "ordinal",
) -> float:
    """
    Krippendorff's Alpha.

    Parameters
    ----------
    data : np.ndarray, shape (K, N)
        K raters × N items.  np.nan = missing.
    level_of_measurement : str
        'nominal' | 'ordinal' | 'interval' | 'ratio'

    Returns
    -------
    float : alpha ∈ [-1, 1]
    """
    K, N = data.shape

    # Coincidence matrix
    categories = sorted(set(
        v for row in data for v in row
        if not np.isnan(v)
    ))
    cat_idx = {c: i for i, c in enumerate(categories)}
    C = len(categories)

    if C < 2:
        return float("nan")

    n_c = np.zeros(C, dtype=float)
    o_ck = np.zeros((C, C), dtype=float)

    for col in range(N):
        vals = [data[r, col] for r in range(K) if not np.isnan(data[r, col])]
        m_u = len(vals)
        if m_u < 2:
            continue
        for i, vi in enumerate(vals):
            for j, vj in enumerate(vals):
                if i != j:
                    ci = cat_idx[vi]
                    cj = cat_idx[vj]
                    o_ck[ci, cj] += 1.0 / (m_u - 1)
            n_c[cat_idx[vi]] += 1.0

    n_total = n_c.sum()
    if n_total < 2:
        return float("nan")

    # Metric function
    def metric(c, k):
        vc, vk = categories[c], categories[k]
        if level_of_measurement == "nominal":
            return 0.0 if c == k else 1.0
        elif level_of_measurement == "ordinal":
            # Sum of n_g for g between c and k inclusive
            lo, hi = min(c, k), max(c, k)
            return (sum(n_c[lo:hi+1]) - (n_c[lo] + n_c[hi]) / 2.0) ** 2
        elif level_of_measurement == "interval":
            return (vc - vk) ** 2
        elif level_of_measurement == "ratio":
            if vc + vk == 0:
                return 0.0
            return ((vc - vk) / (vc + vk)) ** 2
        return 0.0

    D_o = sum(
        o_ck[c, k] * metric(c, k)
        for c in range(C) for k in range(C)
    )
    D_e = sum(
        n_c[c] * n_c[k] * metric(c, k)
        for c in range(C) for k in range(C)
    ) / (n_total - 1)

    if abs(D_e) < 1e-10:
        return float("nan")

    return float(1.0 - D_o / D_e)


# ── Per-dimension reliability ─────────────────────────────────────────────────

def compute_dimension_reliability(
    individual_votes_list: List[List[Dict[str, Any]]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute Fleiss' Kappa and Krippendorff's Alpha per dimension.

    Parameters
    ----------
    individual_votes_list : list of list of dicts
        Each element corresponds to one scored turn.
        Inner list = votes from each model for that turn.
        Each vote dict has keys C1..C8 with values 0/1/2/'NA'.

    Returns
    -------
    dict : {dim: {fleiss_kappa, krippendorff_alpha, n_items, pct_consensus}}
    """
    results = {}

    for dim in DIMENSIONS:
        # Build rating matrix (N × K)
        rows = []
        for turn_votes in individual_votes_list:
            row = []
            for vote in turn_votes:
                v = vote.get("scores", {}).get(dim, "NA")
                if v == "NA" or v is None:
                    row.append(np.nan)
                else:
                    try:
                        row.append(float(int(v)))
                    except (ValueError, TypeError):
                        row.append(np.nan)
            rows.append(row)

        if not rows:
            results[dim] = {"fleiss_kappa": float("nan"), "krippendorff_alpha": float("nan"),
                            "gwet_ac2": float("nan"), "n_items": 0, "pct_consensus": 0.0}
            continue

        K = max(len(r) for r in rows)
        # Pad shorter rows
        padded = [r + [np.nan] * (K - len(r)) for r in rows]
        matrix_NK = np.array(padded, dtype=float)
        matrix_KN = matrix_NK.T

        # For Fleiss (categorical), convert to int labels, drop nan
        cat_matrix = np.where(np.isnan(matrix_NK), None, matrix_NK)

        kappa = fleiss_kappa(cat_matrix, categories=[0.0, 1.0, 2.0])
        alpha = krippendorff_alpha(matrix_KN, level_of_measurement="ordinal")
        ac2 = gwet_ac2(matrix_NK, categories=[0.0, 1.0, 2.0])

        # % of items where all raters agreed
        consensus_count = sum(
            1 for row in padded
            if len(set(v for v in row if not np.isnan(v))) <= 1
            and any(not np.isnan(v) for v in row)
        )
        pct_consensus = round(consensus_count / max(len(rows), 1), 3)

        results[dim] = {
            "fleiss_kappa": round(kappa, 4) if not np.isnan(kappa) else None,
            "krippendorff_alpha": round(alpha, 4) if not np.isnan(alpha) else None,
            "gwet_ac2": round(ac2, 4) if not np.isnan(ac2) else None,
            "n_items": len(rows),
            "pct_consensus": pct_consensus,
        }

    return results


# ── Gwet's AC2 ───────────────────────────────────────────────────────────────

def gwet_ac2(
    rating_matrix: np.ndarray,
    categories: Optional[List] = None,
) -> float:
    """
    Gwet's AC2 with quadratic weights for ordinal data.

    Robust to the Kappa Paradox — does not artificially deflate when class
    distributions are highly skewed (common in educational engagement data).

    Parameters
    ----------
    rating_matrix : np.ndarray, shape (N, K)
        N items × K raters.  np.nan or None = missing.
    categories : list, optional
        Explicit list of ordered category values.  Auto-detected if None.

    Returns
    -------
    float : AC2 ∈ [-1, 1]

    Reference
    ---------
    Gwet (2014) Handbook of Inter-Rater Reliability, 4th ed., §4.4
    Formula: AC2 = (p_a - p_e) / (1 - p_e)
    where p_e = T_w - (1/(K-1)) * Σ_k w(k,k) * π_k * (1-π_k)
    and T_w = Σ_{k,l} w(k,l) * π_k * π_l  (quadratic weights)
    """
    N, K = rating_matrix.shape

    if categories is None:
        flat = rating_matrix.flatten()
        categories = sorted(set(
            v for v in flat
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ))

    q = len(categories)
    if q < 2:
        return float("nan")

    cat_idx = {c: i for i, c in enumerate(categories)}
    r = float(max(categories) - min(categories)) or 1.0

    # Quadratic weight matrix: w(i,j) = 1 - ((vi - vj) / range)²
    W = np.array([
        [1.0 - ((categories[ci] - categories[cj]) / r) ** 2 for cj in range(q)]
        for ci in range(q)
    ])

    # p_a: mean weighted pairwise agreement across all items
    pa_items = []
    for i in range(N):
        vals = [
            rating_matrix[i, j] for j in range(K)
            if rating_matrix[i, j] is not None
            and not (isinstance(rating_matrix[i, j], float) and np.isnan(rating_matrix[i, j]))
        ]
        m = len(vals)
        if m < 2:
            continue
        agree, n_pairs = 0.0, 0
        for a in range(m):
            for b in range(a + 1, m):
                ci = cat_idx.get(vals[a])
                cj = cat_idx.get(vals[b])
                if ci is not None and cj is not None:
                    agree += W[ci, cj]
                    n_pairs += 1
        if n_pairs > 0:
            pa_items.append(agree / n_pairs)

    if not pa_items:
        return float("nan")
    p_a = float(np.mean(pa_items))

    # Marginal proportions π per category
    all_vals = [
        rating_matrix[i, j] for i in range(N) for j in range(K)
        if rating_matrix[i, j] is not None
        and not (isinstance(rating_matrix[i, j], float) and np.isnan(rating_matrix[i, j]))
    ]
    if not all_vals:
        return float("nan")
    pi = np.zeros(q)
    for v in all_vals:
        idx = cat_idx.get(v)
        if idx is not None:
            pi[idx] += 1
    pi /= len(all_vals)

    # Gwet's chance-expected agreement (AC2 formula)
    # T_w = Σ_{k,l} w(k,l) * π_k * π_l
    T_w = float(pi @ W @ pi)
    # Diagonal correction: w(k,k)=1 for quadratic weights
    diag_correction = float(np.sum(np.diag(W) * pi * (1.0 - pi)))
    p_e = T_w - diag_correction / (K - 1) if K > 1 else T_w

    if abs(1.0 - p_e) < 1e-10:
        return float("nan")

    return float((p_a - p_e) / (1.0 - p_e))


def interpret_kappa(k: Optional[float]) -> str:
    if k is None or (isinstance(k, float) and np.isnan(k)):
        return "undefined"
    if k < 0.00: return "poor (< chance)"
    if k < 0.20: return "slight"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"


def interpret_ac2(v: Optional[float]) -> str:
    """AC2 uses stricter thresholds — values > 0.80 are typical for good AI evaluators."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "undefined"
    if v < 0.00: return "poor (< chance)"
    if v < 0.20: return "slight"
    if v < 0.40: return "fair"
    if v < 0.60: return "moderate"
    if v < 0.80: return "substantial"
    return "almost perfect (AI-validated)"


def update_model_kappa_weights(
    individual_votes_list: List[List[Dict[str, Any]]],
    output_path: Path = Path("data/processed_jsonl/model_kappa_weights.json"),
):
    """
    Compute per-model accuracy (agreement with majority) and write
    normalised weights to disk so MajorityVoter can load them.
    """
    model_correct: Dict[str, int] = defaultdict(int)
    model_total: Dict[str, int] = defaultdict(int)

    for turn_votes in individual_votes_list:
        for dim in DIMENSIONS:
            # Majority vote for this dim across models
            vals = []
            for v in turn_votes:
                sv = v.get("scores", {}).get(dim, "NA")
                if sv != "NA" and sv is not None:
                    try:
                        vals.append(int(sv))
                    except Exception:
                        pass
            if len(vals) < 2:
                continue
            from collections import Counter
            majority = Counter(vals).most_common(1)[0][0]
            for v in turn_votes:
                name = v.get("model", "unknown")
                sv = v.get("scores", {}).get(dim, "NA")
                model_total[name] += 1
                if sv != "NA":
                    try:
                        if int(sv) == majority:
                            model_correct[name] += 1
                    except Exception:
                        pass

    weights = {}
    for name in model_total:
        acc = model_correct[name] / model_total[name] if model_total[name] > 0 else 0.5
        # Smooth: weight = 0.5 + acc (range [0.5, 1.5])
        weights[name] = round(0.5 + acc, 4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(weights, indent=2))
    logger.info(f"[Reliability] Updated model weights: {weights}")
    return weights
