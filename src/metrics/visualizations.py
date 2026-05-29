"""
visualizations.py
──────────────────
Publication-ready visualizations for the cognitive profile research paper.

Generates (all saved to data/reports/ by run_full_report):
  1. heatmap.png            — Spearman ρ correlation between C1-C8 scores
  2. icap_distribution.png  — ICAP level distribution bar chart (all turns)
  3. trajectory.png         — Cognitive engagement trajectory (C4-C8 vs turn#)
  4. duration_scatter.png   — Response duration vs. causal reasoning depth (H1)
  5. lta_transitions.png    — LTA ICAP state transition probability heatmap
  6. ac2_comparison.png     — Grouped bar: ensemble AC2 vs each individual model
  7. ac2_radar.png          — Radar/spider chart of ensemble AC2 across C1-C8
  8. reliability_table.tex  — LaTeX table: Fleiss κ, Gwet AC2, Krippendorff α

All plots use a clean academic style (no grid clutter, 150 dpi).
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DIMENSIONS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
DIM_LABELS = {
    "C1": "Content Accuracy", "C2": "Concept Depth",
    "C3": "Data Interp.", "C4": "Causal Reasoning",
    "C5": "Evidence Eval.", "C6": "Model Thinking",
    "C7": "Systems/Transfer", "C8": "Metacognition",
}
ICAP_COLORS = {
    "Passive": "#d62728", "Active": "#ff7f0e",
    "Constructive": "#2ca02c", "Interactive": "#1f77b4",
}


def _try_import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.warning("[Viz] matplotlib not installed — skipping plots")
        return None


def _load_scores_from_db(db_path: Path) -> List[Dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("SELECT * FROM scores ORDER BY id")
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]


def plot_dimension_correlation_heatmap(
    scores_rows: List[Dict],
    output_path: Path,
):
    """Spearman correlation heatmap between C1-C8 dominant scores."""
    plt = _try_import_mpl()
    if plt is None:
        return

    import warnings
    from scipy.stats import spearmanr, ConstantInputWarning

    matrix = []
    for row in scores_rows:
        vec = []
        for dim in DIMENSIONS:
            v = row.get(f"{dim}_dominant", "NA")
            try:
                vec.append(float(int(v)))
            except (ValueError, TypeError):
                vec.append(np.nan)
        matrix.append(vec)

    arr = np.array(matrix, dtype=float)
    n_dims = len(DIMENSIONS)
    corr = np.full((n_dims, n_dims), np.nan)

    for i in range(n_dims):
        for j in range(n_dims):
            mask = ~(np.isnan(arr[:, i]) | np.isnan(arr[:, j]))
            if mask.sum() >= 5:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConstantInputWarning)
                    r, _ = spearmanr(arr[mask, i], arr[mask, j])
                corr[i, j] = r

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman rho")

    labels = [DIM_LABELS.get(d, d) for d in DIMENSIONS]
    ax.set_xticks(range(n_dims)); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_dims)); ax.set_yticklabels(labels, fontsize=9)

    for i in range(n_dims):
        for j in range(n_dims):
            if not np.isnan(corr[i, j]):
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(corr[i,j]) < 0.7 else "white")

    ax.set_title("Inter-dimension Spearman Correlation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Viz] Saved heatmap to {output_path}")


def plot_icap_distribution(
    scores_rows: List[Dict],
    output_path: Path,
):
    """Stacked bar of ICAP level distribution across all turns."""
    plt = _try_import_mpl()
    if plt is None:
        return

    from collections import Counter
    counts = Counter(r.get("dominant_icap", "Unknown") for r in scores_rows)
    levels = ["Passive", "Active", "Constructive", "Interactive"]
    vals = [counts.get(l, 0) for l in levels]
    total = sum(vals) or 1
    pcts = [v / total * 100 for v in vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(levels, pcts, color=[ICAP_COLORS.get(l, "#888") for l in levels])
    for bar, pct, cnt in zip(bars, pcts, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%  (n={cnt})", va="center", fontsize=9)
    ax.set_xlabel("% of turns"); ax.set_title("ICAP Level Distribution", fontweight="bold")
    ax.set_xlim(0, max(pcts) + 15)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Viz] Saved ICAP distribution to {output_path}")


def plot_engagement_trajectory(
    scores_rows: List[Dict],
    output_path: Path,
):
    """Mean C4-C8 score vs. turn number (engagement deepening over time)."""
    plt = _try_import_mpl()
    if plt is None:
        return

    from collections import defaultdict
    by_turn: Dict[int, List[float]] = defaultdict(list)
    for row in scores_rows:
        turn = row.get("turn_number", 0)
        vals = []
        for dim in ["C4", "C5", "C6", "C7", "C8"]:
            v = row.get(f"{dim}_dominant", "NA")
            try:
                vals.append(float(int(v)))
            except (ValueError, TypeError):
                pass
        if vals:
            by_turn[turn].append(sum(vals) / len(vals))

    turns = sorted(by_turn.keys())
    means = [np.mean(by_turn[t]) for t in turns]
    sems = [np.std(by_turn[t]) / max(np.sqrt(len(by_turn[t])), 1) for t in turns]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(turns, means, yerr=sems, marker="o", linewidth=2,
                capsize=4, color="#2ca02c", label="Mean C4-C8 +/- SEM")
    ax.set_xlabel("Turn number"); ax.set_ylabel("Mean C4-C8 score (0-2)")
    ax.set_ylim(0, 2); ax.set_title("Cognitive Engagement Trajectory", fontweight="bold")
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Viz] Saved trajectory to {output_path}")


def plot_duration_vs_depth(
    scores_rows: List[Dict],
    output_path: Path,
):
    """Scatter: Duration_Sec vs C4 dominant score (proxy for depth)."""
    plt = _try_import_mpl()
    if plt is None:
        return

    xs, ys, colors = [], [], []
    for row in scores_rows:
        dur = row.get("duration_sec", None)
        c4 = row.get("C4_dominant", "NA")
        icap = row.get("dominant_icap", "Unknown")
        try:
            x = float(dur)
            y = int(c4)
        except (ValueError, TypeError):
            continue
        xs.append(x)
        ys.append(y)
        colors.append(ICAP_COLORS.get(icap, "#888"))

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, c=colors, alpha=0.55, s=35, edgecolors="none")

    ax.axvline(x=8, color="orange", linestyle="--", linewidth=0.9, alpha=0.7, label="Active threshold (8s)")
    ax.axvline(x=25, color="green", linestyle="--", linewidth=0.9, alpha=0.7, label="Constructive threshold (25s)")

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=l)
        for l, c in ICAP_COLORS.items()
    ] + [
        Line2D([0], [0], color="orange", linestyle="--", label="Active (8s)"),
        Line2D([0], [0], color="green", linestyle="--", label="Constructive (25s)"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="upper left")
    ax.set_xlabel("Duration (seconds)"); ax.set_ylabel("C4 Score (Causal Reasoning)")
    ax.set_yticks([0, 1, 2]); ax.set_title("Response Duration vs. Causal Reasoning Depth", fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Viz] Saved duration scatter to {output_path}")


def plot_latent_transition_heatmap(
    scores_rows: List[Dict],
    output_path: Path,
):
    """ICAP state transition probability heatmap (Latent Transition Analysis).

    Rows = origin state, columns = destination state.  Each cell shows the
    empirical probability of transitioning from state i to state j on the
    next consecutive turn within the same session.
    """
    plt = _try_import_mpl()
    if plt is None:
        return

    from collections import defaultdict

    LEVELS = ["Passive", "Active", "Constructive", "Interactive"]
    lvl_idx = {l: i for i, l in enumerate(LEVELS)}
    n = len(LEVELS)
    counts = np.zeros((n, n), dtype=float)

    by_session: Dict[Any, List] = defaultdict(list)
    for row in scores_rows:
        sid = row.get("session_id", row.get("user_id", "unknown"))
        turn = row.get("turn_number", 0)
        icap = row.get("dominant_icap", None)
        if icap in lvl_idx:
            by_session[sid].append((turn, icap))

    for turns in by_session.values():
        turns_sorted = sorted(turns, key=lambda x: x[0])
        for (_, icap1), (_, icap2) in zip(turns_sorted, turns_sorted[1:]):
            counts[lvl_idx[icap1], lvl_idx[icap2]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.where(row_sums > 0, counts / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(probs, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Transition probability")

    ax.set_xticks(range(n)); ax.set_xticklabels(LEVELS, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(LEVELS, fontsize=9)
    ax.set_xlabel("To state"); ax.set_ylabel("From state")
    ax.set_title("ICAP State Transition Probabilities (LTA)", fontweight="bold")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{probs[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if probs[i, j] > 0.5 else "black")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Viz] Saved LTA heatmap to {output_path}")


def _compute_per_model_agreement(
    scores_rows: List[Dict],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute each individual model's Gwet's AC2 against the ensemble dominant score.

    Reads individual_votes_json from every scored row and builds a 2-column
    ratings matrix (model vote, ensemble dominant) per dimension, then computes
    Gwet's AC2 for each (model, dimension) pair.

    Returns
    -------
    dict : {model_name: {dim: ac2_float_or_None}}
    """
    from collections import defaultdict
    from .reliability import gwet_ac2 as _gwet_ac2

    # {model_name: {dim: [(model_val, ensemble_val)]}}
    model_pairs: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for row in scores_rows:
        try:
            votes = json.loads(row.get("individual_votes_json", "[]"))
        except (json.JSONDecodeError, TypeError):
            continue

        for dim in DIMENSIONS:
            ens_raw = row.get(f"{dim}_dominant", "NA")
            try:
                ens_val = float(int(ens_raw))
            except (ValueError, TypeError):
                continue  # no valid ensemble score for this turn/dim

            for vote in votes:
                model_name = vote.get("model", "unknown")
                mv = vote.get("scores", {}).get(dim, "NA")
                try:
                    model_val = float(int(mv))
                except (ValueError, TypeError):
                    model_val = np.nan
                model_pairs[model_name][dim].append((model_val, ens_val))

    results: Dict[str, Dict[str, Optional[float]]] = {}
    for model_name, dim_data in model_pairs.items():
        results[model_name] = {}
        for dim in DIMENSIONS:
            pairs = dim_data.get(dim, [])
            if len(pairs) < 5:
                results[model_name][dim] = None
                continue
            mat = np.array(pairs, dtype=float)  # shape (N, 2)
            ac2 = _gwet_ac2(mat, categories=[0.0, 1.0, 2.0])
            results[model_name][dim] = round(float(ac2), 4) if not np.isnan(ac2) else None

    return results


def plot_ac2_comparison(
    reliability_results: Dict[str, Dict],
    per_model_results: Dict[str, Dict],
    output_path: Path,
):
    """Grouped bar chart: ensemble Gwet's AC2 vs. each individual model per dimension.

    Produces the figure for the paper comparing how much the ensemble outperforms
    any single scorer across all eight cognitive dimensions.

    Parameters
    ----------
    reliability_results : dict
        Output of compute_dimension_reliability — {dim: {"gwet_ac2": float, ...}}.
    per_model_results : dict
        Output of _compute_per_model_agreement — {model_name: {dim: float}}.
    output_path : Path
        Destination PNG (e.g. data/reports/ac2_comparison.png).
    """
    plt = _try_import_mpl()
    if plt is None:
        return

    n_dims = len(DIMENSIONS)
    dim_labels = [DIM_LABELS.get(d, d) for d in DIMENSIONS]

    ensemble_ac2 = [
        max(0.0, reliability_results.get(d, {}).get("gwet_ac2") or 0.0)
        for d in DIMENSIONS
    ]

    model_names = sorted(per_model_results.keys())
    n_groups = len(model_names) + 1  # individual models + ensemble
    width = 0.80 / n_groups
    x = np.arange(n_dims)

    fig, ax = plt.subplots(figsize=(11, 5))

    # Ensemble bars (leftmost in each group, highlighted)
    offset = -(n_groups / 2 - 0.5) * width
    ax.bar(x + offset, ensemble_ac2, width=width, label="Ensemble",
           color="#1f77b4", edgecolor="white", linewidth=0.5, zorder=3)

    # Individual model bars
    model_palette = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for k, (mname, color) in enumerate(zip(model_names, model_palette)):
        model_ac2 = [
            max(0.0, per_model_results[mname].get(d) or 0.0)
            for d in DIMENSIONS
        ]
        offset_k = -(n_groups / 2 - 1.5 - k) * width
        short_name = mname.split(":")[0]  # drop version tag (e.g. "qwen3:8b" → "qwen3")
        ax.bar(x + offset_k, model_ac2, width=width, label=short_name,
               color=color, edgecolor="white", linewidth=0.5, zorder=3)

    # Reference threshold lines
    ax.axhline(0.60, color="#1f77b4", linestyle="--", linewidth=1.0, alpha=0.5,
               label="Target C1–C4 (AC2 ≥ 0.60)")
    ax.axhline(0.50, color="gray", linestyle=":", linewidth=1.0, alpha=0.5,
               label="Target C5–C8 (AC2 ≥ 0.50)")

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Gwet's AC2")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ensemble vs. Individual Model Gwet's AC2 per Cognitive Dimension",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Viz] Saved AC2 comparison bar chart to {output_path}")


def plot_ac2_radar(
    reliability_results: Dict[str, Dict],
    output_path: Path,
    per_model_results: Optional[Dict[str, Dict]] = None,
):
    """Radar/spider chart of Gwet's AC2 across all eight cognitive dimensions.

    The ensemble's AC2 polygon is filled in blue.  If per_model_results is
    provided, the best individual model is overlaid as a dashed orange outline,
    giving a visual sense of the ensemble's improvement.  Reference dashed
    rings mark the 0.60 (C1–C4 target) and 0.50 (C5–C8 target) thresholds.

    Parameters
    ----------
    reliability_results : dict
        Output of compute_dimension_reliability — {dim: {"gwet_ac2": float, ...}}.
    output_path : Path
        Destination PNG (e.g. data/reports/ac2_radar.png).
    per_model_results : dict, optional
        Output of _compute_per_model_agreement.  When provided, the best model
        (highest mean AC2) is overlaid.
    """
    plt = _try_import_mpl()
    if plt is None:
        return

    n = len(DIMENSIONS)
    angles = [i * 2 * np.pi / n for i in range(n)] + [0]  # +1 to close the polygon

    ensemble_vals = [
        max(0.0, min(1.0, reliability_results.get(d, {}).get("gwet_ac2") or 0.0))
        for d in DIMENSIONS
    ]
    ensemble_closed = ensemble_vals + [ensemble_vals[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    # Reference threshold rings
    for ref, ls, lbl in [
        (0.60, "--", "Target C1–C4 (0.60)"),
        (0.50, ":",  "Target C5–C8 (0.50)"),
    ]:
        ax.plot(angles, [ref] * (n + 1), linestyle=ls, linewidth=1.0,
                color="gray", alpha=0.55, label=lbl)

    # Optional: best individual model overlay
    if per_model_results:
        best_model = max(
            per_model_results,
            key=lambda m: float(np.nanmean(
                [v for v in per_model_results[m].values() if v is not None] or [0]
            )),
        )
        bm_vals = [
            max(0.0, per_model_results[best_model].get(d) or 0.0)
            for d in DIMENSIONS
        ]
        bm_closed = bm_vals + [bm_vals[0]]
        ax.fill(angles, bm_closed, alpha=0.10, color="#ff7f0e")
        ax.plot(angles, bm_closed, linewidth=1.4, linestyle="--", color="#ff7f0e",
                label=f"Best model ({best_model.split(':')[0]})")

    # Ensemble polygon
    ax.fill(angles, ensemble_closed, alpha=0.22, color="#1f77b4")
    ax.plot(angles, ensemble_closed, linewidth=2.2, color="#1f77b4",
            marker="o", markersize=5, label="Ensemble AC2")

    # Dimension axis labels
    dim_labels = [DIM_LABELS.get(d, d) for d in DIMENSIONS]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title("Ensemble Gwet's AC2 by Cognitive Dimension",
                 fontweight="bold", pad=18, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18), fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Viz] Saved AC2 radar chart to {output_path}")


def export_latex_table(
    reliability_results: Dict,
    ece_result: Dict,
    output_path: Path,
):
    """Export a publication-ready LaTeX table of reliability metrics."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Ensemble Internal Consistency: Model-to-Model Agreement Across Cognitive"
        r" Dimensions (C1--C8). Metrics computed from \texttt{individual\_votes\_json} in"
        r" the session database; this measures intra-ensemble agreement, not agreement with"
        r" human gold-standard annotations (see Table~\ref{tab:C1}).}",
        r"\label{tab:ensemble-consistency}",
        r"\scriptsize",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Dimension & Fleiss' $\kappa$ & Gwet's AC2 & Interpretation & Krippendorff's $\alpha$ & Consensus \% \\",
        r"\hline",
    ]

    from .reliability import interpret_kappa, interpret_ac2
    for dim in DIMENSIONS:
        r = reliability_results.get(dim, {})
        kappa = r.get("fleiss_kappa")
        ac2 = r.get("gwet_ac2")
        alpha = r.get("krippendorff_alpha")
        consensus = r.get("pct_consensus", 0.0)

        interp = interpret_ac2(ac2) if ac2 is not None else interpret_kappa(kappa)
        kappa_str = f"{kappa:.3f}" if kappa is not None else "---"
        ac2_str = f"{ac2:.3f}" if ac2 is not None else "---"
        alpha_str = f"{alpha:.3f}" if alpha is not None else "---"
        lines.append(
            f"  {DIM_LABELS.get(dim, dim)} & {kappa_str} & {ac2_str} & {interp} & "
            f"{alpha_str} & {consensus * 100:.1f}\\% \\\\"
        )

    ece_val = ece_result.get("ece", None)
    n_pred = ece_result.get("n_predictions", 0)
    ece_str = f"{ece_val:.4f}" if ece_val is not None else "N/A"
    lines += [
        r"\hline",
        f"  \\multicolumn{{6}}{{l}}{{Expected Calibration Error (ECE): "
        f"{ece_str} over {n_pred} predictions}} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info(f"[Viz] Exported LaTeX table to {output_path}")


def compute_results_summary(
    scores: List[Dict],
    db_path: Path,
    rel: Dict,
    output_path: Path,
    n_models: int = 4,
):
    """
    Compute H1–H4 hypothesis metrics and system metrics from real scored data,
    then export a publication-ready LaTeX table to output_path.

    Metrics produced
    ----------------
    H1  Spearman ρ between duration_sec and mean C4–C8 dominant scores.
    H2  P(upgrade) and P(downgrade) from consecutive ICAP transitions.
    H3  Gwet's AC2 per dimension from the reliability dict (already computed).
    H4  Typhoon agreement rate on Thai-language turns vs other models.
    System  responded rate, controversy rate, avg eval latency, KB utilisation.
    """
    ICAP_ORDER = {"Passive": 0, "Active": 1, "Constructive": 2, "Interactive": 3}
    _thai_re = re.compile(r"[฀-๿]")

    # ── H1: Spearman ρ (duration_sec vs mean C4–C8 score) ────────────────────
    durations, depths = [], []
    for row in scores:
        dur = row.get("duration_sec")
        vals = []
        for dim in ["C4", "C5", "C6", "C7", "C8"]:
            v = row.get(f"{dim}_dominant", "NA")
            try:
                vals.append(float(int(v)))
            except (ValueError, TypeError):
                pass
        if dur is not None and vals:
            dur_float = float(dur)
            if 3.0 <= dur_float <= 300.0:
                durations.append(dur_float)
                depths.append(sum(vals) / len(vals))

    h1_rho = h1_p = float("nan")
    if len(durations) >= 5:
        from scipy.stats import spearmanr
        h1_rho, h1_p = spearmanr(durations, depths)

    # ── H2: P(upgrade) / P(downgrade) from LTA transitions ───────────────────
    by_session: Dict[str, List] = defaultdict(list)
    for row in scores:
        by_session[row["session_id"]].append(row)

    upgrades = downgrades = total_trans = 0
    for session_turns in by_session.values():
        sorted_turns = sorted(session_turns, key=lambda x: x.get("turn_number", 0))
        for i in range(len(sorted_turns) - 1):
            fi = ICAP_ORDER.get(sorted_turns[i].get("dominant_icap", ""), -1)
            ti = ICAP_ORDER.get(sorted_turns[i + 1].get("dominant_icap", ""), -1)
            if fi >= 0 and ti >= 0:
                total_trans += 1
                if ti > fi:
                    upgrades += 1
                elif ti < fi:
                    downgrades += 1

    p_upgrade = upgrades / total_trans if total_trans > 0 else 0.0
    p_downgrade = downgrades / total_trans if total_trans > 0 else 0.0

    # ── H4: Typhoon Thai-turn agreement rate ──────────────────────────────────
    conn = sqlite3.connect(str(db_path))
    turn_inputs = {
        row[0]: row[1]
        for row in conn.execute("SELECT id, user_input FROM turns").fetchall()
    }
    conn.close()

    typhoon_thai = [0, 0]    # [agreed, total]
    typhoon_nonthai = [0, 0]
    others_thai = [0, 0]

    for row in scores:
        user_input = turn_inputs.get(row.get("turn_id"), "")
        is_thai = bool(_thai_re.search(user_input))
        consensus = row.get("dominant_icap", "")
        try:
            votes = json.loads(row.get("individual_votes_json", "[]"))
        except Exception:
            continue
        for v in votes:
            agrees = int(v.get("icap_level", "") == consensus)
            if v.get("model", "") == "typhoon":
                if is_thai:
                    typhoon_thai[0] += agrees; typhoon_thai[1] += 1
                else:
                    typhoon_nonthai[0] += agrees; typhoon_nonthai[1] += 1
            elif is_thai:
                others_thai[0] += agrees; others_thai[1] += 1

    typhoon_thai_rate = typhoon_thai[0] / typhoon_thai[1] if typhoon_thai[1] > 0 else float("nan")
    others_thai_rate = others_thai[0] / others_thai[1] if others_thai[1] > 0 else float("nan")
    delta_pp = (typhoon_thai_rate - others_thai_rate) * 100 \
        if not (math.isnan(typhoon_thai_rate) or math.isnan(others_thai_rate)) else float("nan")

    # ── System metrics ────────────────────────────────────────────────────────
    total_dim_slots = controversial_count = 0
    for row in scores:
        for dim in DIMENSIONS:
            if row.get(f"{dim}_dominant", "NA") != "NA":
                total_dim_slots += 1
                if row.get(f"{dim}_controversial", 0):
                    controversial_count += 1

    controversy_rate = controversial_count / total_dim_slots if total_dim_slots > 0 else 0.0
    avg_elapsed = sum(row.get("elapsed_sec", 0) for row in scores) / len(scores)
    avg_responded = sum(row.get("models_responded", 0) for row in scores) / len(scores)
    responded_rate = avg_responded / n_models

    conn2 = sqlite3.connect(str(db_path))
    kb_row = conn2.execute("SELECT AVG(CAST(kb_context_used AS FLOAT)) FROM turns").fetchone()
    conn2.close()
    kb_rate = float(kb_row[0] or 0.0)

    # ── Helper: pass/fail label ───────────────────────────────────────────────
    def _pf(condition: bool, note: str = "") -> str:
        mark = r"\checkmark" if condition else r"\texttimes"
        label = "Pass" if condition else "Fail"
        suffix = f" ({note})" if note else ""
        return f"${mark}$ {label}{suffix}"

    def _fmt(v) -> str:
        if math.isnan(v):
            return "N/A"
        return f"{v:.2f}"

    def _fmtpct(v) -> str:
        return "N/A" if math.isnan(v) else f"{v * 100:.1f}\\%"

    # ── H3: pull AC2 values for C1–C4 and mean C5–C8 ─────────────────────────
    ac2 = {dim: (rel.get(dim) or {}).get("gwet_ac2") for dim in DIMENSIONS}
    c58_vals = [ac2[d] for d in ["C5", "C6", "C7", "C8"] if ac2.get(d) is not None]
    ac2_c58_mean = sum(c58_vals) / len(c58_vals) if c58_vals else float("nan")

    h1_pass  = (not math.isnan(h1_rho)) and h1_rho >= 0.35 and h1_p < 0.05
    h2_pass  = p_upgrade > p_downgrade
    h4_pass  = (not math.isnan(delta_pp)) and delta_pp >= 5.0
    h3_c1    = ac2.get("C1") or float("nan")
    h3_c2    = ac2.get("C2") or float("nan")
    h3_c3    = ac2.get("C3") or float("nan")
    h3_c4    = ac2.get("C4") or float("nan")

    def _h3pf(v, threshold):
        return _pf(not math.isnan(v) and v >= threshold)

    # ── Build LaTeX rows ──────────────────────────────────────────────────────
    p_note = f"$p = {h1_p:.3f}$" if not math.isnan(h1_p) else ""
    rows = [
        (
            r"H1: Spearman $\rho$(C4--C8, duration\_sec)",
            f"${_fmt(h1_rho)}$ {p_note}",
            r"$\geq 0.35$",
            _pf(h1_pass),
        ),
        (
            r"H2: $P(\text{upgrade}) > P(\text{downgrade})$",
            f"${p_upgrade:.2f} > {p_downgrade:.2f}$",
            "Positive",
            _pf(h2_pass),
        ),
        (r"H3: Gwet's AC2 — C1 (Content Accuracy)",     f"${_fmt(h3_c1)}$", r"$\geq 0.60$", _h3pf(h3_c1, 0.60)),
        (r"H3: Gwet's AC2 — C2 (Concept Depth)",        f"${_fmt(h3_c2)}$", r"$\geq 0.60$", _h3pf(h3_c2, 0.60)),
        (r"H3: Gwet's AC2 — C3 (Data Interp.)",         f"${_fmt(h3_c3)}$", r"$\geq 0.60$", _h3pf(h3_c3, 0.60)),
        (r"H3: Gwet's AC2 — C4 (Causal Reasoning)",     f"${_fmt(h3_c4)}$", r"$\geq 0.60$", _h3pf(h3_c4, 0.60)),
        (r"H3: Gwet's AC2 — C5--C8 mean",               f"${_fmt(ac2_c58_mean)}$", r"$\geq 0.50$", _h3pf(ac2_c58_mean, 0.50)),
        (
            r"H4: Typhoon2 Thai-turn agreement rate",
            f"{_fmtpct(typhoon_thai_rate)} "
            f"(others: {_fmtpct(others_thai_rate)}, "
            f"$\\Delta = {'+' if not math.isnan(delta_pp) and delta_pp >= 0 else ''}"
            f"{'N/A' if math.isnan(delta_pp) else f'{delta_pp:.1f}'}$~pp)",
            r"Others median $+5$~pp",
            _pf(h4_pass),
        ),
        (r"Models responded rate", f"${responded_rate * 100:.1f}\\%$", "---", ""),
        (r"Controversy rate",      f"${controversy_rate * 100:.1f}\\%$", "---", ""),
        (r"Avg eval latency / turn", f"${avg_elapsed:.1f}$~s", "---", ""),
        (r"KB utilisation rate",   f"${kb_rate * 100:.1f}\\%$", "---", ""),
    ]

    n = len(scores)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Experimental Results Summary (computed from $N={n}$ scored turns).}}",
        r"\label{tab:results-computed}",
        r"\scriptsize",
        r"\begin{tabularx}{\columnwidth}{@{}Xllc@{}}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Value} & \textbf{Target} & \textbf{Result} \\",
        r"\midrule",
    ]
    for metric, value, target, result in rows:
        lines.append(f"{metric} & {value} & {target} & {result} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"[Viz] Saved results summary to {output_path}")

    # CSV export — plain values, no LaTeX markup
    import csv, re as _re
    _strip = lambda s: _re.sub(r"\\[a-zA-Z]+\{([^}]*)\}|\\[a-zA-Z]+|\$|\\%|~", "", s).strip()
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "target", "result", "n_turns"])
        for metric, value, target, result in rows:
            writer.writerow([_strip(metric), _strip(value), _strip(target), _strip(result), n])
    logger.info(f"[Viz] Saved results summary CSV to {csv_path}")


def run_full_report(
    db_path: Path = Path("data/chat_logs/sessions.db"),
    output_dir: Path = Path("data/reports"),
):
    """Generate all visualizations and LaTeX table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = _load_scores_from_db(db_path)
    if not scores:
        logger.warning("[Viz] No scored data found — run the analyzer pipeline first")
        return

    logger.info(f"[Viz] Generating report from {len(scores)} scored turns...")

    plot_dimension_correlation_heatmap(scores, output_dir / "heatmap.png")
    plot_icap_distribution(scores, output_dir / "icap_distribution.png")
    plot_engagement_trajectory(scores, output_dir / "trajectory.png")
    plot_duration_vs_depth(scores, output_dir / "duration_scatter.png")
    plot_latent_transition_heatmap(scores, output_dir / "lta_transitions.png")

    # Reliability
    votes_lists = []
    for row in scores:
        try:
            votes = json.loads(row.get("individual_votes_json", "[]"))
            votes_lists.append(votes)
        except Exception:
            pass

    from .reliability import compute_dimension_reliability
    rel = compute_dimension_reliability(votes_lists)

    # Per-model agreement vs ensemble dominant (needed for new plots)
    per_model = _compute_per_model_agreement(scores)

    # New: ensemble vs individual model bar chart + radar chart
    plot_ac2_comparison(rel, per_model, output_dir / "ac2_comparison.png")
    plot_ac2_radar(rel, output_dir / "ac2_radar.png", per_model_results=per_model)

    # Calibration
    from .calibration import compute_ece_from_scores_db
    ece = compute_ece_from_scores_db(db_path, output_dir)

    export_latex_table(rel, ece, output_dir / "reliability_table.tex")
    compute_results_summary(scores, db_path, rel, output_dir / "results_summary.tex")
    logger.info(f"[Viz] Full report saved to {output_dir}/")
