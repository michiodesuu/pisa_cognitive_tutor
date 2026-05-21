"""
visualizations.py
──────────────────
Publication-ready visualizations for the cognitive profile research paper.

Generates:
  1. Dimension correlation heatmap (Spearman ρ between C1-C8 scores)
  2. ICAP level distribution bar chart (across all students)
  3. Engagement trajectory plot (C4-C8 avg over turn number)
  4. Duration vs. cognitive depth scatter (Duration_Sec vs C4 score)
  5. Latent Transition Analysis (LTA) ICAP state transition heatmap
  6. Comprehensive LaTeX table export (Fleiss κ, Gwet AC2, Krippendorff α)

All plots use a clean academic style (no grid clutter).
"""

from __future__ import annotations

import json
import logging
import sqlite3
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

    from scipy.stats import spearmanr

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
            xs.append(float(dur))
            ys.append(int(c4))
            colors.append(ICAP_COLORS.get(icap, "#888"))
        except (ValueError, TypeError):
            pass

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


def export_latex_table(
    reliability_results: Dict,
    ece_result: Dict,
    output_path: Path,
):
    """Export a publication-ready LaTeX table of reliability metrics."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Inter-rater Reliability and Calibration Metrics for the Cognitive Assessment Ensemble}",
        r"\label{tab:reliability}",
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

    # Calibration
    from .calibration import compute_ece_from_scores_db
    ece = compute_ece_from_scores_db(db_path, output_dir)

    export_latex_table(rel, ece, output_dir / "reliability_table.tex")
    logger.info(f"[Viz] Full report saved to {output_dir}/")
