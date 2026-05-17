"""
calibration.py
───────────────
Expected Calibration Error (ECE) and reliability diagrams for the AI ensemble.

ECE measures whether the model's confidence score actually correlates with
accuracy.  If the model says "confidence=0.8" for a set of predictions,
approximately 80% of those should be correct.

Usage:
  from src.metrics.calibration import compute_ece, plot_reliability_diagram
  ece = compute_ece(confidences, correctness_flags, n_bins=10)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_ece(
    confidences: List[float],
    correct: List[bool],
    n_bins: int = 10,
) -> Tuple[float, List[dict]]:
    """
    Expected Calibration Error (ECE).

    Parameters
    ----------
    confidences : list of float ∈ [0, 1]
        Model confidence for each prediction.
    correct : list of bool
        Whether that prediction matched ground truth.
    n_bins : int
        Number of equal-width confidence bins.

    Returns
    -------
    ece : float
        Weighted average |accuracy - confidence| across bins.
    bin_stats : list of dicts
        Per-bin statistics for plotting.
    """
    assert len(confidences) == len(correct), "Length mismatch"
    n = len(confidences)
    if n == 0:
        return float("nan"), []

    confs = np.array(confidences, dtype=float)
    corrs = np.array(correct, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_stats = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs >= lo) & (confs < hi)
        if i == n_bins - 1:          # include right edge on last bin
            mask = (confs >= lo) & (confs <= hi)
        m = mask.sum()
        if m == 0:
            bin_stats.append({
                "bin_lo": round(lo, 3), "bin_hi": round(hi, 3),
                "count": 0, "avg_confidence": None,
                "accuracy": None, "gap": None,
            })
            continue

        avg_conf = float(confs[mask].mean())
        accuracy = float(corrs[mask].mean())
        gap = abs(accuracy - avg_conf)
        ece += (m / n) * gap

        bin_stats.append({
            "bin_lo": round(lo, 3), "bin_hi": round(hi, 3),
            "count": int(m),
            "avg_confidence": round(avg_conf, 4),
            "accuracy": round(accuracy, 4),
            "gap": round(gap, 4),
        })

    return round(float(ece), 6), bin_stats


def plot_reliability_diagram(
    bin_stats: List[dict],
    output_path: Optional[Path] = None,
    title: str = "Reliability Diagram",
):
    """
    Plot a reliability (calibration) diagram.
    Saves to output_path if provided, else shows interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("[Calibration] matplotlib not installed — skipping plot")
        return

    valid = [b for b in bin_stats if b["avg_confidence"] is not None]
    if not valid:
        logger.warning("[Calibration] No valid bins to plot")
        return

    confs = [b["avg_confidence"] for b in valid]
    accs = [b["accuracy"] for b in valid]
    gaps = [b["gap"] for b in valid]
    counts = [b["count"] for b in valid]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: Reliability diagram ─────────────────────────────────────────
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    sc = ax.scatter(confs, accs, c=gaps, cmap="RdYlGn_r", s=100, zorder=3,
                    vmin=0, vmax=0.3)
    plt.colorbar(sc, ax=ax, label="Gap |acc − conf|")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Fraction correct (accuracy)")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()

    # ── Right: Confidence histogram ───────────────────────────────────────
    ax2 = axes[1]
    bin_mids = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bin_stats]
    all_counts = [b["count"] for b in bin_stats]
    ax2.bar(bin_mids, all_counts, width=1/len(bin_stats)*0.9, color="#4C72B0", alpha=0.8)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence histogram")

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info(f"[Calibration] Saved reliability diagram to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def compute_ece_from_scores_db(
    db_path: Path,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Compute ECE directly from the scores SQLite database.
    Treats each (model_vote, gold_dominant) pair as a prediction.
    """
    import sqlite3, json as _json
    if not db_path.exists():
        logger.warning(f"[Calibration] DB not found: {db_path}")
        return {}

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("SELECT individual_votes_json, C1_dominant FROM scores")
    rows = cursor.fetchall()
    conn.close()

    confidences, correct = [], []
    for votes_json, gold_str in rows:
        try:
            votes = _json.loads(votes_json)
            gold = int(gold_str)
        except Exception:
            continue
        for vote in votes:
            conf = float(vote.get("confidence", 0.5))
            pred_str = str(vote.get("scores", {}).get("C1", "NA"))
            try:
                pred = int(pred_str)
                confidences.append(conf)
                correct.append(pred == gold)
            except (ValueError, TypeError):
                pass

    if not confidences:
        return {}

    ece, bin_stats = compute_ece(confidences, correct)
    logger.info(f"[Calibration] ECE = {ece:.4f} over {len(confidences)} predictions")

    if output_path:
        plot_reliability_diagram(bin_stats, output_path / "reliability_diagram.png")

    return {"ece": ece, "n_predictions": len(confidences), "bin_stats": bin_stats}
