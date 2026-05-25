"""
src/metrics/neuro_metrics.py

Computational Neuro-Software Engineering — C1–C8 Research Metric Formulas.

Implements:
  - LCAI: Lifespan Cognitive Adaptability Index (Research Topic 1)
  - CASM: Cervical Autonomic Strain Metric       (Research Topic 2)
  - MCI:  Metacognitive Coping Index             (Research Topic 3)
  - PCRS: Proteomic Cognitive Resilience Score   (Research Topic 4)

Also exports the full C1–C8 multi-taxonomy dataset for the API.
"""

from __future__ import annotations
import math
from typing import Optional


# ─── TAXONOMY DATA ────────────────────────────────────────────────────────────

NEURO_TAXONOMY: list[dict] = [
    {
        "id": "C1",
        "cognitive_capacity": "Sustained Attention",
        "neuro_structural": "Anterior/Middle/Posterior Cingulate (58.3% of largest cluster)",
        "cervical_segment": "Head & neck movement; supraspinal cardiac control",
        "clinical_criteria": "Price per test for patient",
        "defense_level": 1,
        "defense_label": "High Adaptive Defenses (e.g., humor, anticipation)",
        "activity_reserve": "Youth/adulthood reading and writing activities",
        "icap_levels": ["Passive", "Active"],
    },
    {
        "id": "C2",
        "cognitive_capacity": "Response Inhibition",
        "neuro_structural": "Medial Frontal Gyrus (39.3% of cluster)",
        "cervical_segment": "Upper shoulders & diaphragm control (breathing)",
        "clinical_criteria": "Portability of testing device",
        "defense_level": 2,
        "defense_label": "Mental Inhibition Defenses (e.g., displacement)",
        "activity_reserve": "Computational and strategic gaming activities",
        "icap_levels": ["Active"],
    },
    {
        "id": "C3",
        "cognitive_capacity": "Speed of Information Processing",
        "neuro_structural": "Superior Frontal Gyrus (21.7% of largest cluster)",
        "cervical_segment": "Deltoid & biceps control (upper arms)",
        "clinical_criteria": "Level of skill needed for administration",
        "defense_level": 3,
        "defense_label": "Minor Image Distortion (e.g., devaluation)",
        "activity_reserve": "Professional and technical continuing education",
        "icap_levels": ["Active", "Constructive"],
    },
    {
        "id": "C4",
        "cognitive_capacity": "Cognitive Flexibility",
        "neuro_structural": "Cingulate Gyrus medial cluster (40.0% of cluster)",
        "cervical_segment": "Wrist extension & partial biceps control",
        "clinical_criteria": "Time required for assessment",
        "defense_level": 4,
        "defense_label": "Disavowal Defenses (e.g., denial, rationalization)",
        "activity_reserve": "Linguistic translation and multi-lingual tasks",
        "icap_levels": ["Constructive"],
    },
    {
        "id": "C5",
        "cognitive_capacity": "Multiple Simultaneous Attention",
        "neuro_structural": "Frontoparietal attentional networks",
        "cervical_segment": "Triceps & wrist extension control",
        "clinical_criteria": "Area Under the Curve (AUC) statistical power",
        "defense_level": 5,
        "defense_label": "Major Image Distortion (e.g., splitting)",
        "activity_reserve": "Complex visual-spatial arts and crafting",
        "icap_levels": ["Constructive"],
    },
    {
        "id": "C6",
        "cognitive_capacity": "Working Memory",
        "neuro_structural": "Prefrontal-parietal retrieval loops",
        "cervical_segment": "Hand & finger flexion (grip strength)",
        "clinical_criteria": "Diagnostic accuracy coefficient",
        "defense_level": 6,
        "defense_label": "Action-oriented Defenses (e.g., acting out)",
        "activity_reserve": "Memory-intensive board games and chess",
        "icap_levels": ["Constructive", "Interactive"],
    },
    {
        "id": "C7",
        "cognitive_capacity": "Category Formation",
        "neuro_structural": "Cingulate Gyrus medial cluster (25.0% of cluster)",
        "cervical_segment": "Upper limb neuro-sensory feedback integration",
        "clinical_criteria": "Usage feasibility in clinical setups",
        "defense_level": 7,
        "defense_label": "Borderline Defensive Level (e.g., projection)",
        "activity_reserve": "Logical categorization and archiving tasks",
        "icap_levels": ["Interactive"],
    },
    {
        "id": "C8",
        "cognitive_capacity": "Pattern Recognition & Inductive Thinking",
        "neuro_structural": "Cerebellar-cortical loop networks",
        "cervical_segment": "Cardiorespiratory autonomic sympathetic outflow",
        "clinical_criteria": "Correlative link with overall brain health",
        "defense_level": 8,
        "defense_label": "Psychotic / Delusional defenses",
        "activity_reserve": "Multi-step scientific and mathematical inquiry",
        "icap_levels": ["Interactive"],
    },
]


# ─── METRIC 1: LCAI ──────────────────────────────────────────────────────────

def compute_lcai(
    theta_a: float,
    kede: float,
    tau: float,
    icp_sum: float,
    epsilon: float = 1e-9,
) -> float:
    """
    Lifespan Cognitive Adaptability Index (Research Topic 1).

    LCAI = (θ_a · log10(KEDE)) / (τ × Σ ICP(u))

    Parameters
    ----------
    theta_a   : Age-bracketed white-matter network integration score (0–1)
    kede      : Knowledge Discovery Efficiency score (1–100)
    tau       : Visual fixation-to-completion latency in seconds
    icp_sum   : Sum of Intrinsic Complexity Points across active AST nodes
    epsilon   : Guard against division-by-zero

    Returns
    -------
    float : LCAI value (higher = more cognitively adaptable)
    """
    kede_clamped = max(kede, 1.0)  # log10(0) is undefined
    denominator = (tau + epsilon) * (icp_sum + epsilon)
    return (theta_a * math.log10(kede_clamped)) / denominator


# ─── METRIC 2: CASM ──────────────────────────────────────────────────────────

def compute_casm(
    lf_hf_ratio: float,
    rmssd: float,
    emg_cervical_delta: float,
    icp_active: float,
    epsilon: float = 1e-9,
) -> float:
    """
    Cervical Autonomic Strain Metric (Research Topic 2).

    CASM = (LF/HF / RMSSD) × (1 + ΔEMG_cervical) · ln(ICP_active)

    Parameters
    ----------
    lf_hf_ratio         : LF/HF spectral power ratio from ECG HRV analysis
    rmssd               : Root mean square of successive heartbeat differences (ms)
    emg_cervical_delta  : Normalized RMS amplitude of cervical trapezius EMG
    icp_active          : Static ICP complexity score of the active code block

    Returns
    -------
    float : CASM value (higher = more autonomic strain)
    """
    if rmssd <= 0:
        rmssd = epsilon
    if icp_active < 1:
        icp_active = 1  # ln(0) is undefined

    hrv_ratio = lf_hf_ratio / rmssd
    emg_factor = 1.0 + emg_cervical_delta
    return hrv_ratio * emg_factor * math.log(icp_active)


# ─── METRIC 3: MCI ───────────────────────────────────────────────────────────

def compute_mci(
    adaptive_freqs: list[float],
    defensive_freqs: list[float],
    dmrs_weights: Optional[list[float]] = None,
    perplexity: float = 0.0,
    epsilon: float = 1e-9,
) -> float:
    """
    Metacognitive Coping Index (Research Topic 3).

    MCI = Σ(w_i · Φ_adaptive_i) / (Σ(w_j · Φ_defensive_j) + Ψ_perplexity)

    Parameters
    ----------
    adaptive_freqs  : Frequencies of high-level defenses (DMRS Levels 1–4), length 4
    defensive_freqs : Frequencies of maladaptive defenses (DMRS Levels 5–8), length 4
    dmrs_weights    : Optional hierarchical DMRS weights (default: uniform)
    perplexity      : Semantic uncertainty of AI suggestions (0–1)

    Returns
    -------
    float : MCI value (higher = more adaptive metacognitive coping)
    """
    if dmrs_weights is None:
        dmrs_weights = [1.0, 1.0, 1.0, 1.0]

    numerator = sum(w * f for w, f in zip(dmrs_weights, adaptive_freqs))
    denominator = sum(w * f for w, f in zip(dmrs_weights, defensive_freqs)) + perplexity
    return numerator / (denominator + epsilon)


# ─── METRIC 4: PCRS ──────────────────────────────────────────────────────────

def compute_pcrs(
    c1_inhibitor: float,
    stie1: float,
    kede_observed: float,
    kede_baseline: float = 20.0,
    casm: float = 1.0,
    epsilon: float = 1e-9,
) -> float:
    """
    Proteomic Cognitive Resilience Score (Research Topic 4).

    PCRS = (ln(C1) / ln(T1)) × (KEDE_obs / KEDE_base) × (1 / CASM)

    Parameters
    ----------
    c1_inhibitor    : Normalized CSF C1-esterase inhibitor concentration (SERPING1, µg/mL)
    stie1           : Normalized CSF sTie-1 tyrosine kinase concentration (µg/mL)
    kede_observed   : Real-time observed KEDE score (1–100)
    kede_baseline   : Developer's baseline KEDE score (default 20 for expert)
    casm            : Somatic strain from compute_casm() — should be > 0

    Returns
    -------
    float : PCRS value (higher = more resilient to choice overload)
    """
    c1 = max(c1_inhibitor, epsilon)
    t1 = max(stie1, epsilon)
    k_obs = max(kede_observed, epsilon)
    k_base = max(kede_baseline, epsilon)
    casm_safe = max(casm, epsilon)

    biomarker_ratio = math.log(c1) / math.log(t1)
    kede_ratio = k_obs / k_base
    return biomarker_ratio * kede_ratio * (1.0 / casm_safe)


# ─── ICP CALCULATOR ──────────────────────────────────────────────────────────

def estimate_icp_from_code(code: str) -> dict:
    """
    Estimate Intrinsic Complexity Points (ICP) from source code.

    Uses AST-based heuristics:
      - Each 'if/else if/elif'           → +1 ICP
      - Each 'for/while' loop            → +1 ICP
      - Each nested structure (indent)   → +0.5 ICP per level beyond 2
      - Recursive calls (def f... f(...))→ +2 ICP
      - Try/except blocks                → +1 ICP

    Returns
    -------
    dict with: total_icp, breakdown, description
    """
    import ast, re

    total_icp = 0
    breakdown: list[str] = []

    # Count control flow keywords
    keywords = {
        "if ": ("conditional", 1),
        "elif ": ("elif branch", 1),
        "for ": ("for loop", 1),
        "while ": ("while loop", 1),
        "try:": ("try block", 1),
        "except": ("except handler", 1),
    }
    for kw, (label, cost) in keywords.items():
        count = code.count(kw)
        if count > 0:
            total_icp += count * cost
            breakdown.append(f"{label}: {count} × {cost} = {count * cost}")

    # Detect recursion (simple heuristic: function name appears inside its own body)
    fn_match = re.search(r"def (\w+)\s*\(", code)
    if fn_match:
        fn_name = fn_match.group(1)
        # Count recursive calls (excluding the definition line)
        body = "\n".join(code.split("\n")[1:])
        rec_calls = len(re.findall(rf"\b{fn_name}\s*\(", body))
        if rec_calls > 0:
            total_icp += 2
            breakdown.append(f"recursion ({fn_name}): +2")

    # Nesting penalty: max indentation level
    max_indent = 0
    for line in code.split("\n"):
        stripped = line.lstrip()
        if stripped:
            indent = (len(line) - len(stripped)) // 4
            max_indent = max(max_indent, indent)
    if max_indent > 2:
        penalty = round((max_indent - 2) * 0.5)
        total_icp += penalty
        breakdown.append(f"deep nesting (level {max_indent}): +{penalty}")

    return {
        "total_icp": total_icp,
        "breakdown": breakdown,
        "description": f"Estimated ICP = {total_icp} (heuristic AST analysis)",
        "complexity_level": (
            "optimal" if total_icp <= 1 else
            "moderate" if total_icp <= 3 else
            "high" if total_icp <= 5 else
            "critical"
        ),
    }
