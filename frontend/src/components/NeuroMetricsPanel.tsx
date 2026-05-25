"use client";
import { useEffect, useRef, useState } from "react";
import "katex/dist/katex.min.css";
import katex from "katex";

// ── KaTeX Inline Renderer ────────────────────────────────────────────────────
function KaTeX({ formula, display = false }: { formula: string; display?: boolean }) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    try {
      katex.render(formula, ref.current, {
        throwOnError: false,
        displayMode: display,
        strict: false,
      });
    } catch (e) {
      if (ref.current) ref.current.textContent = formula;
    }
  }, [formula, display]);
  return <span ref={ref} />;
}

// ── Animated Gauge ────────────────────────────────────────────────────────────
function MetricGauge({
  value,
  max,
  color,
  label,
  sublabel,
}: {
  value: number;
  max: number;
  color: string;
  label: string;
  sublabel: string;
}) {
  const [animated, setAnimated] = useState(0);
  const r = 42;
  const circumference = 2 * Math.PI * r;
  const offset = circumference - (animated / max) * circumference;

  useEffect(() => {
    const t = setTimeout(() => setAnimated(value), 200);
    return () => clearTimeout(t);
  }, [value]);

  const pct = Math.round((value / max) * 100);

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-24 h-24">
        <svg viewBox="0 0 100 100" className="w-24 h-24 -rotate-90">
          <circle cx="50" cy="50" r={r} fill="none" stroke="rgba(148,163,184,0.15)" strokeWidth="8" />
          <circle
            cx="50"
            cy="50"
            r={r}
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(0.34, 1.56, 0.64, 1)", filter: `drop-shadow(0 0 6px ${color}80)` }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-xl font-bold font-mono" style={{ color }}>{pct}</span>
          <span className="text-[10px] text-[var(--text-muted)] font-medium">/ 100</span>
        </div>
      </div>
      <div className="text-center">
        <p className="text-sm font-bold text-[var(--text-primary)]">{label}</p>
        <p className="text-[10px] text-[var(--text-muted)] mt-0.5">{sublabel}</p>
      </div>
    </div>
  );
}

// ── Metric Card ───────────────────────────────────────────────────────────────
interface MetricDef {
  id: string;
  title: string;
  subtitle: string;
  color: string;
  formula: string;
  description: string;
  value: number;
  max: number;
  components: { sym: string; def: string }[];
}

const METRICS: MetricDef[] = [
  {
    id: "LCAI",
    title: "Lifespan Cognitive Adaptability Index",
    subtitle: "Research Topic 1 · Age-bracketed processing efficiency",
    color: "#00b4cc",
    formula: "LCAI = \\frac{\\theta_a \\cdot \\log_{10}(KEDE)}{\\tau \\times \\sum_{u \\in \\text{AST}} ICP(u)}",
    description:
      "Models a developer's efficiency in resolving software complexity as a function of age-bracketed white-matter network integration and processing latency.",
    value: 63,
    max: 100,
    components: [
      { sym: "\\theta_a", def: "Age-bracketed white-matter network integration score" },
      { sym: "KEDE", def: "Knowledge Discovery Efficiency (0–100)" },
      { sym: "\\tau", def: "Visual fixation-to-completion latency (seconds)" },
      { sym: "ICP(u)", def: "Intrinsic Complexity Points of active AST nodes" },
    ],
  },
  {
    id: "CASM",
    title: "Cervical Autonomic Strain Metric",
    subtitle: "Research Topic 2 · Somatic coupling of debugging stress",
    color: "#a855f7",
    formula: "CASM = \\left(\\frac{LF/HF}{RMSSD}\\right) \\times \\left(1 + \\overline{\\Delta EMG}_{\\text{cervical}}\\right) \\cdot \\ln(ICP_{\\text{active}})",
    description:
      "Quantifies somatic coupling of emotional stress during code execution using heart-rate variability spectral ratio and cervical EMG amplitude.",
    value: 41,
    max: 100,
    components: [
      { sym: "LF/HF", def: "Low-to-high frequency HRV spectral power ratio" },
      { sym: "RMSSD", def: "Root mean square of successive heartbeat differences" },
      { sym: "\\overline{\\Delta EMG}", def: "Normalized cervical trapezius muscle activation" },
      { sym: "ICP_{\\text{active}}", def: "Static complexity score of active code block" },
    ],
  },
  {
    id: "MCI",
    title: "Metacognitive Coping Index",
    subtitle: "Research Topic 3 · Adaptive vs. defensive AI response",
    color: "#22c55e",
    formula: "MCI = \\frac{\\sum_{i=1}^{4} w_i \\cdot \\Phi(\\text{Adaptive}_i)}{\\sum_{j=5}^{8} w_j \\cdot \\Phi(\\text{Defensive}_j) + \\Psi(\\text{Perplexity})}",
    description:
      "Evaluates balance between adaptive problem-solving and defensive regression under AI-induced cognitive load using DMRS defense classifications.",
    value: 55,
    max: 100,
    components: [
      { sym: "\\Phi(\\text{Adaptive}_i)", def: "Frequency of high-level adaptive defenses (Levels 1–4)" },
      { sym: "\\Phi(\\text{Defensive}_j)", def: "Frequency of maladaptive defenses (Levels 5–8)" },
      { sym: "w_i, w_j", def: "Hierarchical DMRS weights per defense level" },
      { sym: "\\Psi(\\text{Perplexity})", def: "Semantic uncertainty of AI code suggestions" },
    ],
  },
  {
    id: "PCRS",
    title: "Proteomic Cognitive Resilience Score",
    subtitle: "Research Topic 4 · CSF biomarker resilience to overload",
    color: "#f97316",
    formula: "PCRS = \\left(\\frac{\\ln(C_1)}{\\ln(T_1)}\\right) \\times \\left(\\frac{KEDE_{\\text{obs}}}{KEDE_{\\text{base}}}\\right) \\cdot \\left(\\frac{1}{CASM}\\right)",
    description:
      "Models developer resilience to computational choice overload using CSF C1-esterase inhibitor and sTie-1 protein concentrations as predictors.",
    value: 72,
    max: 100,
    components: [
      { sym: "C_1", def: "Normalized CSF C1-esterase inhibitor concentration (SERPING1)" },
      { sym: "T_1", def: "sTie-1 tyrosine-protein kinase receptor concentration" },
      { sym: "KEDE_{\\text{obs}}", def: "Observed real-time Knowledge Discovery Efficiency" },
      { sym: "KEDE_{\\text{base}}", def: "Developer's baseline KEDE score (~20 for expert)" },
    ],
  },
];

export default function NeuroMetricsPanel() {
  const [active, setActive] = useState<string>(METRICS[0].id);
  const metric = METRICS.find((m) => m.id === active)!;

  return (
    <div className="space-y-6">
      {/* Gauge Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {METRICS.map((m) => (
          <button
            key={m.id}
            onClick={() => setActive(m.id)}
            className={`glass-card p-4 text-left transition-all duration-200 cursor-pointer
              ${active === m.id ? "ring-2" : "hover:ring-1 ring-transparent"}`}
            style={{
              ...(active === m.id
                ? { boxShadow: `0 0 20px ${m.color}30, 0 0 0 2px ${m.color}60` }
                : {}),
            }}
          >
            <MetricGauge
              value={m.value}
              max={m.max}
              color={m.color}
              label={m.id}
              sublabel={m.id === "LCAI" ? "Adaptability" : m.id === "CASM" ? "Strain" : m.id === "MCI" ? "Coping" : "Resilience"}
            />
          </button>
        ))}
      </div>

      {/* Detail Card */}
      <div className="glass-card p-6 transition-all duration-300" style={{ borderColor: `${metric.color}30` }}>
        <div className="flex items-start justify-between mb-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span
                className="text-xs font-mono font-bold px-2 py-0.5 rounded"
                style={{ color: metric.color, background: `${metric.color}15`, border: `1px solid ${metric.color}30` }}
              >
                {metric.id}
              </span>
              <span className="text-xs text-[var(--text-muted)]">{metric.subtitle}</span>
            </div>
            <h3 className="text-lg font-bold text-[var(--text-primary)]">{metric.title}</h3>
            <p className="text-sm text-[var(--text-secondary)] mt-1 leading-relaxed">{metric.description}</p>
          </div>
        </div>

        {/* Formula */}
        <div className="katex-formula-block my-4">
          <KaTeX formula={metric.formula} display />
        </div>

        {/* Components */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-4">
          {metric.components.map((c) => (
            <div
              key={c.sym}
              className="flex items-start gap-3 p-2.5 rounded-lg bg-[var(--bg-tertiary)] border border-[var(--border-color)]"
            >
              <span
                className="font-mono text-xs flex-shrink-0 px-1.5 py-0.5 rounded mt-0.5"
                style={{ color: metric.color, background: `${metric.color}10` }}
              >
                <KaTeX formula={c.sym} />
              </span>
              <span className="text-xs text-[var(--text-secondary)] leading-relaxed">{c.def}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
