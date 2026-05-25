"use client";
import { useEffect, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import { Dna, TrendingUp, AlertTriangle } from "lucide-react";

function KaTeX({ formula, display = false }: { formula: string; display?: boolean }) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    try {
      katex.render(formula, ref.current, { throwOnError: false, displayMode: display, strict: false });
    } catch { if (ref.current) ref.current.textContent = formula; }
  }, [formula, display]);
  return <span ref={ref} />;
}

export default function NeuroBiomarkerCard() {
  return (
    <div className="glass-card p-6 space-y-5">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="p-2.5 rounded-xl bg-neuro-500/10 border border-neuro-500/20">
          <Dna size={22} className="text-neuro-500 dark:text-neuro-400" />
        </div>
        <div>
          <h3 className="font-bold text-[var(--text-primary)]">CSF Proteomic Biomarker Nexus</h3>
          <p className="text-xs text-[var(--text-muted)]">
            C1-Esterase Inhibitor (SERPING1) · Mendelian Randomization Evidence
          </p>
        </div>
      </div>

      {/* Key Stat */}
      <div
        className="grid grid-cols-3 gap-3"
      >
        <div className="p-3 rounded-xl bg-neuro-500/08 border border-neuro-500/20 text-center">
          <p className="text-2xl font-bold text-neuro-500 dark:text-neuro-400 font-mono">+0.23</p>
          <p className="text-[10px] text-[var(--text-muted)] mt-0.5">SD increase in<br />cognitive performance</p>
        </div>
        <div className="p-3 rounded-xl bg-purple-500/08 border border-purple-500/20 text-center">
          <p className="text-xs font-mono font-bold text-purple-500 dark:text-purple-400 leading-relaxed">
            7.91 × 10<sup>-5</sup>
          </p>
          <p className="text-[10px] text-[var(--text-muted)] mt-0.5">p-value<br />(Mendelian RZ)</p>
        </div>
        <div className="p-3 rounded-xl bg-green-500/08 border border-green-500/20 text-center">
          <p className="text-2xl font-bold text-green-500 dark:text-green-400 font-mono">257k</p>
          <p className="text-[10px] text-[var(--text-muted)] mt-0.5">individuals in<br />cohort analysis</p>
        </div>
      </div>

      {/* MR result description */}
      <div className="text-sm text-[var(--text-secondary)] leading-relaxed">
        Quantitative Mendelian randomization across European-ancestry cohorts demonstrates a
        <span className="font-semibold text-[var(--text-primary)]"> 1 SD increase</span> in genetically
        predicted CSF C1-esterase inhibitor levels yields a
        <span className="font-semibold text-neuro-500 dark:text-neuro-400"> +0.23 SD increase</span> in
        general cognitive performance (95% CI: 0.12–0.35).
      </div>

      {/* PCRS Formula */}
      <div className="katex-formula-block">
        <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-2 font-semibold">
          PCRS Formula (Research Topic 4)
        </p>
        <KaTeX
          formula="PCRS = \left(\frac{\ln(C_1)}{\ln(T_1)}\right) \times \left(\frac{KEDE_{\text{obs}}}{KEDE_{\text{base}}}\right) \cdot \left(\frac{1}{CASM}\right)"
          display
        />
      </div>

      {/* Gene & Contrast */}
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded-xl bg-[var(--bg-tertiary)] border border-[var(--border-color)]">
          <div className="flex items-center gap-1.5 mb-1.5">
            <TrendingUp size={13} className="text-green-500" />
            <span className="text-xs font-semibold text-[var(--text-primary)]">C1-Esterase Inhibitor</span>
          </div>
          <p className="text-[10px] text-[var(--text-muted)] leading-relaxed">
            Gene: <span className="font-mono text-neuro-500 dark:text-neuro-400">SERPING1</span><br />
            ↑ Levels → ↑ Grey matter volume<br />
            ↑ Total brain volume<br />
            <span className="text-green-500">Positive causal link to cognition</span>
          </p>
        </div>
        <div className="p-3 rounded-xl bg-[var(--bg-tertiary)] border border-[var(--border-color)]">
          <div className="flex items-center gap-1.5 mb-1.5">
            <AlertTriangle size={13} className="text-red-500" />
            <span className="text-xs font-semibold text-[var(--text-primary)]">sTie-1 (Tyrosine Kinase)</span>
          </div>
          <p className="text-[10px] text-[var(--text-muted)] leading-relaxed">
            Receptor: <span className="font-mono text-red-400">TIE1</span><br />
            ↑ sTie-1 → ↓ Cognitive performance<br />
            Negative causal relationship<br />
            <span className="text-red-500">Denominator in PCRS formula</span>
          </p>
        </div>
      </div>

      <p className="text-[10px] text-[var(--text-muted)] italic">
        Source: Zagkos et al. — Mendelian randomization analysis, European-ancestry cohorts (n=257,841)
      </p>
    </div>
  );
}
