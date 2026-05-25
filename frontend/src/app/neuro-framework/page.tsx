"use client";
import { useEffect, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import NeuroTaxonomyTable from "@/components/NeuroTaxonomyTable";
import NeuroBiomarkerCard from "@/components/NeuroBiomarkerCard";
import { Brain, Activity, Microscope, AlertCircle, BookOpen } from "lucide-react";

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

const ALLEN_LEVELS = [
  {
    level: 6,
    label: "Normal Cognition",
    color: "#22c55e",
    stitch: "Cordovan Stitch",
    code: "Abstract, anticipatory code; foresees edge cases; proactively manages security risks",
  },
  {
    level: 5,
    label: "Mild Impairment",
    color: "#3b82f6",
    stitch: "Whip Stitch",
    code: "Working code with deficits in abstraction, long-term optimization, and structural organization",
  },
  {
    level: 4,
    label: "Early Decline",
    color: "#f97316",
    stitch: "Running Stitch (complex)",
    code: "Familiar puzzles solved; fails to independently learn new APIs or safely resolve novel logic errors",
  },
  {
    level: 3,
    label: "Moderate Decline",
    color: "#ef4444",
    stitch: "Running Stitch (basic)",
    code: "Requires step-by-step cues and direct 1:1 supervision for basic, repetitive task modules",
  },
];

const PATHWAY_STEPS = [
  { label: "Inferior Colliculus", desc: "Directs sensory attention toward salient, threatening, or frustrating inputs" },
  { label: "Periaqueductal Grey (PAG)", desc: "Integrates affective and motoric info; mediates fight-or-flight muscle prep-responses" },
  { label: "Pedunculopontine Tegmental", desc: "Cholinergic pontine structure modulating arousal and muscle tone under stress" },
  { label: "C1–C8 Cervical Cord", desc: "Executes descending commands; disruption leads to cardiac dysrhythmias and HRV reduction" },
];

export default function NeuroFrameworkPage() {
  return (
    <div className="min-h-screen bg-[var(--bg-primary)] transition-theme">

      {/* ── Hero ───────────────────────────────────────────────────────────── */}
      <div className="relative overflow-hidden border-b border-[var(--border-color)]">
        {/* Background pattern */}
        <div className="absolute inset-0 neuro-bg-pattern" />
        <div className="absolute inset-0"
          style={{
            backgroundImage: "radial-gradient(circle at 30% 50%, rgba(0,180,204,0.06) 0%, transparent 60%), radial-gradient(circle at 70% 50%, rgba(168,85,247,0.05) 0%, transparent 60%)"
          }}
        />

        <div className="relative max-w-6xl mx-auto px-6 py-14">
          <div className="flex items-center gap-2 mb-4">
            <span className="badge-cyan">Computational Neuro-Software Engineering</span>
            <span className="badge-purple">C1–C8 Unified Framework</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-black font-display leading-tight text-[var(--text-primary)] mb-4">
            The <span className="gradient-text">C1–C8</span> Multi-Taxonomy<br />
            Landscape
          </h1>
          <p className="text-[var(--text-secondary)] text-lg max-w-2xl leading-relaxed">
            A unified, multi-dimensional taxonomy mapping cognitive parameters across neurological,
            physiological, and computational domains — enabling novel, code-integrable assessment
            paradigms.
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-10 space-y-14">

        {/* ── Full Taxonomy Table ─────────────────────────────────────────── */}
        <section>
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-xl bg-neuro-500/10 border border-neuro-500/20">
              <Brain size={20} className="text-neuro-500" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-[var(--text-primary)] font-display">
                Full C1–C8 Taxonomy Table
              </h2>
              <p className="text-xs text-[var(--text-muted)]">
                Click any row to expand · Toggle columns using the filters above
              </p>
            </div>
          </div>
          <NeuroTaxonomyTable />
        </section>

        {/* ── Working Memory ─────────────────────────────────────────────── */}
        <section className="grid md:grid-cols-2 gap-6">
          <div className="glass-card p-6 space-y-4">
            <div className="flex items-center gap-2">
              <BookOpen size={18} className="text-purple-500" />
              <h2 className="text-lg font-bold text-[var(--text-primary)] font-display">
                Working Memory Limits
              </h2>
            </div>
            <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
              Human working memory — the "mental sketchpad" — is biologically constrained.
              Early cognitive psychology proposed <strong>7 ± 2 chunks</strong>; contemporary
              neuroscience shows the operational limit for complex interactive elements is
              closer to <strong>4–5 items</strong>.
            </p>

            {/* KEDE formula */}
            <div>
              <p className="text-[10px] font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Knowledge Discovery Efficiency (KEDE)
              </p>
              <div className="katex-formula-block">
                <p className="text-xs text-[var(--text-secondary)] mb-2">
                  Scale: 0 = novel scientific breakthrough · 100 = absolute omniscience
                </p>
                <p className="text-xs text-[var(--text-secondary)]">
                  Expert developer baseline: <strong>KEDE ≈ 20</strong> (continuous low-level discovery)
                </p>
              </div>
            </div>
          </div>

          {/* Progressive Variable Tracking */}
          <div className="glass-card p-6 space-y-4">
            <div className="flex items-center gap-2">
              <AlertCircle size={18} className="text-orange-500" />
              <h2 className="text-lg font-bold text-[var(--text-primary)] font-display">
                Progressive Variable Tracking
              </h2>
            </div>
            <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
              Traditional metrics (SonarQube) assign identical complexity to sequential
              assignments. Human cognition accumulates <em>temporal state</em>:
            </p>
            <div className="rounded-xl overflow-hidden border border-[var(--border-color)]">
              <div className="bg-[#0d1117] px-4 py-3 font-mono text-xs leading-relaxed">
                <div className="flex gap-4">
                  <div className="text-gray-500 text-right select-none">1<br/>2<br/>3</div>
                  <div>
                    <span className="text-[#79c0ff]">let</span>{" "}
                    <span className="text-[#e3e4e6]">a</span>{" "}
                    <span className="text-[#ff7b72]">=</span>{" "}
                    <span className="text-[#79c0ff]">1</span>
                    <span className="text-gray-500 ml-4">// ICP = x</span>
                    <br/>
                    <span className="text-[#e3e4e6]">a</span>{" "}
                    <span className="text-[#ff7b72]">=</span>{" "}
                    <span className="text-[#e3e4e6]">a</span>{" "}
                    <span className="text-[#ff7b72]">*</span>{" "}
                    <span className="text-[#79c0ff]">2</span>
                    <span className="text-gray-500 ml-4">// ICP = y {">"} x</span>
                    <br/>
                    <span className="text-[#e3e4e6]">a</span>{" "}
                    <span className="text-[#ff7b72]">=</span>{" "}
                    <span className="text-[#e3e4e6]">a</span>{" "}
                    <span className="text-[#ff7b72]">*</span>{" "}
                    <span className="text-[#79c0ff]">2</span>
                    <span className="text-orange-400 ml-4">// ICP = z {">"} y {">"} x ← human cost!</span>
                  </div>
                </div>
              </div>
            </div>
            <p className="text-xs text-[var(--text-muted)]">
              Line 3 has strictly greater cognitive complexity because the developer must
              retrieve the full accumulated history of <code className="text-neuro-400">a</code> from working memory.
              Traditional tools score this sequence as <strong>0</strong>.
            </p>
          </div>
        </section>

        {/* ── Emo-Motoric Pathway ─────────────────────────────────────────── */}
        <section>
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-xl bg-red-500/10 border border-red-500/20">
              <Activity size={20} className="text-red-500" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-[var(--text-primary)] font-display">
                Emo-Motoric Threat Response Pathway
              </h2>
              <p className="text-xs text-[var(--text-muted)]">
                Cervical spinal cord (C1–C8) as the central nexus of cognitive-somatic stress
              </p>
            </div>
          </div>

          <div className="grid md:grid-cols-4 gap-3">
            {PATHWAY_STEPS.map((step, i) => (
              <div key={step.label} className="relative">
                {i < PATHWAY_STEPS.length - 1 && (
                  <div className="hidden md:block absolute top-6 right-0 w-4 h-px bg-gradient-to-r from-red-500/40 to-transparent z-10" />
                )}
                <div className="glass-card p-4 h-full">
                  <div className="flex items-start gap-2 mb-2">
                    <span
                      className="flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold"
                      style={{ background: "rgba(239,68,68,0.15)", color: "#ef4444", border: "1px solid rgba(239,68,68,0.3)" }}
                    >
                      {i + 1}
                    </span>
                    <p className="text-xs font-bold text-[var(--text-primary)] leading-snug">{step.label}</p>
                  </div>
                  <p className="text-[11px] text-[var(--text-muted)] leading-relaxed">{step.desc}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 glass-card p-4 border-l-4 border-l-orange-500">
            <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
              <strong className="text-[var(--text-primary)]">Cardiac Dysregulation:</strong>{" "}
              During intense cognitive load, sympathetic outflow through upper spinal pathways
              increases, decreasing HRV total power (TP) and driving up the Physical Stress
              Index (PSI). Rehabilitation uses high-PAS (TMS + PNS) to restore homeostatic
              cardiorespiratory control.
            </p>
          </div>
        </section>

        {/* ── CSF Biomarker ─────────────────────────────────────────────── */}
        <section>
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-xl bg-neuro-500/10 border border-neuro-500/20">
              <Microscope size={20} className="text-neuro-500" />
            </div>
            <h2 className="text-xl font-bold text-[var(--text-primary)] font-display">
              CSF Proteomic Biomarker Nexus
            </h2>
          </div>
          <NeuroBiomarkerCard />
        </section>

        {/* ── Allen Model ─────────────────────────────────────────────────── */}
        <section>
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-xl bg-purple-500/10 border border-purple-500/20">
              <Brain size={20} className="text-purple-500" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-[var(--text-primary)] font-display">
                Allen Cognitive Disabilities Model
              </h2>
              <p className="text-xs text-[var(--text-muted)]">
                Claudia Allen's lacing tasks mapped to software developer functional levels
              </p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            {ALLEN_LEVELS.map((al) => (
              <div
                key={al.level}
                className="glass-card p-5"
                style={{ borderColor: `${al.color}20` }}
              >
                <div className="flex items-start gap-3">
                  <div
                    className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 font-bold text-lg"
                    style={{ background: `${al.color}15`, color: al.color, border: `1px solid ${al.color}30` }}
                  >
                    {al.level}
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="font-bold text-[var(--text-primary)] text-sm">{al.label}</span>
                      <span className="text-[10px] px-2 py-0.5 rounded-full font-mono"
                        style={{ color: al.color, background: `${al.color}10`, border: `1px solid ${al.color}25` }}>
                        {al.stitch}
                      </span>
                    </div>
                    <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{al.code}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
