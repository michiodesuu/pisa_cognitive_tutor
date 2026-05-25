"use client";
import NeuroMetricsPanel from "@/components/NeuroMetricsPanel";
import ResearchTopicCard, { TOPICS } from "@/components/ResearchTopicCard";
import NeuroBiomarkerCard from "@/components/NeuroBiomarkerCard";
import { BarChart3, FlaskConical, BookOpen, Atom } from "lucide-react";

export default function ResearchMetricsPage() {
  return (
    <div className="min-h-screen bg-[var(--bg-primary)] transition-theme">

      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <div className="relative overflow-hidden border-b border-[var(--border-color)]">
        <div className="absolute inset-0 neuro-bg-pattern" />
        <div
          className="absolute inset-0 opacity-30"
          style={{
            backgroundImage:
              "radial-gradient(ellipse at 20% 50%, rgba(0,180,204,0.12) 0%, transparent 50%), radial-gradient(ellipse at 80% 30%, rgba(168,85,247,0.10) 0%, transparent 50%)",
          }}
        />
        <div className="relative max-w-6xl mx-auto px-6 py-12">
          <div className="flex items-center gap-2 mb-3">
            <span className="badge-cyan">Novel Research Metrics</span>
            <span className="badge-purple">4 Publishable Topics</span>
          </div>
          <h1 className="text-4xl font-black font-display text-[var(--text-primary)] mb-3">
            Research <span className="gradient-text">Metrics & Topics</span>
          </h1>
          <p className="text-[var(--text-secondary)] text-base max-w-2xl leading-relaxed">
            Four original, publishable research proposals bridging computational neuro-science,
            developer telemetry, and software complexity metrics — each with a novel mathematical
            metric formulation.
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-10 space-y-14">

        {/* ── Metric Formulas ─────────────────────────────────────────────── */}
        <section>
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-xl bg-neuro-500/10 border border-neuro-500/20">
              <BarChart3 size={20} className="text-neuro-500" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-[var(--text-primary)] font-display">
                Novel Mathematical Metrics
              </h2>
              <p className="text-xs text-[var(--text-muted)]">
                Click a metric card to view its formula and component definitions
              </p>
            </div>
          </div>
          <NeuroMetricsPanel />
        </section>

        {/* ── CSF Biomarker ─────────────────────────────────────────────── */}
        <section>
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-xl bg-neuro-500/10 border border-neuro-500/20">
              <Atom size={20} className="text-neuro-500" />
            </div>
            <h2 className="text-xl font-bold text-[var(--text-primary)] font-display">
              Molecular Biomarker Nexus
            </h2>
          </div>
          <div className="max-w-xl">
            <NeuroBiomarkerCard />
          </div>
        </section>

        {/* ── Research Topics ─────────────────────────────────────────────── */}
        <section>
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-xl bg-purple-500/10 border border-purple-500/20">
              <BookOpen size={20} className="text-purple-500" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-[var(--text-primary)] font-display">
                Publishable Research Topics
              </h2>
              <p className="text-xs text-[var(--text-muted)]">
                Click any card to expand literature basis, protocol, formula, and research questions
              </p>
            </div>
          </div>
          <div className="space-y-3">
            {TOPICS.map((topic) => (
              <ResearchTopicCard key={topic.id} topic={topic} />
            ))}
          </div>
        </section>

        {/* ── Emerging Literature ─────────────────────────────────────────── */}
        <section>
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-xl bg-green-500/10 border border-green-500/20">
              <FlaskConical size={20} className="text-green-500" />
            </div>
            <h2 className="text-xl font-bold text-[var(--text-primary)] font-display">
              Emerging Research Integrations
            </h2>
          </div>
          <div className="grid md:grid-cols-3 gap-4">
            {[
              {
                color: "#22c55e",
                title: "Brain Age Reversal",
                author: "Moser et al., 2025",
                desc: "Reversing brain aging via youthful immune cells (human iPSC-derived mononuclear phagocytes). Implications for LCAI age-bracket stratification.",
                tag: "Topic 1",
              },
              {
                color: "#3b82f6",
                title: "Lifespan Network Topology",
                author: "Mousley et al., 2025",
                desc: "Five major brain-network topological turning points at ages 9, 32, 66, and 83. Debunks 'mid-20s peak' — maps to developer KEDE and LCAI.",
                tag: "Topic 1",
              },
              {
                color: "#ec4899",
                title: "Neural Prosthesis Speech Decoding",
                author: "Card et al., 2024",
                desc: "Neural-prosthesis speech decoding at 97 words/min. Foundation for future developer brain-computer interface telemetry integration.",
                tag: "Topics 2 & 3",
              },
              {
                color: "#f97316",
                title: "PSYDEFCONV Corpus",
                author: "Na et al., 2026b",
                desc: "First dataset pairing empathetic dialogues with DMRS defense classifications across 8 hierarchical levels (C1–C8). Used in MCI metric (Topic 3).",
                tag: "Topic 3",
              },
              {
                color: "#a855f7",
                title: "AI Code Detection Failure",
                author: "Cuellar Argotty & Manrique, 2025–2026",
                desc: "Reliable classifiers for AI-generated code don't exist across 1,644 samples — slight prompt modifications bypass detector logic. Motivates behavioral tracing.",
                tag: "Topics 3 & 4",
              },
              {
                color: "#00b4cc",
                title: "Computerized Adaptive Testing",
                author: "CDM Framework",
                desc: "Cognitive Diagnostic Modeling groups items by cognitive attributes (C1–C8), providing targeted formative feedback instead of aggregate scores.",
                tag: "All Topics",
              },
            ].map((item) => (
              <div
                key={item.title}
                className="glass-card p-4 space-y-2"
                style={{ borderColor: `${item.color}20` }}
              >
                <div className="flex items-start justify-between gap-2">
                  <p className="font-bold text-[var(--text-primary)] text-sm leading-snug">{item.title}</p>
                  <span
                    className="text-[9px] font-mono font-bold px-1.5 py-0.5 rounded flex-shrink-0"
                    style={{ color: item.color, background: `${item.color}15`, border: `1px solid ${item.color}25` }}
                  >
                    {item.tag}
                  </span>
                </div>
                <p className="text-[10px] font-semibold" style={{ color: item.color }}>{item.author}</p>
                <p className="text-[11px] text-[var(--text-muted)] leading-relaxed">{item.desc}</p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
