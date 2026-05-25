"use client";
import { useEffect, useRef, useState } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import { ChevronDown, ChevronUp, BookOpen, FlaskConical, BarChart3, HelpCircle } from "lucide-react";

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

interface ResearchTopic {
  id: number;
  title: string;
  subtitle: string;
  color: string;
  litBasis: string;
  protocol: string;
  metricId: string;
  formula: string;
  rq: [string, string];
  dimensions: string[];
}

const TOPICS: ResearchTopic[] = [
  {
    id: 1,
    title: "Mapping Developer Cognitive Debt to Lifespan Brain-Network Topological Turning Points",
    subtitle: "C1–C8 × Brain Development × Age Stratification",
    color: "#00b4cc",
    litBasis:
      "Mousley et al. (2025) debunked the 'mid-20s peak' myth and identified 5 major developmental phases with critical transitions at ages 9, 32, 66, and 83. This study maps these neurological 'OS upgrades' to programming proficiency.",
    protocol:
      "N=150 professional developers, stratified across 3 age brackets (<32, 32–65, >66). Real-time cognitive load via fNIRS (oxygenated hemoglobin), NASA-TLX, and continuous HRV. Codebases vary Halstead vocabulary, McCabe control flow, and DepDegree.",
    metricId: "LCAI",
    formula: "LCAI = \\frac{\\theta_a \\cdot \\log_{10}(KEDE)}{\\tau \\times \\sum_{u \\in \\text{AST}} ICP(u)}",
    rq: [
      "How do brain-network turning points at ages 32 and 66 affect working memory (C6) and cognitive flexibility (C4) when managing nested software architectures?",
      "Does 'slower but more strategic processing' in developers aged 66+ yield lower Physical Stress Index (PSI) when resolving silent unhandled exceptions?",
    ],
    dimensions: ["C4", "C6"],
  },
  {
    id: 2,
    title: "Somatosensory Diagnostics in High-Pressure Debugging: Cervical Spinal Cord (C1–C8) Emo-Motoric Stress",
    subtitle: "Debugging Stress × Cervical Autonomics × HRV Coupling",
    color: "#a855f7",
    litBasis:
      "The cervical spine innervates trapezius muscles and regulates supraspinal autonomic cardiac control. High-pressure debugging induces emo-motoric responses via PAG and pontine projections. Builds on spinal cord fMRI BOLD signal evidence.",
    protocol:
      "Participants debug cascade-failing live software under time pressure. High-resolution spinal cord fMRI captures BOLD in C1–C8 dorsal/ventral regions. High-frequency ECG and cervical sEMG track HRV (TP, RMSSD) and muscle tension.",
    metricId: "CASM",
    formula: "CASM = \\left(\\frac{LF/HF}{RMSSD}\\right) \\times \\left(1 + \\overline{\\Delta EMG}_{\\text{cervical}}\\right) \\cdot \\ln(ICP_{\\text{active}})",
    rq: [
      "To what extent do emotionally aversive software crashes elicit statistically significant BOLD signal increases in C1–C8 cervical spinal cord segments?",
      "How does CASM-measured autonomic strain correlate with speed of information processing (C3) and error detection rate during complex refactoring?",
    ],
    dimensions: ["C1", "C2", "C3"],
  },
  {
    id: 3,
    title: "Generative AI as a Metacognitive Disruptor: Developer Defense Classifications via PSYDEFCONV Corpus",
    subtitle: "AI Code Bugs × DMRS Defense Levels × Pupillometry",
    color: "#22c55e",
    litBasis:
      "GitHub Copilot shifts cognitive effort from writing to validation, increasing metacognitive demands. The PSYDEFCONV corpus (Na et al., 2026b) maps dialogic interactions to 8 DMRS defense levels. A QLoRA-adapted Ministral-8B classifies developer verbalizations.",
    protocol:
      "Developers complete engineering tasks with AI assistance. AI introduces subtle logical bugs in 30% of suggestions. Think-aloud protocols + chat logs classified via fine-tuned 9-class Ministral-8B DMRS model. Real-time pupillometry + NASA-TLX indices.",
    metricId: "MCI",
    formula: "MCI = \\frac{\\sum_{i=1}^{4} w_i \\cdot \\Phi(\\text{Adaptive}_i)}{\\sum_{j=5}^{8} w_j \\cdot \\Phi(\\text{Defensive}_j) + \\Psi(\\text{Perplexity})}",
    rq: [
      "How does introducing incorrect AI code suggestions affect the distribution of a developer's C1–C8 PSYDEFCONV defense levels?",
      "Does higher frequency of maladaptive defenses (Levels 5–8) correlate with elevated pupillary hippus and lower final code validation accuracy?",
    ],
    dimensions: ["C6", "C7", "C8"],
  },
  {
    id: 4,
    title: "Quantitative Proteomic Profiling of CSF Biomarkers: C1-Esterase Inhibitor vs. Choice Overload",
    subtitle: "SERPING1 × Cognitive Resilience × Software Perplexity",
    color: "#f97316",
    litBasis:
      "Zagkos et al. showed CSF C1-esterase inhibitor (SERPING1) causally linked to superior cognition (+0.23 SD, p=7.91×10⁻⁵). sTie-1 shows negative causal relationship. This paper bridges molecular neurobiology with software engineering performance.",
    protocol:
      "N=80 clinically characterized individuals with lumbar puncture CSF profiling (C1-esterase inhibitor + sTie-1 quantification). Algorithmic wayfinding and debugging tasks with CogniSense batteries, pupillometry, and ECG-derived HRV/PSI.",
    metricId: "PCRS",
    formula: "PCRS = \\left(\\frac{\\ln(C_1)}{\\ln(T_1)}\\right) \\times \\left(\\frac{KEDE_{\\text{obs}}}{KEDE_{\\text{base}}}\\right) \\cdot \\left(\\frac{1}{CASM}\\right)",
    rq: [
      "Do higher CSF C1-esterase inhibitor concentrations yield greater resilience to software choice overload, measured by lower CASM and preserved task accuracy?",
      "How do genetically predicted C1-esterase inhibitor and sTie-1 levels correlate with category formation (C7) and processing speed (C3) during structural refactoring?",
    ],
    dimensions: ["C3", "C7"],
  },
];

interface Props {
  topic: ResearchTopic;
}

export default function ResearchTopicCard({ topic }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div
      className="glass-card overflow-hidden transition-all duration-300"
      style={{ borderColor: `${topic.color}20` }}
    >
      {/* Header */}
      <button
        onClick={() => setOpen(!open)}
        className="w-full text-left p-5"
      >
        <div className="flex items-start gap-4">
          {/* Number */}
          <div
            className="flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center font-bold text-lg"
            style={{ background: `${topic.color}15`, color: topic.color, border: `1px solid ${topic.color}30` }}
          >
            {topic.id}
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex flex-wrap gap-1.5 mb-1.5">
              {topic.dimensions.map((d) => (
                <span
                  key={d}
                  className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded"
                  style={{ color: topic.color, background: `${topic.color}10`, border: `1px solid ${topic.color}25` }}
                >
                  {d}
                </span>
              ))}
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded border border-[var(--border-color)] text-[var(--text-muted)]">
                {topic.metricId}
              </span>
            </div>
            <h4 className="font-bold text-[var(--text-primary)] text-sm leading-snug">{topic.title}</h4>
            <p className="text-[10px] text-[var(--text-muted)] mt-0.5">{topic.subtitle}</p>
          </div>

          <span className="flex-shrink-0 text-[var(--text-muted)] mt-1">
            {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </span>
        </div>
      </button>

      {/* Expanded content */}
      {open && (
        <div className="border-t border-[var(--border-color)] px-5 pb-5 pt-4 space-y-4">
          {/* Lit basis */}
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <BookOpen size={12} style={{ color: topic.color }} />
              <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: topic.color }}>
                Literature Basis
              </span>
            </div>
            <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{topic.litBasis}</p>
          </div>

          {/* Protocol */}
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <FlaskConical size={12} style={{ color: topic.color }} />
              <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: topic.color }}>
                Experimental Protocol
              </span>
            </div>
            <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{topic.protocol}</p>
          </div>

          {/* Formula */}
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <BarChart3 size={12} style={{ color: topic.color }} />
              <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: topic.color }}>
                {topic.metricId} Formula
              </span>
            </div>
            <div className="katex-formula-block">
              <KaTeX formula={topic.formula} display />
            </div>
          </div>

          {/* RQs */}
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <HelpCircle size={12} style={{ color: topic.color }} />
              <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: topic.color }}>
                Research Questions
              </span>
            </div>
            <div className="space-y-2">
              {topic.rq.map((rq, i) => (
                <div
                  key={i}
                  className="flex gap-2.5 p-3 rounded-lg bg-[var(--bg-tertiary)] border border-[var(--border-color)]"
                >
                  <span
                    className="flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold mt-0.5"
                    style={{ background: `${topic.color}15`, color: topic.color }}
                  >
                    {i + 1}
                  </span>
                  <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{rq}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Export topics for use in page
export { TOPICS };
