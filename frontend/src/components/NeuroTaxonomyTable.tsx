"use client";
import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

// ── Taxonomy Data ─────────────────────────────────────────────────────────────
const TAXONOMY = [
  {
    id: "C1",
    cognitive: "Sustained Attention",
    neuro: "Anterior/Middle/Posterior Cingulate (58.3%)",
    cervical: "Head & neck movement; supraspinal cardiac control",
    clinical: "Price per test for patient",
    defense: "Level 1: High Adaptive Defenses (humor, anticipation)",
    activity: "Youth/adulthood reading & writing activities",
    icap: "Passive–Active",
    icapClass: "icap-passive",
    color: "#ef4444",
    dimClass: "dim-c1",
    icapLabel: "P–A",
  },
  {
    id: "C2",
    cognitive: "Response Inhibition",
    neuro: "Medial Frontal Gyrus (39.3%)",
    cervical: "Upper shoulders & diaphragm control (breathing)",
    clinical: "Portability of testing device",
    defense: "Level 2: Mental Inhibition Defenses (displacement)",
    activity: "Computational & strategic gaming",
    icap: "Active",
    icapClass: "icap-active",
    color: "#f97316",
    dimClass: "dim-c2",
    icapLabel: "A",
  },
  {
    id: "C3",
    cognitive: "Speed of Information Processing",
    neuro: "Superior Frontal Gyrus (21.7%)",
    cervical: "Deltoid & biceps control (upper arms)",
    clinical: "Level of skill needed for administration",
    defense: "Level 3: Minor Image Distortion (devaluation)",
    activity: "Professional & technical continuing education",
    icap: "Active–Constructive",
    icapClass: "icap-active",
    color: "#eab308",
    dimClass: "dim-c3",
    icapLabel: "A–C",
  },
  {
    id: "C4",
    cognitive: "Cognitive Flexibility",
    neuro: "Cingulate Gyrus medial cluster (40.0%)",
    cervical: "Wrist extension & partial biceps control",
    clinical: "Time required for assessment",
    defense: "Level 4: Disavowal Defenses (denial, rationalization)",
    activity: "Linguistic translation & multi-lingual tasks",
    icap: "Constructive",
    icapClass: "icap-constructive",
    color: "#22c55e",
    dimClass: "dim-c4",
    icapLabel: "C",
  },
  {
    id: "C5",
    cognitive: "Multiple Simultaneous Attention",
    neuro: "Frontoparietal attentional networks",
    cervical: "Triceps & wrist extension control",
    clinical: "Area Under the Curve (AUC) statistical power",
    defense: "Level 5: Major Image Distortion (splitting)",
    activity: "Complex visual-spatial arts & crafting",
    icap: "Constructive",
    icapClass: "icap-constructive",
    color: "#14b8a6",
    dimClass: "dim-c5",
    icapLabel: "C",
  },
  {
    id: "C6",
    cognitive: "Working Memory",
    neuro: "Prefrontal-parietal retrieval loops",
    cervical: "Hand & finger flexion (grip strength)",
    clinical: "Diagnostic accuracy coefficient",
    defense: "Level 6: Action-oriented Defenses (acting out)",
    activity: "Memory-intensive board games & chess",
    icap: "Constructive–Interactive",
    icapClass: "icap-constructive",
    color: "#3b82f6",
    dimClass: "dim-c6",
    icapLabel: "C–I",
  },
  {
    id: "C7",
    cognitive: "Category Formation",
    neuro: "Cingulate Gyrus medial cluster (25.0%)",
    cervical: "Upper limb neuro-sensory feedback integration",
    clinical: "Usage feasibility in clinical setups",
    defense: "Level 7: Borderline Defensive Level (projection)",
    activity: "Logical categorization & archiving tasks",
    icap: "Interactive",
    icapClass: "icap-interactive",
    color: "#8b5cf6",
    dimClass: "dim-c7",
    icapLabel: "I",
  },
  {
    id: "C8",
    cognitive: "Pattern Recognition & Inductive Thinking",
    neuro: "Cerebellar-cortical loop networks",
    cervical: "Cardiorespiratory autonomic sympathetic outflow",
    clinical: "Correlative link with overall brain health",
    defense: "Level 8: Psychotic / Delusional defenses",
    activity: "Multi-step scientific & mathematical inquiry",
    icap: "Interactive",
    icapClass: "icap-interactive",
    color: "#ec4899",
    dimClass: "dim-c8",
    icapLabel: "I",
  },
];

const COLUMNS = [
  { key: "cognitive",  label: "Core Cognitive Capacity" },
  { key: "neuro",      label: "Neuro-Structural ALE Cluster" },
  { key: "cervical",   label: "Cervical Segment (C1–C8)" },
  { key: "clinical",   label: "Clinical Valuation Criteria" },
  { key: "defense",    label: "PSYDEFCONV Defense Level" },
  { key: "activity",   label: "Activity Reserve Classification" },
];

interface Props {
  compact?: boolean;
}

export default function NeuroTaxonomyTable({ compact = false }: Props) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [visibleCols, setVisibleCols] = useState<string[]>(
    compact ? ["cognitive", "neuro", "defense"] : COLUMNS.map((c) => c.key)
  );

  const toggleCol = (key: string) => {
    setVisibleCols((prev) =>
      prev.includes(key) ? (prev.length > 1 ? prev.filter((k) => k !== key) : prev) : [...prev, key]
    );
  };

  return (
    <div className="rounded-2xl border border-[var(--border-color)] overflow-hidden bg-[var(--bg-secondary)] transition-theme">
      {/* Column toggles */}
      {!compact && (
        <div className="px-5 py-3 border-b border-[var(--border-color)] flex flex-wrap gap-2 bg-[var(--bg-tertiary)]">
          <span className="text-xs text-[var(--text-muted)] font-medium mr-1 self-center">Columns:</span>
          {COLUMNS.map((col) => (
            <button
              key={col.key}
              onClick={() => toggleCol(col.key)}
              className={`text-xs px-2.5 py-1 rounded-lg border transition-all duration-200 font-medium
                ${visibleCols.includes(col.key)
                  ? "bg-neuro-500/10 border-neuro-500/40 text-neuro-600 dark:text-neuro-400"
                  : "border-[var(--border-color)] text-[var(--text-muted)] hover:border-neuro-500/30"
                }`}
            >
              {col.label}
            </button>
          ))}
        </div>
      )}

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-[var(--bg-tertiary)] border-b border-[var(--border-color)]">
              <th className="px-4 py-3 text-left text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider w-20">
                Dim.
              </th>
              <th className="px-3 py-3 text-left text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider w-16">
                ICAP
              </th>
              {COLUMNS.filter((c) => visibleCols.includes(c.key)).map((col) => (
                <th
                  key={col.key}
                  className="px-4 py-3 text-left text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider"
                >
                  {col.label}
                </th>
              ))}
              <th className="w-8" />
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-color)]">
            {TAXONOMY.map((row) => (
              <>
                <tr
                  key={row.id}
                  onClick={() => setExpanded(expanded === row.id ? null : row.id)}
                  className="hover:bg-[var(--bg-tertiary)] cursor-pointer transition-colors group"
                >
                  {/* Dimension ID */}
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-2 h-8 rounded-full flex-shrink-0"
                        style={{ backgroundColor: row.color, opacity: 0.8 }}
                      />
                      <span
                        className="font-mono font-bold text-sm"
                        style={{ color: row.color }}
                      >
                        {row.id}
                      </span>
                    </div>
                  </td>

                  {/* ICAP Badge */}
                  <td className="px-3 py-3">
                    <span
                      className={`text-xs font-bold px-2 py-0.5 rounded-full border ${row.icapClass}`}
                    >
                      {row.icapLabel}
                    </span>
                  </td>

                  {/* Dynamic columns */}
                  {COLUMNS.filter((c) => visibleCols.includes(c.key)).map((col) => (
                    <td key={col.key} className="px-4 py-3 text-[var(--text-secondary)] text-xs leading-relaxed max-w-[220px]">
                      {row[col.key as keyof typeof row]}
                    </td>
                  ))}

                  {/* Expand toggle */}
                  <td className="pr-3 py-3">
                    <span className="text-[var(--text-muted)] group-hover:text-[var(--neuro-cyan)] transition-colors">
                      {expanded === row.id ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    </span>
                  </td>
                </tr>

                {/* Expanded row */}
                {expanded === row.id && (
                  <tr key={`${row.id}-expand`} className="bg-[var(--bg-tertiary)]">
                    <td colSpan={visibleCols.length + 3} className="px-6 py-4">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="glass-card p-3">
                          <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Core Cognitive</p>
                          <p className="text-sm font-semibold" style={{ color: row.color }}>{row.cognitive}</p>
                        </div>
                        <div className="glass-card p-3">
                          <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Neuro-Structural</p>
                          <p className="text-sm text-[var(--text-secondary)]">{row.neuro}</p>
                        </div>
                        <div className="glass-card p-3">
                          <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Somatosensory</p>
                          <p className="text-sm text-[var(--text-secondary)]">{row.cervical}</p>
                        </div>
                        <div className="glass-card p-3">
                          <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Clinical Criteria</p>
                          <p className="text-sm text-[var(--text-secondary)]">{row.clinical}</p>
                        </div>
                        <div className="glass-card p-3">
                          <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">PSYDEFCONV Level</p>
                          <p className="text-sm text-[var(--text-secondary)]">{row.defense}</p>
                        </div>
                        <div className="glass-card p-3">
                          <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Activity Reserve</p>
                          <p className="text-sm text-[var(--text-secondary)]">{row.activity}</p>
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
