"use client";
import type { ReliabilityData } from "@/lib/types";
import clsx from "clsx";

const DIM_COLOR: Record<string, string> = {
  C1: "#ef4444", C2: "#f97316", C3: "#eab308", C4: "#22c55e",
  C5: "#14b8a6", C6: "#3b82f6", C7: "#8b5cf6", C8: "#ec4899",
};

const kappaLevel = (k: number | null) => {
  if (k == null) return { label: "—",          cls: "text-[var(--text-muted)]" };
  if (k >= 0.8)  return { label: "Near perfect", cls: "text-green-500 font-bold" };
  if (k >= 0.6)  return { label: "Substantial",  cls: "text-green-500" };
  if (k >= 0.4)  return { label: "Moderate",     cls: "text-yellow-500" };
  if (k >= 0.2)  return { label: "Fair",         cls: "text-orange-500" };
  return           { label: "Slight",           cls: "text-red-500" };
};

const DIM_LABEL: Record<string, string> = {
  C1: "Content Accuracy",    C2: "Concept Depth",       C3: "Data Interpretation",
  C4: "Causal Reasoning",    C5: "Evidence Evaluation", C6: "Model Thinking",
  C7: "Systems / Transfer",  C8: "Metacognition",
};

interface Props { data: ReliabilityData; }

export default function ReliabilityTable({ data }: Props) {
  return (
    <div className="glass-card overflow-hidden">
      <div className="px-6 py-4 border-b border-[var(--border-color)] flex items-center justify-between">
        <h2 className="font-bold text-[var(--text-primary)]">Inter-rater Reliability</h2>
        <span className="text-xs text-[var(--text-muted)]">{data.n_turns} turns scored</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-[var(--bg-tertiary)] border-b border-[var(--border-color)]">
              <th className="px-6 py-3 text-left text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Dimension</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Fleiss κ</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Level</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Krippendorff α</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Consensus</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-color)]">
            {Object.entries(data.reliability || {}).map(([dim, r]) => {
              const level = kappaLevel(r.fleiss_kappa);
              const color = DIM_COLOR[dim] || "#94a3b8";
              const pct = (r.pct_consensus || 0) * 100;
              return (
                <tr key={dim} className="hover:bg-[var(--bg-tertiary)] transition-colors">
                  <td className="px-6 py-3">
                    <div className="flex items-center gap-2">
                      <span
                        className="text-xs font-mono font-bold px-1.5 py-0.5 rounded"
                        style={{ color, background: `${color}15`, border: `1px solid ${color}30` }}
                      >
                        {dim}
                      </span>
                      <span className="text-[var(--text-secondary)] text-xs">{DIM_LABEL[dim] || dim}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 font-mono text-sm" style={{ color }}>
                    {r.fleiss_kappa != null ? r.fleiss_kappa.toFixed(3) : "—"}
                  </td>
                  <td className={clsx("px-4 py-3 text-xs", level.cls)}>{level.label}</td>
                  <td className="px-4 py-3 font-mono text-xs text-[var(--text-secondary)]">
                    {r.krippendorff_alpha != null ? r.krippendorff_alpha.toFixed(3) : "—"}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="w-20 h-1.5 bg-[var(--border-color)] rounded-full overflow-hidden">
                        <div
                          className="h-1.5 rounded-full transition-all duration-700"
                          style={{ width: `${pct}%`, background: color, boxShadow: `0 0 6px ${color}50` }}
                        />
                      </div>
                      <span className="text-xs text-[var(--text-secondary)]">{pct.toFixed(0)}%</span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}