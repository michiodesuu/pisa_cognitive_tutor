"use client";
import type { ReliabilityData } from "@/lib/types";
import clsx from "clsx";

const KAPPA_COLOR = (k: number | null) => {
  if (k == null) return "text-gray-400";
  if (k >= 0.8)  return "text-green-600 font-semibold";
  if (k >= 0.6)  return "text-green-500";
  if (k >= 0.4)  return "text-yellow-600";
  if (k >= 0.2)  return "text-orange-500";
  return "text-red-500";
};

const DIM_LABEL: Record<string, string> = {
  C1:"Content Accuracy",C2:"Concept Depth",C3:"Data Interpretation",C4:"Causal Reasoning",
  C5:"Evidence Evaluation",C6:"Model Thinking",C7:"Systems/Transfer",C8:"Metacognition",
};

interface Props { data: ReliabilityData; }

export default function ReliabilityTable({ data }: Props) {
  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-100 flex items-baseline justify-between">
        <h2 className="font-semibold text-gray-800">Inter-rater Reliability</h2>
        <span className="text-xs text-gray-400">{data.n_turns} turns scored</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 text-left">
              <th className="px-6 py-3 text-xs font-medium text-gray-500 uppercase tracking-wide">Dimension</th>
              <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wide">Fleiss κ</th>
              <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wide">Interpretation</th>
              <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wide">Krippendorff α</th>
              <th className="px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wide">Consensus %</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {Object.entries(data.reliability || {}).map(([dim, r]) => (
              <tr key={dim} className="hover:bg-gray-50">
                <td className="px-6 py-3 font-medium text-gray-800">
                  <span className="text-xs font-mono bg-gray-100 px-1.5 py-0.5 rounded mr-2">{dim}</span>
                  {DIM_LABEL[dim] || dim}
                </td>
                <td className={clsx("px-4 py-3 font-mono", KAPPA_COLOR(r.fleiss_kappa))}>
                  {r.fleiss_kappa != null ? r.fleiss_kappa.toFixed(3) : "—"}
                </td>
                <td className="px-4 py-3 text-gray-600 capitalize">{r.interpretation}</td>
                <td className="px-4 py-3 font-mono text-gray-600">
                  {r.krippendorff_alpha != null ? r.krippendorff_alpha.toFixed(3) : "—"}
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-1.5 bg-gray-100 rounded-full">
                      <div
                        className="h-1.5 bg-brand-500 rounded-full"
                        style={{ width: `${(r.pct_consensus || 0) * 100}%` }}
                      />
                    </div>
                    <span className="text-gray-600">{((r.pct_consensus || 0)*100).toFixed(0)}%</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}