"use client";
import type { CognitiveProfile } from "@/lib/types";
import clsx from "clsx";

const DIM_COLORS: Record<string, string> = {
  C1: "#ef4444", C2: "#f97316", C3: "#eab308", C4: "#22c55e",
  C5: "#14b8a6", C6: "#3b82f6", C7: "#8b5cf6", C8: "#ec4899",
};

const ICAP_STYLE: Record<string, { border: string; bg: string; text: string }> = {
  Passive:      { border: "border-red-500/30",    bg: "bg-red-500/10",    text: "text-red-500" },
  Active:       { border: "border-orange-500/30", bg: "bg-orange-500/10", text: "text-orange-500" },
  Constructive: { border: "border-green-500/30",  bg: "bg-green-500/10",  text: "text-green-500" },
  Interactive:  { border: "border-blue-500/30",   bg: "bg-blue-500/10",   text: "text-blue-400" },
};

const TRAJ_ICON: Record<string, { icon: string; color: string }> = {
  improving: { icon: "↑", color: "text-green-500" },
  declining: { icon: "↓", color: "text-red-500" },
  stable:    { icon: "→", color: "text-yellow-500" },
  unknown:   { icon: "?", color: "text-[var(--text-muted)]" },
};

const DIMS = ["C1","C2","C3","C4","C5","C6","C7","C8"];
const DIM_SHORT: Record<string, string> = {
  C1: "Accuracy", C2: "Depth",    C3: "Data",     C4: "Causal",
  C5: "Evidence", C6: "Model",    C7: "Transfer",  C8: "Meta",
};
const DIM_NEURO: Record<string, string> = {
  C1: "Cingulate (58%)", C2: "Medial Frontal", C3: "Superior Frontal",
  C4: "Cingulate (40%)", C5: "Frontoparietal", C6: "PFC-Parietal",
  C7: "Cingulate (25%)", C8: "Cerebellar loop",
};

function ScoreBar({ dominant, max, color }: { dominant: number | "NA"; max: number | "NA"; color: string }) {
  const d = typeof dominant === "number" ? dominant : -1;
  const m = typeof max === "number" ? max : -1;
  return (
    <div className="relative h-1.5 bg-[var(--border-color)] rounded-full overflow-hidden">
      {m >= 0 && (
        <div
          className="absolute inset-y-0 left-0 rounded-full opacity-25"
          style={{ width: `${(m / 2) * 100}%`, background: color }}
        />
      )}
      {d >= 0 && (
        <div
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ width: `${(d / 2) * 100}%`, background: color, boxShadow: `0 0 6px ${color}60` }}
        />
      )}
    </div>
  );
}

interface Props { profile: CognitiveProfile; }

export default function ProfileCard({ profile }: Props) {
  const icapStyle = ICAP_STYLE[profile.dominant_icap] || ICAP_STYLE.Passive;
  const traj = TRAJ_ICON[profile.engagement_trajectory] || TRAJ_ICON.unknown;

  return (
    <div className="glass-card p-5 hover:scale-[1.01] transition-all duration-200">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="min-w-0">
          <p className="font-mono text-xs text-[var(--text-muted)] truncate">{profile.user_id}</p>
          <p className="text-sm text-[var(--text-secondary)] mt-0.5">{profile.n_turns_scored} turns scored</p>
        </div>
        <div className="flex flex-col items-end gap-1.5 flex-shrink-0">
          <span
            className={clsx(
              "px-2.5 py-0.5 rounded-full text-xs font-bold border",
              icapStyle.border, icapStyle.bg, icapStyle.text
            )}
          >
            {profile.dominant_icap}
          </span>
          <span className={clsx("text-xs font-semibold", traj.color)}>
            {traj.icon} {profile.engagement_trajectory}
          </span>
        </div>
      </div>

      {/* Dimension bars */}
      <div className="space-y-2.5">
        {DIMS.map((dim) => {
          const dp = profile.dimension_profile[dim];
          if (!dp) return null;
          const color = DIM_COLORS[dim] || "#94a3b8";
          return (
            <div key={dim}>
              <div className="flex justify-between items-end text-xs mb-1">
                <div>
                  <span className="font-mono font-bold" style={{ color }}>{dim}</span>
                  <span className="text-[10px] text-[var(--text-muted)] ml-1.5">{DIM_SHORT[dim]}</span>
                  <span className="text-[9px] text-[var(--text-muted)] ml-1 opacity-60">· {DIM_NEURO[dim]}</span>
                </div>
                <span className="text-[var(--text-secondary)] font-mono text-[10px]">
                  {dp.dominant_score} → {dp.max_capability}
                </span>
              </div>
              <ScoreBar dominant={dp.dominant_score} max={dp.max_capability} color={color} />
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="mt-4 pt-3 border-t border-[var(--border-color)] flex justify-between text-[10px] text-[var(--text-muted)]">
        <span>Avg: {profile.avg_duration_sec}s</span>
        {profile.duration_causal_correlation != null && (
          <span>ρ(dur,C4)={profile.duration_causal_correlation.toFixed(2)}</span>
        )}
        <span>Peak: {profile.max_icap_achieved}</span>
      </div>
    </div>
  );
}