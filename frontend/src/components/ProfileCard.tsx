"use client";
import type { CognitiveProfile } from "@/lib/types";
import clsx from "clsx";

const ICAP_COLOR: Record<string, string> = {
  Passive:     "bg-red-100 text-red-700",
  Active:      "bg-orange-100 text-orange-700",
  Constructive:"bg-green-100 text-green-700",
  Interactive: "bg-blue-100 text-blue-700",
};

const TRAJ_COLOR: Record<string, string> = {
  improving: "text-green-600",
  declining: "text-red-500",
  stable:    "text-yellow-600",
  unknown:   "text-gray-400",
};

const TRAJ_ICON: Record<string, string> = {
  improving: "↑", declining: "↓", stable: "→", unknown: "?",
};

const DIMS = ["C1","C2","C3","C4","C5","C6","C7","C8"];
const DIM_SHORT: Record<string, string> = {
  C1:"Accuracy",C2:"Depth",C3:"Data",C4:"Causal",
  C5:"Evidence",C6:"Model",C7:"Transfer",C8:"Meta",
};

function ScoreBar({ dominant, max }: { dominant: number|"NA"; max: number|"NA" }) {
  const d = typeof dominant === "number" ? dominant : -1;
  const m = typeof max === "number" ? max : -1;
  return (
    <div className="relative h-2 bg-gray-100 rounded-full overflow-hidden">
      {m >= 0 && (
        <div
          className="absolute inset-y-0 left-0 bg-brand-200 rounded-full"
          style={{ width: `${(m / 2) * 100}%` }}
        />
      )}
      {d >= 0 && (
        <div
          className="absolute inset-y-0 left-0 bg-brand-500 rounded-full"
          style={{ width: `${(d / 2) * 100}%` }}
        />
      )}
    </div>
  );
}

interface Props { profile: CognitiveProfile; }

export default function ProfileCard({ profile }: Props) {
  return (
    <div className="bg-white rounded-2xl border border-gray-200 p-5 shadow-sm hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="font-mono text-xs text-gray-400">{profile.user_id}</p>
          <p className="text-sm text-gray-600 mt-0.5">{profile.n_turns_scored} turns scored</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span className={clsx(
            "px-2 py-0.5 rounded-full text-xs font-medium",
            ICAP_COLOR[profile.dominant_icap] || "bg-gray-100 text-gray-600"
          )}>
            {profile.dominant_icap}
          </span>
          <span className={clsx("text-xs font-medium", TRAJ_COLOR[profile.engagement_trajectory])}>
            {TRAJ_ICON[profile.engagement_trajectory]} {profile.engagement_trajectory}
          </span>
        </div>
      </div>

      {/* Dimension bars */}
      <div className="space-y-2">
        {DIMS.map((dim) => {
          const dp = profile.dimension_profile[dim];
          if (!dp) return null;
          return (
            <div key={dim}>
              <div className="flex justify-between text-xs mb-0.5">
                <span className="text-gray-500">{DIM_SHORT[dim]}</span>
                <span className="text-gray-700 font-medium">
                  {dp.dominant_score} → {dp.max_capability}
                </span>
              </div>
              <ScoreBar dominant={dp.dominant_score} max={dp.max_capability} />
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="mt-4 pt-3 border-t border-gray-100 flex justify-between text-xs text-gray-400">
        <span>Avg: {profile.avg_duration_sec}s</span>
        {profile.duration_causal_correlation != null && (
          <span>ρ(dur,C4)={profile.duration_causal_correlation.toFixed(2)}</span>
        )}
        <span>Peak: {profile.max_icap_achieved}</span>
      </div>
    </div>
  );
}