"use client";
import { useEffect, useState } from "react";

interface Props {
  value?: number; // 0-100
  label?: string;
}

export default function KedeMeter({ value = 20, label = "KEDE" }: Props) {
  const [animated, setAnimated] = useState(0);

  useEffect(() => {
    const t = setTimeout(() => setAnimated(value), 300);
    return () => clearTimeout(t);
  }, [value]);

  // Color zones
  const getColor = (v: number) => {
    if (v < 20) return "#ef4444"; // red — near omniscience gap
    if (v < 40) return "#f97316"; // orange — low
    if (v < 60) return "#eab308"; // yellow — medium
    if (v < 80) return "#22c55e"; // green — good
    return "#3b82f6";             // blue — expert
  };

  const color = getColor(animated);
  const r = 52;
  const circumference = 2 * Math.PI * r;
  // Only show 270 degrees (three-quarters gauge)
  const arcLength = circumference * 0.75;
  const offset = arcLength - (animated / 100) * arcLength;

  const getZoneLabel = (v: number) => {
    if (v < 20) return "Critical Gap";
    if (v < 40) return "High Perplexity";
    if (v < 60) return "Moderate";
    if (v < 80) return "Expert Baseline";
    return "Near Omniscient";
  };

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative w-36 h-36">
        <svg viewBox="0 0 120 120" className="w-36 h-36">
          {/* Track arc — 270° starting from 135° */}
          <circle
            cx="60" cy="60" r={r}
            fill="none"
            stroke="rgba(148,163,184,0.12)"
            strokeWidth="10"
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeDashoffset={0}
            strokeLinecap="round"
            transform="rotate(135 60 60)"
          />
          {/* Value arc */}
          <circle
            cx="60" cy="60" r={r}
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeDashoffset={offset}
            strokeLinecap="round"
            transform="rotate(135 60 60)"
            style={{
              transition: "stroke-dashoffset 1.4s cubic-bezier(0.34, 1.56, 0.64, 1), stroke 0.5s ease",
              filter: `drop-shadow(0 0 8px ${color}80)`,
            }}
          />
        </svg>

        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-bold font-mono" style={{ color }}>
            {animated}
          </span>
          <span className="text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider">
            {label}
          </span>
        </div>

        {/* Min/Max labels */}
        <span className="absolute bottom-1 left-2 text-[9px] text-[var(--text-muted)] font-mono">0</span>
        <span className="absolute bottom-1 right-2 text-[9px] text-[var(--text-muted)] font-mono">100</span>
      </div>

      {/* Zone label */}
      <div
        className="px-3 py-1 rounded-full text-xs font-semibold border"
        style={{ color, background: `${color}10`, borderColor: `${color}30` }}
      >
        {getZoneLabel(animated)}
      </div>

      {/* Expert baseline marker */}
      <p className="text-[10px] text-[var(--text-muted)] text-center leading-relaxed max-w-[140px]">
        Expert baseline ≈ 20 · Omniscience = 100 · Novel discovery ≈ 0
      </p>
    </div>
  );
}
