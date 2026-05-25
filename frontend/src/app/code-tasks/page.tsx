"use client";
import { useState } from "react";
import CodeTaskAssessor from "@/components/CodeTaskAssessor";
import KedeMeter from "@/components/KedeMeter";
import { Brain, Eye, Zap, Activity, Info } from "lucide-react";

const TASK_KEDE = [38, 22, 45]; // Simulated KEDE per task (higher = easier/more familiar)
const TASK_ICP  = [5, 0, 0];    // ICP before

// Simulated pupil pulse animation
function PupilSimulator({ active, color }: { active: boolean; color: string }) {
  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-20 h-20 flex items-center justify-center">
        {/* Outer rings */}
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="absolute rounded-full border"
            style={{
              width: `${20 + i * 18}px`,
              height: `${20 + i * 18}px`,
              borderColor: `${color}${active ? Math.round(60 / i).toString(16).padStart(2, "0") : "15"}`,
              animation: active ? `ping ${0.8 + i * 0.3}s ease-in-out ${i * 0.2}s infinite` : "none",
              opacity: active ? 1 : 0.2,
            }}
          />
        ))}
        {/* Pupil */}
        <div
          className="rounded-full transition-all duration-1000 flex items-center justify-center"
          style={{
            width: active ? "22px" : "14px",
            height: active ? "22px" : "14px",
            background: `radial-gradient(circle, #1a1a2e 60%, ${color}40 100%)`,
            border: `2px solid ${color}60`,
            boxShadow: active ? `0 0 20px ${color}60` : "none",
          }}
        />
      </div>
      <p className="text-[10px] text-[var(--text-muted)] text-center">
        {active ? (
          <span style={{ color }}>⚡ Overload detected</span>
        ) : (
          <span>Baseline</span>
        )}
      </p>
    </div>
  );
}

export default function CodeTasksPage() {
  const [activeTask, setActiveTask] = useState(0);
  const [pupilActive, setPupilActive] = useState(false);

  const taskColors = ["#3b82f6", "#ef4444", "#8b5cf6"];
  const taskIcons = [Brain, Eye, Zap];
  const taskLabels = ["Task 1 · C6+C4", "Task 2 · C1+C2", "Task 3 · C8+C7"];

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] transition-theme">

      {/* Hero */}
      <div className="relative overflow-hidden border-b border-[var(--border-color)]">
        <div className="absolute inset-0 neuro-bg-pattern" />
        <div className="relative max-w-6xl mx-auto px-6 py-12">
          <div className="flex items-center gap-2 mb-3">
            <span className="badge-cyan">Code-Integrated Assessment</span>
            <span className="badge-purple">Telemetry-Linked</span>
          </div>
          <h1 className="text-4xl font-black font-display text-[var(--text-primary)] mb-3">
            Cognitive Assessment{" "}
            <span className="gradient-text">Code Tasks</span>
          </h1>
          <p className="text-[var(--text-secondary)] text-base max-w-2xl leading-relaxed">
            Three standardized, code-integrated assessment modules targeting specific C1–C8
            cognitive capacities, incorporating real-time physiological telemetry triggers
            (pupillometry, eye tracking, HRV).
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-[1fr_260px] gap-6">

          {/* Main: Code Assessor */}
          <div className="space-y-4">
            <CodeTaskAssessor />
          </div>

          {/* Sidebar: Metrics */}
          <div className="space-y-4">

            {/* KEDE Meter */}
            <div className="glass-card p-5 flex flex-col items-center gap-4">
              <div className="flex items-center gap-2 self-start">
                <Activity size={14} className="text-neuro-500" />
                <p className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
                  Task KEDE Score
                </p>
              </div>
              <KedeMeter value={TASK_KEDE[activeTask]} label="KEDE" />
              <div className="w-full space-y-1.5">
                {taskLabels.map((label, i) => (
                  <button
                    key={i}
                    onClick={() => setActiveTask(i)}
                    className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium transition-all border
                      ${activeTask === i
                        ? "border-transparent text-white"
                        : "border-[var(--border-color)] text-[var(--text-muted)] hover:border-[var(--neuro-cyan)]"
                      }`}
                    style={activeTask === i ? { background: taskColors[i], boxShadow: `0 0 12px ${taskColors[i]}40` } : {}}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* Simulated Pupilometry */}
            <div className="glass-card p-5 space-y-3">
              <div className="flex items-center gap-2">
                <Eye size={14} className="text-purple-500" />
                <p className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
                  Pupillometry Sim
                </p>
              </div>
              <div className="flex justify-center">
                <PupilSimulator active={pupilActive} color={taskColors[activeTask]} />
              </div>
              <button
                onClick={() => { setPupilActive(true); setTimeout(() => setPupilActive(false), 3000); }}
                className="w-full py-2 rounded-xl text-xs font-medium border border-[var(--border-color)]
                           text-[var(--text-secondary)] hover:border-[var(--neuro-cyan)] transition-all"
              >
                Simulate Overload Response
              </button>
            </div>

            {/* ICP Info */}
            <div className="glass-card p-4 space-y-2">
              <div className="flex items-center gap-2">
                <Info size={14} className="text-[var(--text-muted)]" />
                <p className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
                  ICP Legend
                </p>
              </div>
              {[
                { range: "0", color: "#22c55e", label: "Optimal / Iterative" },
                { range: "1–2", color: "#eab308", label: "Moderate cognitive load" },
                { range: "3–4", color: "#f97316", label: "High nested complexity" },
                { range: "5+", color: "#ef4444", label: "Working memory overload" },
              ].map((item) => (
                <div key={item.range} className="flex items-center gap-2">
                  <span
                    className="text-[10px] font-mono font-bold w-8 text-center px-1 py-0.5 rounded"
                    style={{ color: item.color, background: `${item.color}15` }}
                  >
                    {item.range}
                  </span>
                  <span className="text-[10px] text-[var(--text-muted)]">{item.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
