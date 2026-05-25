"use client";
import { useEffect, useRef, useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus, prism } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Brain, Eye, Zap, CheckCircle, AlertCircle, Play, ChevronRight } from "lucide-react";

// ── Task Definitions ──────────────────────────────────────────────────────────
const TASKS = [
  {
    id: 1,
    title: "Recursive State Refactoring",
    targets: ["C6 · Working Memory", "C4 · Cognitive Flexibility"],
    colors: ["#3b82f6", "#22c55e"],
    telemetry: "Monitor mean pupil diameter — a sustained increase >15% during tracing of `updated_weight` signals working memory overload.",
    icpBefore: 5,
    icpAfter: 1,
    icon: Brain,
    glowColor: "#3b82f6",
    objective:
      "Refactor the highly nested recursive function to eliminate progressive temporal memory accumulation. Reduce Intrinsic Complexity Points from 5 to 1 using an iterative, constant-memory state tracker.",
    prompt: "Refactor `process_nested_state` to eliminate deep recursion and progressive state accumulation. Use iterative O(1) memory.",
    beforeCode: `# Intrinsic Complexity: Deep recursion + progressive state accumulation
# Each frame forces retrieval of previous state → working memory overload

def process_nested_state(node_list, depth, accumulated_weight):
    if not node_list or depth == 0:  # +1 ICP (Branch)
        return accumulated_weight
    else:                            # +1 ICP (Branch)
        current_node = node_list[0]
        # Cumulative temporal dependency increases cognitive strain
        updated_weight = (accumulated_weight * current_node['multiplier']) + current_node['bias']
        if updated_weight > 1000:    # +1 ICP (Branch inside Branch)
            updated_weight = updated_weight * 0.9
        return process_nested_state(node_list[1:], depth - 1, updated_weight)`,
    afterCode: `def process_nested_state_iterative(node_list, depth, accumulated_weight):
    # Iterative state tracking eliminates frame accumulation
    # Minimizes working memory load: ICP = 1 (linear control flow)
    weight = accumulated_weight
    limit = min(len(node_list), depth)
    for i in range(limit):  # ICP = 1 (single linear loop)
        node = node_list[i]
        weight = (weight * node['multiplier']) + node['bias']
        if weight > 1000:
            weight *= 0.9
    return weight`,
    lang: "python",
  },
  {
    id: 2,
    title: "Distractor-Heavy Logic Debugging",
    targets: ["C1 · Sustained Attention", "C2 · Response Inhibition"],
    colors: ["#ef4444", "#f97316"],
    telemetry: "Monitor eye-movement saccades & fixation duration — regressions to warning comments or decoy blocks indicate response inhibition failure.",
    icpBefore: 0,
    icpAfter: 0,
    icon: Eye,
    glowColor: "#ef4444",
    objective:
      "Trace the value of `control_register` for input x=5. Ignore all visual and programmatic decoy instructions that do not directly alter the final returned state.",
    prompt: "What is the final return value of process_logic_stream(5)? Ignore all DECOY comments and functions.",
    answer: "22",
    explanation: "control_register starts at 12. x=5 < 10, so: 12 + (5*2) = 12 + 10 = 22. The decoy call returns a value but it's not used. Final: 22.",
    beforeCode: `// DECOY: System critical warning - do not trace loop depletion events!
// DECOY: Memory stack overflow imminent in thread 0x08A

function log_system_telemetry_decoy(state) {
    let internal_decoy = Math.sin(state) * 45;
    return internal_decoy;
}

function process_logic_stream(x) {
    let control_register = 12;
    log_system_telemetry_decoy(x); // Decoy call — return value discarded

    if (x < 10) {                  // Active execution path
        control_register += (x * 2);
    } else {
        // Decoy path — only reached when x >= 10
        control_register = 0;
        for (let i = 0; i < 1000; i++) {
            control_register += log_system_telemetry_decoy(i);
        }
    }
    // DECOY: Resetting control register to baseline is scheduled here
    return control_register;
}`,
    lang: "javascript",
  },
  {
    id: 3,
    title: "Algorithmic Pattern Synthesis",
    targets: ["C8 · Pattern Recognition", "C7 · Category Formation"],
    colors: ["#ec4899", "#8b5cf6"],
    telemetry: "Track pupillary hippus and HRV — the transition from search to recognition shows brief dilation followed by cardiac deceleration (PSI drop).",
    icpBefore: 0,
    icpAfter: 0,
    icon: Zap,
    glowColor: "#8b5cf6",
    objective:
      "Analyze the execution pattern of algorithm_alpha and algorithm_beta. Categorize their time complexities, then synthesize a unified iterative solution with O(N) time and O(1) space.",
    prompt: "Categorize time complexity of alpha and beta, then write algorithm_synthesized() solving the same recurrence in O(N) time, O(1) space.",
    beforeCode: `# Alpha: Solves recurrence relation recursively → O(2^N) complexity
def algorithm_alpha(n):
    if n == 0: return 0
    if n == 1: return 1
    return algorithm_alpha(n-1) + algorithm_alpha(n-2)

# Beta: Memoization solves redundancy → O(N) time, O(N) auxiliary space
def algorithm_beta(n, memo={}):
    if n in memo: return memo[n]
    if n == 0: return 0
    if n == 1: return 1
    memo[n] = algorithm_beta(n-1, memo) + algorithm_beta(n-2, memo)
    return memo[n]`,
    afterCode: `def algorithm_synthesized(n):
    # Linear time O(N), constant space O(1)
    # Pattern: Fibonacci recurrence F(n) = F(n-1) + F(n-2)
    # Complexity categories:
    #   alpha → O(2^N)  exponential (no memoization)
    #   beta  → O(N)    linear time, O(N) space
    #   this  → O(N)    linear time, O(1) space (optimal)
    if n == 0: return 0
    prev_2, prev_1 = 0, 1
    for _ in range(2, n + 1):
        current = prev_1 + prev_2
        prev_2 = prev_1
        prev_1 = current
    return prev_1`,
    lang: "python",
  },
];

// ── ICP Badge ─────────────────────────────────────────────────────────────────
function IcpBadge({ value, label }: { value: number; label: string }) {
  const colors = [
    "border-green-500/40 text-green-500 bg-green-500/10",
    "border-yellow-500/40 text-yellow-500 bg-yellow-500/10",
    "border-orange-500/40 text-orange-500 bg-orange-500/10",
    "border-red-500/40 text-red-500 bg-red-500/10",
  ];
  const cls = value === 0 ? colors[0] : value <= 2 ? colors[1] : value <= 4 ? colors[2] : colors[3];
  return (
    <div className="flex flex-col items-center">
      <span className={`text-xl font-bold font-mono px-3 py-1 rounded-lg border ${cls}`}>{value}</span>
      <span className="text-[10px] text-[var(--text-muted)] mt-1">{label}</span>
    </div>
  );
}

// ── Code Panel ────────────────────────────────────────────────────────────────
function CodePanel({ code, lang, label, color }: { code: string; lang: string; label: string; color: string }) {
  const isDark = typeof document !== "undefined" && document.documentElement.classList.contains("dark");
  return (
    <div className="rounded-xl overflow-hidden border border-[var(--border-color)]">
      <div className="flex items-center gap-2 px-4 py-2.5 bg-[var(--bg-tertiary)] border-b border-[var(--border-color)]">
        <div className="flex gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-red-400" />
          <span className="w-2.5 h-2.5 rounded-full bg-yellow-400" />
          <span className="w-2.5 h-2.5 rounded-full bg-green-400" />
        </div>
        <span className="text-xs font-mono text-[var(--text-muted)] ml-2" style={{ color }}>{label}</span>
      </div>
      <SyntaxHighlighter
        language={lang}
        style={vscDarkPlus}
        customStyle={{
          margin: 0,
          borderRadius: 0,
          fontSize: "0.78rem",
          lineHeight: "1.6",
          background: "#0d1117",
          padding: "1rem",
        }}
        showLineNumbers
        lineNumberStyle={{ color: "#444", fontSize: "0.7rem" }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function CodeTaskAssessor() {
  const [activeTask, setActiveTask] = useState(0);
  const [showAnswer, setShowAnswer] = useState(false);
  const [userAnswer, setUserAnswer] = useState("");
  const [gradeResult, setGradeResult] = useState<"correct" | "incorrect" | null>(null);

  const task = TASKS[activeTask];
  const Icon = task.icon;

  const checkAnswer = () => {
    if (!task.answer) return;
    const clean = userAnswer.trim().replace(/\s/g, "");
    setGradeResult(clean === task.answer ? "correct" : "incorrect");
  };

  const reset = () => {
    setShowAnswer(false);
    setUserAnswer("");
    setGradeResult(null);
  };

  return (
    <div className="space-y-4">
      {/* Task Tabs */}
      <div className="flex gap-2 flex-wrap">
        {TASKS.map((t, i) => {
          const TIcon = t.icon;
          return (
            <button
              key={t.id}
              onClick={() => { setActiveTask(i); reset(); }}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl border text-sm font-medium transition-all duration-200
                ${activeTask === i
                  ? "text-white border-transparent shadow-lg"
                  : "border-[var(--border-color)] text-[var(--text-secondary)] hover:border-[var(--neuro-cyan)] bg-[var(--bg-secondary)]"
                }`}
              style={activeTask === i ? { background: task.glowColor, boxShadow: `0 0 20px ${task.glowColor}40` } : {}}
            >
              <TIcon size={14} />
              Task {t.id}
            </button>
          );
        })}
      </div>

      {/* Task Card */}
      <div
        className="glass-card p-5 transition-all duration-300"
        style={{ borderColor: `${task.glowColor}25`, boxShadow: `0 0 30px ${task.glowColor}0a` }}
      >
        {/* Header */}
        <div className="flex items-start gap-3 mb-4">
          <div
            className="p-2 rounded-xl flex-shrink-0"
            style={{ background: `${task.glowColor}15`, border: `1px solid ${task.glowColor}30` }}
          >
            <Icon size={20} style={{ color: task.glowColor }} />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="font-bold text-[var(--text-primary)] text-base">
              Task {task.id}: {task.title}
            </h3>
            <div className="flex flex-wrap gap-1.5 mt-1.5">
              {task.targets.map((t, i) => (
                <span
                  key={t}
                  className="text-xs px-2.5 py-0.5 rounded-full font-semibold border"
                  style={{ color: task.colors[i], background: `${task.colors[i]}10`, borderColor: `${task.colors[i]}30` }}
                >
                  {t}
                </span>
              ))}
            </div>
          </div>
          {task.icpBefore > 0 && (
            <div className="flex items-center gap-3 flex-shrink-0">
              <IcpBadge value={task.icpBefore} label="ICP Before" />
              <ChevronRight size={16} className="text-[var(--text-muted)]" />
              <IcpBadge value={task.icpAfter} label="ICP After" />
            </div>
          )}
        </div>

        {/* Objective */}
        <div className="p-3 rounded-xl bg-[var(--bg-tertiary)] border border-[var(--border-color)] mb-4">
          <p className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-1">Objective</p>
          <p className="text-sm text-[var(--text-secondary)] leading-relaxed">{task.objective}</p>
        </div>

        {/* Code Panels */}
        <div className={`grid gap-4 ${task.afterCode ? "md:grid-cols-2" : "grid-cols-1"}`}>
          <CodePanel
            code={task.beforeCode}
            lang={task.lang}
            label={task.afterCode ? "⚠ Original Code (High ICP)" : "📋 Code Under Analysis"}
            color={task.glowColor}
          />
          {task.afterCode && (
            <CodePanel
              code={task.afterCode}
              lang={task.lang}
              label="✓ Expected Refactored Output (ICP = 1)"
              color="#22c55e"
            />
          )}
        </div>

        {/* Answer input for Task 2 */}
        {task.answer && (
          <div className="mt-4 space-y-3">
            <div className="flex gap-2">
              <input
                type="text"
                value={userAnswer}
                onChange={(e) => { setUserAnswer(e.target.value); setGradeResult(null); }}
                onKeyDown={(e) => e.key === "Enter" && checkAnswer()}
                placeholder="Enter the return value of process_logic_stream(5)…"
                className="flex-1 px-4 py-2.5 rounded-xl border border-[var(--border-color)] bg-[var(--bg-tertiary)]
                           text-sm text-[var(--text-primary)] font-mono placeholder-[var(--text-muted)]
                           focus:outline-none focus:ring-2 focus:ring-[var(--neuro-cyan)] focus:border-transparent"
              />
              <button
                onClick={checkAnswer}
                className="px-4 py-2.5 rounded-xl text-sm font-medium text-white transition-all"
                style={{ background: task.glowColor, boxShadow: `0 0 12px ${task.glowColor}40` }}
              >
                <Play size={14} className="inline mr-1" />Grade
              </button>
            </div>
            {gradeResult && (
              <div
                className={`flex items-start gap-2 p-3 rounded-xl border text-sm ${
                  gradeResult === "correct"
                    ? "border-green-500/30 bg-green-500/10 text-green-600 dark:text-green-400"
                    : "border-red-500/30 bg-red-500/10 text-red-600 dark:text-red-400"
                }`}
              >
                {gradeResult === "correct" ? <CheckCircle size={16} className="mt-0.5 flex-shrink-0" /> : <AlertCircle size={16} className="mt-0.5 flex-shrink-0" />}
                <div>
                  <p className="font-semibold">{gradeResult === "correct" ? "Correct! C1+C2 intact." : `Incorrect. Answer: ${task.answer}`}</p>
                  <p className="text-xs mt-0.5 opacity-80">{task.explanation}</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Telemetry */}
        <div
          className="mt-4 p-3 rounded-xl border"
          style={{ background: `${task.glowColor}07`, borderColor: `${task.glowColor}25` }}
        >
          <p className="text-[10px] font-semibold uppercase tracking-wider mb-1" style={{ color: task.glowColor }}>
            🔬 Telemetry Trigger
          </p>
          <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{task.telemetry}</p>
        </div>
      </div>
    </div>
  );
}
