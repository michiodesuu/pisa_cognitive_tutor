"use client";
import { useEffect, useState } from "react";
import { getProfiles, getReliability, triggerAnalyze } from "@/lib/api";
import ProfileCard from "@/components/ProfileCard";
import ReliabilityTable from "@/components/ReliabilityTable";
import NeuroTaxonomyTable from "@/components/NeuroTaxonomyTable";
import { Play, Users, BarChart3, Brain, Clock } from "lucide-react";

interface StatCard {
  label: string;
  value: string | number;
  sub?: string;
  color: string;
  icon: React.ReactNode;
}

export default function Dashboard() {
  const [profiles, setProfiles] = useState<Record<string, any>>({});
  const [reliability, setReliability] = useState<any>(null);
  const [analyzing, setAnalyzing] = useState(false);

  useEffect(() => {
    getProfiles().then((d) => setProfiles(d?.profiles || {}));
    getReliability().then((d) => { if (d) setReliability(d); });
  }, []);

  const handleAnalyze = async () => {
    setAnalyzing(true);
    await triggerAnalyze();
    setTimeout(() => {
      getProfiles().then((d) => setProfiles(d?.profiles || {}));
      getReliability().then((d) => { if (d) setReliability(d); });
      setAnalyzing(false);
    }, 3000);
  };

  const profileList = Object.values(profiles);
  const nTurns = reliability?.n_turns ?? 0;
  const avgKappa =
    reliability?.reliability
      ? (
          Object.values(reliability.reliability)
            .map((r: any) => r.fleiss_kappa)
            .filter(Boolean)
            .reduce((a: number, b: number) => a + b, 0) /
          Object.values(reliability.reliability).filter((r: any) => r.fleiss_kappa).length
        ).toFixed(3)
      : "—";

  const statCards: StatCard[] = [
    {
      label: "Student Profiles",
      value: profileList.length,
      sub: "unique users",
      color: "#00b4cc",
      icon: <Users size={18} className="text-neuro-500" />,
    },
    {
      label: "Turns Scored",
      value: nTurns,
      sub: "C1–C8 assessed",
      color: "#a855f7",
      icon: <BarChart3 size={18} className="text-purple-500" />,
    },
    {
      label: "Avg Fleiss κ",
      value: avgKappa,
      sub: "inter-rater agreement",
      color: "#22c55e",
      icon: <Brain size={18} className="text-green-500" />,
    },
    {
      label: "Avg Response",
      value:
        profileList.length > 0
          ? `${(profileList.reduce((a, p) => a + (p.avg_duration_sec || 0), 0) / profileList.length).toFixed(1)}s`
          : "—",
      sub: "per turn",
      color: "#f97316",
      icon: <Clock size={18} className="text-orange-500" />,
    },
  ];

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] p-6 transition-theme">
      <div className="max-w-7xl mx-auto space-y-8">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-[var(--text-primary)] font-display">
              Research Dashboard
            </h1>
            <p className="text-[var(--text-secondary)] mt-1 text-sm">
              PISA Cognitive Tutor · C1–C8 ICAP Assessment Platform
            </p>
          </div>
          <button
            onClick={handleAnalyze}
            disabled={analyzing}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-white text-sm transition-all
                       disabled:opacity-50 disabled:cursor-not-allowed"
            style={{
              background: "linear-gradient(135deg, #00b4cc, #0090a8)",
              boxShadow: analyzing ? "none" : "0 0 20px rgba(0,180,204,0.3)",
            }}
          >
            <Play size={14} className={analyzing ? "animate-spin" : ""} />
            {analyzing ? "Analyzing…" : "Run Analyzer"}
          </button>
        </div>

        {/* Stat Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {statCards.map((stat) => (
            <div key={stat.label} className="glass-card p-4">
              <div className="flex items-center gap-2 mb-2">
                <div
                  className="p-1.5 rounded-lg"
                  style={{ background: `${stat.color}15`, border: `1px solid ${stat.color}25` }}
                >
                  {stat.icon}
                </div>
                <span className="text-xs text-[var(--text-muted)] font-medium">{stat.label}</span>
              </div>
              <p className="text-2xl font-bold font-mono" style={{ color: stat.color }}>
                {stat.value}
              </p>
              <p className="text-[10px] text-[var(--text-muted)] mt-0.5">{stat.sub}</p>
            </div>
          ))}
        </div>

        {/* Reliability */}
        {reliability && <ReliabilityTable data={reliability} />}

        {/* Mini Neuro Taxonomy */}
        <div>
          <h2 className="text-lg font-bold text-[var(--text-primary)] font-display mb-3">
            C1–C8 Neuro Taxonomy Overview
          </h2>
          <NeuroTaxonomyTable compact />
        </div>

        {/* Student Profiles */}
        <div>
          <h2 className="text-xl font-bold text-[var(--text-primary)] font-display mb-4">
            Student Profiles
            <span className="text-sm font-normal text-[var(--text-muted)] ml-2">
              ({profileList.length} users)
            </span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {profileList.map((p: any) => (
              <ProfileCard key={p.user_id} profile={p} />
            ))}
            {profileList.length === 0 && (
              <div className="col-span-3 glass-card p-10 text-center">
                <Brain size={40} className="mx-auto text-[var(--text-muted)] mb-3 opacity-30" />
                <p className="text-[var(--text-secondary)]">No profiles yet.</p>
                <p className="text-xs text-[var(--text-muted)] mt-1">
                  Run a student session, then click <strong>Run Analyzer</strong>.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}