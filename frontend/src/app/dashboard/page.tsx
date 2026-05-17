"use client";
import { useEffect, useState } from "react";
import { getProfiles, getReliability, triggerAnalyze } from "@/lib/api";
import ProfileCard from "@/components/ProfileCard";
import ReliabilityTable from "@/components/ReliabilityTable";

export default function Dashboard() {
  const [profiles, setProfiles] = useState<Record<string, any>>({});
  const [reliability, setReliability] = useState<any>(null);
  const [analyzing, setAnalyzing] = useState(false);

  useEffect(() => {
    getProfiles().then((d) => setProfiles(d.profiles || {}));
    getReliability().then(setReliability);
  }, []);

  const handleAnalyze = async () => {
    setAnalyzing(true);
    await triggerAnalyze();
    setTimeout(() => {
      getProfiles().then((d) => setProfiles(d.profiles || {}));
      getReliability().then(setReliability);
      setAnalyzing(false);
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Research Dashboard</h1>
            <p className="text-gray-500 mt-1">
              Cognitive profile analysis — PISA Cognitive Tutor
            </p>
          </div>
          <button
            onClick={handleAnalyze}
            disabled={analyzing}
            className="px-5 py-2.5 bg-brand-600 text-white rounded-xl font-medium
                       hover:bg-brand-700 disabled:opacity-50 transition-colors"
          >
            {analyzing ? "Analyzing…" : "Run Analyzer"}
          </button>
        </div>

        {reliability && <ReliabilityTable data={reliability} />}

        <h2 className="text-xl font-semibold mt-10 mb-4 text-gray-800">
          Student Profiles ({Object.keys(profiles).length})
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {Object.values(profiles).map((p: any) => (
            <ProfileCard key={p.user_id} profile={p} />
          ))}
          {Object.keys(profiles).length === 0 && (
            <p className="text-gray-400 col-span-3">
              No profiles yet. Run a student session and then click "Run Analyzer".
            </p>
          )}
        </div>
      </div>
    </div>
  );
}