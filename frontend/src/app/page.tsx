"use client";
import { useEffect, useState } from "react";
import ChatWindow from "@/components/ChatWindow";
import Sidebar from "@/components/Sidebar";
import { createSession } from "@/lib/api";

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [userId, setUserId] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [category, setCategory] = useState<string | null>(null);

  useEffect(() => {
    createSession()
      .then(({ session_id, user_id }) => {
        setSessionId(session_id);
        setUserId(user_id);
        setLoading(false);
      })
      .catch((e) => {
        setError("Cannot connect to the backend. Is the API running on port 8001?");
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="flex flex-col items-center gap-4 text-gray-500">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-brand-500 border-t-transparent" />
          <p className="text-sm">Starting tutor session…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="rounded-2xl bg-red-50 border border-red-200 p-8 max-w-md text-center">
          <p className="text-red-600 font-medium">{error}</p>
          <p className="text-sm text-gray-500 mt-2">
            Run: <code className="bg-gray-100 px-1 rounded">uvicorn src.api.main:app --port 8001</code>
          </p>
        </div>
      </div>
    );
  }

  if (!category) {
    const categories = ["General Science", "Biology", "Chemistry", "Physics", "Earth Science"];
    return (
      <div className="flex h-full items-center justify-center bg-gray-50">
        <div className="bg-white p-8 rounded-2xl shadow-sm border border-gray-100 max-w-md w-full text-center space-y-6">
          <h2 className="text-2xl font-bold text-gray-800">Select Subject</h2>
          <p className="text-gray-500">What would you like to explore today?</p>
          <div className="grid gap-3">
            {categories.map((c) => (
              <button
                key={c}
                onClick={() => setCategory(c)}
                className="w-full py-3 px-4 bg-gray-50 hover:bg-brand-50 hover:text-brand-600 border border-gray-200 hover:border-brand-200 rounded-xl transition-all font-medium text-gray-700 text-left flex items-center justify-between"
              >
                {c}
                <span className="text-brand-500 opacity-0 group-hover:opacity-100 transition-opacity">→</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full overflow-hidden">
      <Sidebar userId={userId} sessionId={sessionId!} />
      <main className="flex-1 flex flex-col overflow-hidden">
        <ChatWindow sessionId={sessionId!} userId={userId} category={category} />
      </main>
    </div>
  );
}