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

  return (
    <div className="flex h-full overflow-hidden">
      <Sidebar userId={userId} sessionId={sessionId!} />
      <main className="flex-1 flex flex-col overflow-hidden">
        <ChatWindow sessionId={sessionId!} userId={userId} />
      </main>
    </div>
  );
}