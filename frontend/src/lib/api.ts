const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
const WS_URL  = process.env.NEXT_PUBLIC_WS_URL  || "ws://localhost:8001";

export async function createSession(questionTopic = "") {
  const res = await fetch(`${API_URL}/api/session/new`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question_topic: questionTopic }),
  });
  if (!res.ok) throw new Error("Failed to create session");
  return res.json() as Promise<{ session_id: string; user_id: string }>;
}

export async function getHistory(sessionId: string) {
  const res = await fetch(`${API_URL}/api/session/${sessionId}/history`);
  return res.json();
}

export async function getProfiles() {
  const res = await fetch(`${API_URL}/api/profiles`);
  return res.json();
}

export async function getReliability() {
  const res = await fetch(`${API_URL}/api/metrics/reliability`);
  return res.json();
}

export async function triggerAnalyze() {
  const res = await fetch(`${API_URL}/api/analyze`, { method: "POST" });
  return res.json();
}

export function createChatWebSocket(sessionId: string): WebSocket {
  return new WebSocket(`${WS_URL}/api/chat/${sessionId}`);
}

export async function uploadFile(
  sessionId: string,
  file: File,
  onProgress?: (pct: number) => void
): Promise<{
  file_id: string;
  filename: string;
  file_type: string;
  description_preview: string;
}> {
  const form = new FormData();
  form.append("file", file);

  // Use XMLHttpRequest to get upload progress
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new Error(`Upload failed: ${xhr.status} ${xhr.responseText}`));
      }
    };
    xhr.onerror = () => reject(new Error("Upload network error"));

    xhr.open("POST", `${API_URL}/api/upload/${sessionId}`);
    xhr.send(form);
  });
}