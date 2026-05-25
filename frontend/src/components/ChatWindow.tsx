"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send, Wifi, WifiOff, Paperclip, X,
  FileText, Table2, FileImage, File,
} from "lucide-react";
import { createChatWebSocket, uploadFile } from "@/lib/api";
import type { ChatMessage, UploadedFile } from "@/lib/types";
import MessageBubble from "./MessageBubble";
import TypingIndicator from "./TypingIndicator";

interface Props { sessionId: string; userId: string; }

const ACCEPTED = ".png,.jpg,.jpeg,.webp,.gif,.pdf,.csv,.xls,.xlsx,.txt,.md";
const MAX_MB = 20;

function fileIcon(type: UploadedFile["file_type"]) {
  const cls = "w-4 h-4 flex-shrink-0";
  if (type === "image") return <FileImage className={cls} />;
  if (type === "table") return <Table2 className={cls} />;
  if (type === "pdf")   return <FileText className={cls} />;
  return <File className={cls} />;
}

export default function ChatWindow({ sessionId, userId }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Welcome! I am here to help you explore science questions step by step.\n\n" +
        "You can write in **Thai, English, or Chinese** — and you can also " +
        "**attach images, tables, or PDFs** using the 📎 button. " +
        "I will describe what I see and ask you questions about it. 🔬",
      timestamp: Date.now(),
    },
  ]);
  const [input, setInput]           = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [turnNumber, setTurnNumber]   = useState(0);
  const [uploads, setUploads]         = useState<UploadedFile[]>([]);
  const [isDragOver, setIsDragOver]   = useState(false);

  const wsRef       = useRef<WebSocket | null>(null);
  const bottomRef   = useRef<HTMLDivElement>(null);
  const inputRef    = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const startTimeRef = useRef<number>(Date.now());

  // ── WebSocket ──────────────────────────────────────────────────────────────
  useEffect(() => {
    const ws = createChatWebSocket(sessionId);
    wsRef.current = ws;
    ws.onopen  = () => setIsConnected(true);
    ws.onclose = () => setIsConnected(false);
    ws.onerror = () => setIsConnected(false);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "token") {
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last?.role === "assistant" && last.id === "streaming") {
            return [...prev.slice(0, -1), { ...last, content: last.content + data.content }];
          }
          return prev;
        });
      } else if (data.type === "done") {
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last?.id === "streaming") {
            return [...prev.slice(0, -1), {
              ...last, id: `msg_${Date.now()}`, kb_used: data.kb_used,
            }];
          }
          return prev;
        });
        setIsStreaming(false);
        startTimeRef.current = Date.now();
        inputRef.current?.focus();
      } else if (data.type === "error") {
        setIsStreaming(false);
      }
    };
    return () => ws.close();
  }, [sessionId]);

  // ── Auto-scroll ────────────────────────────────────────────────────────────
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── File handling ──────────────────────────────────────────────────────────
  const processFiles = useCallback(async (files: FileList | File[]) => {
    const arr = Array.from(files);
    for (const file of arr) {
      if (file.size > MAX_MB * 1024 * 1024) {
        alert(`"${file.name}" exceeds the ${MAX_MB} MB limit.`);
        continue;
      }

      const tempId = `temp_${Date.now()}_${Math.random()}`;
      const localPreviewUrl = file.type.startsWith("image/")
        ? URL.createObjectURL(file) : undefined;

      // Add placeholder with progress=0
      const placeholder: UploadedFile = {
        file_id: tempId,
        filename: file.name,
        file_type: file.type.startsWith("image/") ? "image"
          : file.name.match(/\.(csv|xls|xlsx)$/i) ? "table"
          : file.name.endsWith(".pdf") ? "pdf" : "text",
        description_preview: "",
        localPreviewUrl,
        uploadProgress: 0,
      };
      setUploads((prev) => [...prev, placeholder]);

      try {
        const result = await uploadFile(sessionId, file, (pct) => {
          setUploads((prev) =>
            prev.map((u) => u.file_id === tempId ? { ...u, uploadProgress: pct } : u)
          );
        });
        // Replace placeholder with real entry
        setUploads((prev) =>
          prev.map((u) =>
            u.file_id === tempId
              ? { ...result, localPreviewUrl, uploadProgress: undefined }
              : u
          )
        );
      } catch (e: any) {
        setUploads((prev) =>
          prev.map((u) =>
            u.file_id === tempId
              ? { ...u, uploadProgress: undefined, error: e.message }
              : u
          )
        );
      }
    }
  }, [sessionId]);

  const removeUpload = (fileId: string) => {
    setUploads((prev) => {
      const entry = prev.find((u) => u.file_id === fileId);
      if (entry?.localPreviewUrl) URL.revokeObjectURL(entry.localPreviewUrl);
      return prev.filter((u) => u.file_id !== fileId);
    });
  };

  // ── Drag and drop ──────────────────────────────────────────────────────────
  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault(); setIsDragOver(true);
  };
  const onDragLeave = () => setIsDragOver(false);
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setIsDragOver(false);
    if (e.dataTransfer.files.length) processFiles(e.dataTransfer.files);
  };

  // ── Send ───────────────────────────────────────────────────────────────────
  const sendMessage = useCallback(() => {
    const text = input.trim();
    const readyUploads = uploads.filter(
      (u) => u.uploadProgress === undefined && !u.error
    );
    if ((!text && readyUploads.length === 0) || isStreaming) return;
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    const duration_sec = (Date.now() - startTimeRef.current) / 1000;
    const newTurn = turnNumber + 1;
    setTurnNumber(newTurn);

    // Build display content for user bubble
    const displayContent = [
      text,
      readyUploads.length > 0
        ? `\n\n📎 _Attached: ${readyUploads.map((u) => u.filename).join(", ")}_`
        : "",
    ].join("").trim();

    setInput("");
    setUploads([]);

    const userMsg: ChatMessage = {
      id: `user_${Date.now()}`,
      role: "user",
      content: displayContent,
      timestamp: Date.now(),
      duration_sec,
    };
    const streamingMsg: ChatMessage = {
      id: "streaming",
      role: "assistant",
      content: "",
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg, streamingMsg]);
    setIsStreaming(true);

    wsRef.current.send(JSON.stringify({
      user_input: text || "(see attached files)",
      turn_number: newTurn,
      duration_sec,
      user_id: userId,
      file_ids: readyUploads.map((u) => u.file_id),  // ← passes file_ids to backend
    }));
  }, [input, isStreaming, turnNumber, userId, uploads]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault(); sendMessage();
    }
  };

  const pendingUploads = uploads.filter((u) => u.uploadProgress !== undefined);
  const readyUploads   = uploads.filter((u) => u.uploadProgress === undefined && !u.error);
  const canSend = (input.trim() || readyUploads.length > 0) && !isStreaming && isConnected;

  return (
    <div
      className="flex flex-col h-full"
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      {/* Drop overlay */}
      <AnimatePresence>
        {isDragOver && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 flex items-center justify-center
                       bg-neuro-500/10 border-4 border-dashed border-neuro-400/60 rounded-xl backdrop-blur-sm"
          >
            <p className="text-neuro-400 font-semibold text-xl">Drop files here</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <header className="flex items-center gap-3 px-6 py-4 border-b border-[var(--border-color)] bg-[var(--bg-secondary)] transition-theme">
        <div className="flex-1">
          <h1 className="font-semibold text-[var(--text-primary)] font-display">PISA Science Tutor</h1>
          <p className="text-xs text-[var(--text-muted)]">Turn {turnNumber} · {userId}</p>
        </div>
        <div className={`flex items-center gap-1.5 text-xs font-medium
          ${isConnected ? "text-neuro-500 dark:text-neuro-400" : "text-red-500"}`}>
          {isConnected ? <Wifi size={14} /> : <WifiOff size={14} />}
          {isConnected ? "Connected" : "Reconnecting…"}
        </div>
      </header>

      {/* ── Messages ──────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4 bg-[var(--bg-primary)] transition-theme">
        <AnimatePresence initial={false}>
          {messages.map((msg) => (
            <motion.div key={msg.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
            >
              {msg.id === "streaming" && msg.content === "" ? (
                <TypingIndicator />
              ) : (
                <MessageBubble message={msg} />
              )}
            </motion.div>
          ))}
        </AnimatePresence>
        <div ref={bottomRef} />
      </div>

      {/* ── Upload preview strip ───────────────────────────────────────────── */}
      <AnimatePresence>
        {uploads.length > 0 && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-[var(--border-color)] bg-[var(--bg-secondary)] px-4 py-2 overflow-hidden transition-theme"
          >
            <div className="flex flex-wrap gap-2 max-w-3xl mx-auto">
              {uploads.map((u) => (
                <div key={u.file_id}
                  className="relative flex items-center gap-2 pl-2 pr-7 py-1.5
                             bg-white border border-gray-200 rounded-lg text-xs
                             shadow-sm max-w-[200px]"
                >
                  {/* Image thumbnail */}
                  {u.localPreviewUrl ? (
                    <img
                      src={u.localPreviewUrl}
                      alt={u.filename}
                      className="w-8 h-8 rounded object-cover flex-shrink-0"
                    />
                  ) : (
                    <span className="text-gray-400">{fileIcon(u.file_type)}</span>
                  )}

                  <div className="flex-1 min-w-0">
                    <p className="truncate font-medium text-gray-700">{u.filename}</p>
                    {u.uploadProgress !== undefined ? (
                      <div className="mt-0.5 h-1 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-1 bg-brand-500 rounded-full transition-all"
                          style={{ width: `${u.uploadProgress}%` }}
                        />
                      </div>
                    ) : u.error ? (
                      <p className="text-red-500 truncate">{u.error}</p>
                    ) : (
                      <p className="text-green-600">Ready</p>
                    )}
                  </div>

                  {/* Remove button */}
                  {u.uploadProgress === undefined && (
                    <button
                      onClick={() => removeUpload(u.file_id)}
                      className="absolute right-1.5 top-1.5 text-gray-400
                                 hover:text-gray-700 transition-colors"
                    >
                      <X size={13} />
                    </button>
                  )}
                </div>
              ))}
            </div>

            {/* Description previews (collapsed, shown on hover) */}
            {readyUploads.some((u) => u.description_preview) && (
              <details className="max-w-3xl mx-auto mt-2">
                <summary className="text-xs text-gray-400 cursor-pointer hover:text-gray-600">
                  View AI description previews
                </summary>
                <div className="mt-1 space-y-1">
                  {readyUploads.filter((u) => u.description_preview).map((u) => (
                    <div key={u.file_id}
                      className="text-xs text-gray-600 bg-white border border-gray-100
                                 rounded p-2 font-mono leading-relaxed"
                    >
                      <span className="font-semibold text-gray-800">{u.filename}: </span>
                      {u.description_preview}
                    </div>
                  ))}
                </div>
              </details>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Input bar ─────────────────────────────────────────────────────── */}
      <div className="border-t border-[var(--border-color)] bg-[var(--bg-secondary)] px-4 py-4 transition-theme">
        <div className="flex items-end gap-2 max-w-3xl mx-auto">
          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED}
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.length) processFiles(e.target.files);
              e.target.value = ""; // allow same file re-upload
            }}
          />

          {/* Attach button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isStreaming || !isConnected}
            title="Attach image, PDF, CSV, or text"
            className="flex-shrink-0 flex items-center justify-center w-10 h-10
                       rounded-xl border border-[var(--border-color)] text-[var(--text-muted)]
                       hover:border-neuro-500/60 hover:text-neuro-500
                       disabled:opacity-40 disabled:cursor-not-allowed
                       transition-all"
          >
            <Paperclip size={17} />
          </button>

          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isStreaming || !isConnected}
            rows={1}
            placeholder={
              pendingUploads.length > 0
                ? "Uploading…"
                : "Type your answer or attach a file…"
            }
            className="flex-1 resize-none rounded-xl border border-[var(--border-color)] bg-[var(--bg-tertiary)]
                       px-4 py-3 text-sm text-[var(--text-primary)] placeholder-[var(--text-muted)]
                       focus:outline-none focus:ring-2 focus:ring-neuro-500
                       focus:border-transparent disabled:opacity-50 max-h-32 overflow-y-auto transition-theme"
            style={{ minHeight: "48px" }}
            onInput={(e) => {
              const el = e.currentTarget;
              el.style.height = "auto";
              el.style.height = `${Math.min(el.scrollHeight, 128)}px`;
            }}
          />

          <button
            onClick={sendMessage}
            disabled={!canSend || pendingUploads.length > 0}
            className="flex-shrink-0 flex items-center justify-center w-12 h-12
                       rounded-xl bg-neuro-500 text-white
                       hover:bg-neuro-400 disabled:opacity-40 disabled:cursor-not-allowed
                       transition-all shadow-[0_0_16px_rgba(0,180,204,0.3)]
                       hover:shadow-[0_0_24px_rgba(0,180,204,0.5)]"
          >
            <Send size={18} />
          </button>
        </div>

        <p className="text-center text-xs text-gray-400 mt-2">
          <kbd className="px-1 py-0.5 bg-gray-100 rounded text-xs">Enter</kbd> send ·
          <kbd className="px-1 py-0.5 bg-gray-100 rounded text-xs ml-1">Shift+Enter</kbd> newline ·
          <kbd className="px-1 py-0.5 bg-gray-100 rounded text-xs ml-1">📎</kbd> attach image / PDF / table
        </p>
      </div>
    </div>
  );
}