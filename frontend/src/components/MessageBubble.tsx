"use client";
import ReactMarkdown from "react-markdown";
import type { ChatMessage } from "@/lib/types";
import { BookOpen } from "lucide-react";

interface Props { message: ChatMessage; }

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"} items-end`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 text-xs font-bold
          ${isUser
            ? "bg-neuro-500/15 border border-neuro-500/30 text-neuro-400"
            : "bg-purple-500/15 border border-purple-500/30 text-purple-400"
          }`}
      >
        {isUser ? "You" : "AI"}
      </div>

      {/* Bubble */}
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed
          ${isUser
            ? "rounded-br-sm bg-neuro-500/15 border border-neuro-500/20 text-[var(--text-primary)]"
            : "rounded-bl-sm bg-[var(--bg-secondary)] border border-[var(--border-color)] text-[var(--text-secondary)]"
          }`}
      >
        <div className="prose-chat">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>

        {/* Footer */}
        <div className={`flex items-center gap-2 mt-2 text-[10px] ${isUser ? "justify-end" : "justify-start"}`}>
          {message.duration_sec !== undefined && (
            <span className="text-[var(--text-muted)]">{message.duration_sec.toFixed(1)}s</span>
          )}
          {message.kb_used && (
            <span className="flex items-center gap-1 text-neuro-500 dark:text-neuro-400">
              <BookOpen size={10} />
              KB
            </span>
          )}
          <span className="text-[var(--text-muted)]">
            {new Date(message.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </span>
        </div>
      </div>
    </div>
  );
}