"use client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { BookOpen, User } from "lucide-react";
import type { ChatMessage } from "@/lib/types";
import clsx from "clsx";

interface Props { message: ChatMessage; }

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={clsx("flex gap-3", isUser ? "flex-row-reverse" : "flex-row")}>
      {/* Avatar */}
      <div className={clsx(
        "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-white text-sm",
        isUser ? "bg-brand-600" : "bg-gray-700"
      )}>
        {isUser ? <User size={16} /> : <BookOpen size={16} />}
      </div>

      {/* Bubble */}
      <div className={clsx(
        "max-w-[78%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
        isUser
          ? "bg-brand-600 text-white rounded-tr-none"
          : "bg-white border border-gray-200 text-gray-800 rounded-tl-none shadow-sm"
      )}>
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="prose-chat prose prose-sm max-w-none prose-gray">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {/* Metadata row */}
        <div className={clsx(
          "flex items-center gap-2 mt-1.5 text-xs",
          isUser ? "text-blue-200 justify-end" : "text-gray-400"
        )}>
          <span>
            {new Date(message.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </span>
          {isUser && message.duration_sec !== undefined && (
            <span>· {message.duration_sec.toFixed(1)}s</span>
          )}
          {!isUser && message.kb_used && (
            <span className="flex items-center gap-1 text-brand-500">
              <BookOpen size={10} /> KB
            </span>
          )}
        </div>
      </div>
    </div>
  );
}