"use client";
import { BookOpen } from "lucide-react";

export default function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center text-white">
        <BookOpen size={16} />
      </div>
      <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-none px-4 py-3 shadow-sm">
        <div className="flex items-center gap-1.5 h-4">
          {[0, 150, 300].map((delay) => (
            <span
              key={delay}
              className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"
              style={{ animationDelay: `${delay}ms` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}