"use client";
import { useEffect, useState } from "react";
import { Sun, Moon } from "lucide-react";

export default function ThemeToggle() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    // Read stored preference on mount
    const stored = localStorage.getItem("theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const isDark = stored === "dark" || (!stored && prefersDark);
    setDark(isDark);
    document.documentElement.classList.toggle("dark", isDark);
  }, []);

  const toggle = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("theme", next ? "dark" : "light");
  };

  return (
    <button
      onClick={toggle}
      title={dark ? "Switch to light mode" : "Switch to dark mode"}
      className="relative flex items-center justify-center w-9 h-9 rounded-xl
                 border border-[var(--border-color)] bg-[var(--bg-tertiary)]
                 hover:border-[var(--neuro-cyan)] hover:shadow-[0_0_12px_rgba(0,180,204,0.2)]
                 transition-all duration-200 group"
    >
      {dark ? (
        <Sun
          size={15}
          className="text-amber-400 group-hover:text-amber-300 transition-colors"
        />
      ) : (
        <Moon
          size={15}
          className="text-slate-500 group-hover:text-neuro-500 transition-colors"
        />
      )}
    </button>
  );
}
