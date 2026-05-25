"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  MessageSquare, BarChart3, FlaskConical, Info,
  Brain, Code2, Microscope,
} from "lucide-react";
import clsx from "clsx";
import ThemeToggle from "./ThemeToggle";

interface Props { userId: string; sessionId: string; }

export default function Sidebar({ userId, sessionId }: Props) {
  const pathname = usePathname();

  const navItems = [
    { href: "/",                 icon: MessageSquare, label: "Chat",             badge: null },
    { href: "/dashboard",        icon: BarChart3,     label: "Research",         badge: null },
    { href: "/neuro-framework",  icon: Brain,         label: "Neuro Framework",  badge: "C1–C8" },
    { href: "/code-tasks",       icon: Code2,         label: "Code Tasks",       badge: "3" },
    { href: "/research-metrics", icon: Microscope,    label: "Research Metrics", badge: "4" },
  ];

  return (
    <aside className="w-64 flex-shrink-0 flex flex-col border-r border-[var(--border-color)] transition-theme"
      style={{ background: "var(--sidebar-bg)" }}>
      {/* Logo */}
      <div className="px-5 py-5 border-b border-white/8">
        <div className="flex items-center gap-2.5">
          <div className="relative">
            <FlaskConical size={22} className="text-neuro-400" />
            {/* Neural pulse ring */}
            <span className="absolute -inset-1 rounded-full border border-neuro-500/30 animate-neural-pulse" />
          </div>
          <div>
            <p className="font-bold text-sm text-white leading-none font-display">PISA Cognitive</p>
            <p className="text-[10px] text-neuro-400 mt-0.5 font-mono tracking-wider">C1–C8 NEURO FRAMEWORK</p>
          </div>
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
        {navItems.map(({ href, icon: Icon, label, badge }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 group",
                active
                  ? "bg-neuro-500/15 text-neuro-300 shadow-[inset_0_0_20px_rgba(0,180,204,0.08)]"
                  : "text-white/50 hover:bg-white/5 hover:text-white/90"
              )}
            >
              <Icon
                size={16}
                className={clsx(
                  "flex-shrink-0 transition-colors",
                  active ? "text-neuro-400" : "text-white/40 group-hover:text-white/70"
                )}
              />
              <span className="flex-1">{label}</span>
              {badge && (
                <span
                  className={clsx(
                    "text-[10px] font-mono font-bold px-1.5 py-0.5 rounded",
                    active
                      ? "bg-neuro-500/20 text-neuro-400"
                      : "bg-white/8 text-white/30 group-hover:bg-white/12 group-hover:text-white/50"
                  )}
                >
                  {badge}
                </span>
              )}
              {active && (
                <span className="w-1 h-5 rounded-full bg-neuro-400 flex-shrink-0" />
              )}
            </Link>
          );
        })}
      </nav>

      {/* Session info */}
      <div className="px-4 py-4 border-t border-white/8">
        <div className="flex items-start gap-2 text-xs text-white/30">
          <Info size={13} className="mt-0.5 flex-shrink-0 text-white/20" />
          <div className="min-w-0">
            <p className="font-mono text-white/50 truncate">{userId}</p>
            <p className="text-white/25 truncate mt-0.5">
              Session: {sessionId.slice(0, 8)}…
            </p>
            <p className="mt-2 leading-relaxed text-white/20">
              Responses recorded for C1–C8 cognitive science research.
            </p>
          </div>
        </div>
      </div>
    </aside>
  );
}