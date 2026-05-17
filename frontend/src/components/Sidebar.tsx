"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { MessageSquare, BarChart3, FlaskConical, Info } from "lucide-react";
import clsx from "clsx";

interface Props { userId: string; sessionId: string; }

export default function Sidebar({ userId, sessionId }: Props) {
  const pathname = usePathname();

  const navItems = [
    { href: "/",           icon: MessageSquare, label: "Chat" },
    { href: "/dashboard",  icon: BarChart3,     label: "Research" },
  ];

  return (
    <aside className="w-64 flex-shrink-0 flex flex-col bg-gray-900 text-gray-100">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-gray-700">
        <div className="flex items-center gap-2.5">
          <FlaskConical size={22} className="text-brand-400" />
          <div>
            <p className="font-bold text-sm text-white leading-none">PISA Cognitive</p>
            <p className="text-xs text-gray-400 mt-0.5">Science Tutor</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 px-3 space-y-1">
        {navItems.map(({ href, icon: Icon, label }) => (
          <Link
            key={href}
            href={href}
            className={clsx(
              "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors",
              pathname === href
                ? "bg-brand-600 text-white"
                : "text-gray-400 hover:bg-gray-800 hover:text-white"
            )}
          >
            <Icon size={16} />
            {label}
          </Link>
        ))}
      </nav>

      {/* Session info */}
      <div className="px-4 py-4 border-t border-gray-700">
        <div className="flex items-start gap-2 text-xs text-gray-500">
          <Info size={13} className="mt-0.5 flex-shrink-0" />
          <div>
            <p className="font-mono text-gray-400 truncate">{userId}</p>
            <p className="text-gray-600 truncate mt-0.5">
              Session: {sessionId.slice(0, 8)}…
            </p>
            <p className="mt-2 leading-relaxed">
              Responses are recorded for cognitive science research. No personal data is collected.
            </p>
          </div>
        </div>
      </div>
    </aside>
  );
}