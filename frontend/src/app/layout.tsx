import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PISA Cognitive Tutor · C1–C8 Neuro-Software Framework",
  description:
    "AI-powered Socratic science tutor with Computational Neuro-Software Engineering C1–C8 cognitive analysis, KEDE metrics, and physiological telemetry for cognitive research",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var theme = localStorage.getItem('theme');
                  if (theme === 'dark' || (!theme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                    document.documentElement.classList.add('dark');
                  }
                } catch(e) {}
              })();
            `,
          }}
        />
      </head>
      <body className="h-full bg-[var(--bg-primary)] text-[var(--text-primary)] antialiased font-sans transition-theme">
        {children}
      </body>
    </html>
  );
}