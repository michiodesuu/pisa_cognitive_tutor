import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50:  "#f0f7ff",
          100: "#e0effe",
          200: "#b9dcfd",
          300: "#7cc0fb",
          400: "#36a0f7",
          500: "#0c84e8",
          600: "#0066c6",
          700: "#0152a1",
          800: "#064785",
          900: "#0b3d6e",
        },
        icap: {
          passive:      "#ef4444",
          active:       "#f97316",
          constructive: "#22c55e",
          interactive:  "#3b82f6",
        },
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      animation: {
        "fade-in": "fadeIn 0.3s ease-in-out",
        "slide-up": "slideUp 0.3s ease-out",
        typing: "typing 1s steps(3, end) infinite",
      },
      keyframes: {
        fadeIn: { from: { opacity: "0" }, to: { opacity: "1" } },
        slideUp: { from: { transform: "translateY(16px)", opacity: "0" }, to: { transform: "translateY(0)", opacity: "1" } },
        typing:  { "0%,100%": { content: "''" }, "33%": { content: "'·'" }, "66%": { content: "'··'" } },
      },
    },
  },
  plugins: [],
};
export default config;