"use client";

import Link from "next/link";

interface HeaderProps {
  theme: "light" | "dark";
  onToggleTheme: () => void;
}

export default function Header({ theme, onToggleTheme }: HeaderProps) {
  const isDark = theme === "dark";
  return (
    <header
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 20,
        padding: "18px 40px",
        borderBottom: "1px solid var(--line-strong)",
        position: "sticky",
        top: 0,
        zIndex: 30,
        background: "var(--bg)",
      }}
    >
      <Link href="/" style={{ display: "flex", alignItems: "center", gap: 13, textDecoration: "none", color: "inherit" }}>
        <div
          style={{
            width: 30,
            height: 30,
            borderRadius: 8,
            border: "1.5px solid var(--tan)",
            background: "linear-gradient(135deg, var(--tan) 0 46%, transparent 46% 100%)",
          }}
        />
        <div style={{ display: "flex", alignItems: "baseline", gap: 11 }}>
          <span style={{ fontFamily: "var(--font-serif)", fontWeight: 500, fontSize: 24, letterSpacing: "-0.3px" }}>
            ProductCut
          </span>
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              letterSpacing: "1.5px",
              textTransform: "uppercase",
              color: "var(--muted)",
            }}
          >
            ad-film studio
          </span>
        </div>
      </Link>
      <button
        onClick={onToggleTheme}
        role="switch"
        aria-checked={isDark}
        aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
        style={{
          position: "relative",
          width: 62,
          height: 30,
          borderRadius: 999,
          border: "1px solid var(--line-strong)",
          background: "var(--surface2)",
          cursor: "pointer",
          padding: 0,
          flex: "0 0 auto",
        }}
      >
        <span
          style={{
            position: "absolute",
            left: 9,
            top: "50%",
            transform: "translateY(-50%)",
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: "var(--ink-soft)",
            opacity: 0.55,
          }}
        />
        <span
          style={{
            position: "absolute",
            right: 9,
            top: "50%",
            transform: "translateY(-50%)",
            width: 8,
            height: 8,
            borderRadius: "50%",
            border: "1.5px solid var(--ink-soft)",
            opacity: 0.55,
          }}
        />
        <span
          style={{
            position: "absolute",
            top: 2,
            left: 2,
            width: 24,
            height: 24,
            borderRadius: "50%",
            background: "var(--accent)",
            boxShadow: "0 2px 5px var(--shadow)",
            transition: "transform .38s cubic-bezier(.4,1.3,.5,1), background-color .4s ease",
            transform: `translateX(${isDark ? "32px" : "0"})`,
          }}
        />
      </button>
    </header>
  );
}
