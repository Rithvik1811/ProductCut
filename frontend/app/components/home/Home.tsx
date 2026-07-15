"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef } from "react";
import type { CSSProperties } from "react";
import { useMergeState } from "@/lib/useMergeState";
import "./home.css";

type Theme = "light" | "dark";
type StepKind = "upload" | "truths" | "agents" | "dashboard";

interface HomeState {
  theme: Theme;
  navShrunk: boolean;
  scrollY: number;
  visible: Record<string, boolean>;
}

function initialHomeState(): HomeState {
  return { theme: "light", navShrunk: false, scrollY: 0, visible: {} };
}

interface Step {
  num: string;
  title: string;
  body: string;
  kind: StepKind;
}

const STEPS: Step[] = [
  {
    num: "01",
    title: "Upload two or three photos",
    body: "Clear, well-lit shots of your actual product. No stock, no stand-ins — this is what the whole ad gets built from.",
    kind: "upload",
  },
  {
    num: "02",
    title: "AI extracts what's really true",
    body: "The Truth Agent reads material, color, texture, size, and any distinguishing marks straight out of your photos.",
    kind: "truths",
  },
  {
    num: "03",
    title: "Agents write, critique, and shoot",
    body: "A scriptwriter drafts variants, a critic scores and selects the best one, a director builds the treatment, shots get generated and budgeted.",
    kind: "agents",
  },
  {
    num: "04",
    title: "Watch your dashboard build live",
    body: "Every decision streams in as it happens — scores, budget, continuity checks — right up to your finished ad.",
    kind: "dashboard",
  },
];

const TRUTH_CATS = [
  { label: "Material", fact: "Wheel-thrown stoneware, matte glaze." },
  { label: "Color", fact: "Terracotta body, speckled cream interior." },
  { label: "Mark", fact: "Maker's stamp on the unglazed base." },
];

const AGENT_ROLES = ["Scriptwriter", "Critic", "Director", "Producer"];

const POSTER_TRUTHS = [
  { cat: "Material", fact: "Wheel-thrown stoneware, matte glaze." },
  { cat: "Mark", fact: "Maker's stamp on the base." },
  { cat: "Size", fact: "Holds ~12oz, two-finger handle." },
];

const POSTER_SCORES = [
  { label: "The Last Cup", value: "89", pct: 89 },
  { label: "The Maker's Hands", value: "82", pct: 82 },
  { label: "Morning Ritual", value: "81", pct: 81 },
];

const TRUTH_SAMPLES = [
  { cat: "Material", fact: "Wheel-thrown stoneware, matte food-safe glaze." },
  { cat: "Mark", fact: "Maker's stamp pressed into the unglazed base." },
  { cat: "Size", fact: "Holds ~12oz, two-finger handle." },
];

const TRANSPARENCY_ROWS = [
  { label: "Winning script score", value: "89 / 100", pct: 89 },
  { label: "Budget used", value: "108 / 120 credits", pct: 90 },
  { label: "Continuity drift (avg)", value: "0.14", pct: 23 },
];

const RATIO_TILES = [
  { id: "9:16", width: 44, height: 78 },
  { id: "1:1", width: 62, height: 62 },
  { id: "16:9", width: 96, height: 54 },
];

function uploadTiles(alt: boolean) {
  return [0, 1, 2].map((i) => ({
    key: i,
    style: {
      width: "30%",
      aspectRatio: "1 / 1",
      border: `1.5px solid ${alt ? "rgba(249,244,234,0.25)" : "var(--hair-strong)"}`,
      background: i === 1 ? "var(--accent)" : alt ? "rgba(249,244,234,0.06)" : "var(--paper-deep)",
      transform: `rotate(${(i - 1) * 4}deg)`,
      transition: "transform .6s var(--ease)",
    } as CSSProperties,
  }));
}

function agentRoles(alt: boolean) {
  return AGENT_ROLES.map((label, i) => ({
    key: label,
    label,
    style: {
      border: `1px solid ${alt ? "rgba(249,244,234,0.2)" : "var(--hair-strong)"}`,
      padding: 14,
      fontFamily: "var(--font-mono)",
      fontSize: 11,
      color: alt ? "rgba(249,244,234,0.7)" : "var(--ink-soft)",
      animation: `pc-home-drift-${i % 2 === 0 ? "a" : "b"} ${5 + i}s ease-in-out infinite`,
    } as CSSProperties,
  }));
}

function miniShots() {
  return [0, 1, 2, 3, 4, 5].map((i) => ({
    key: i,
    style: {
      aspectRatio: "4 / 3",
      background:
        i === 4
          ? "repeating-linear-gradient(45deg, var(--accent) 0 5px, rgba(0,0,0,0) 5px 10px)"
          : i < 4
            ? "var(--ink)"
            : "var(--hair)",
      opacity: i < 4 ? 0.85 : 0.5,
    } as CSSProperties,
  }));
}

export default function Home() {
  const [state, setState] = useMergeState<HomeState>(initialHomeState);
  const { theme, navShrunk, scrollY, visible } = state;
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let saved: string | null = null;
    try {
      saved = localStorage.getItem("pc-theme");
    } catch {
      // ignore
    }
    if (saved === "light" || saved === "dark") setState({ theme: saved });

    const onScroll = () => setState({ scrollY: window.scrollY, navShrunk: window.scrollY > 40 });
    window.addEventListener("scroll", onScroll, { passive: true });

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((en) => {
          if (en.isIntersecting) {
            const id = en.target.getAttribute("data-reveal-id");
            if (id) setState((s) => (s.visible[id] ? {} : { visible: { ...s.visible, [id]: true } }));
          }
        });
      },
      { threshold: 0.15 },
    );
    const frame = requestAnimationFrame(() => {
      rootRef.current?.querySelectorAll("[data-reveal-id]").forEach((el) => observer.observe(el));
    });

    return () => {
      window.removeEventListener("scroll", onScroll);
      observer.disconnect();
      cancelAnimationFrame(frame);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggleTheme = useCallback(() => {
    setState((s) => {
      const theme: Theme = s.theme === "dark" ? "light" : "dark";
      try {
        localStorage.setItem("pc-theme", theme);
      } catch {
        // ignore
      }
      return { theme };
    });
  }, [setState]);

  const onPlayDemo = useCallback(() => {
    console.log("[ProductCut] play demo — wire real video src here");
  }, []);

  const reveal = useCallback(
    (id: string): CSSProperties => ({
      opacity: visible[id] ? 1 : 0,
      transform: visible[id] ? "translateY(0)" : "translateY(30px)",
      transition: "opacity .85s var(--ease), transform .85s var(--ease)",
    }),
    [visible],
  );

  const isDark = theme === "dark";
  const py = scrollY * 0.12;

  return (
    <div ref={rootRef} className="pc-home" data-theme={theme}>
      {/* NAV */}
      <nav
        style={{
          position: "sticky",
          top: 0,
          zIndex: 40,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: navShrunk ? "14px 48px" : "22px 48px",
          background: navShrunk ? "rgba(249,244,234,0.86)" : "transparent",
          backdropFilter: navShrunk ? "blur(10px)" : "none",
          borderBottom: `1px solid ${navShrunk ? "var(--hair)" : "transparent"}`,
          transition:
            "padding .4s var(--ease), background-color .4s var(--ease), border-color .4s var(--ease)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ position: "relative", width: 18, height: 18, border: "1.3px solid var(--ink)", flex: "0 0 auto" }}>
            <span
              style={{
                position: "absolute",
                width: 4,
                height: 4,
                borderRadius: "50%",
                background: "var(--accent)",
                top: 3,
                left: 3,
              }}
            />
          </div>
          <span style={{ fontFamily: "var(--font-serif)", fontStyle: "italic", fontSize: 19 }}>ProductCut</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
          <div data-rid="nav-links" style={{ display: "flex", alignItems: "center", gap: 24 }}>
            <a href="#how" className="pch-navlink" style={{ fontFamily: "var(--font-sans)", fontSize: "13.5px", fontWeight: 600, textDecoration: "none" }}>
              How it works
            </a>
            <a href="#different" className="pch-navlink" style={{ fontFamily: "var(--font-sans)", fontSize: "13.5px", fontWeight: 600, textDecoration: "none" }}>
              Why it&rsquo;s different
            </a>
            <Link href="/studio?view=library" className="pch-navlink" style={{ fontFamily: "var(--font-sans)", fontSize: "13.5px", fontWeight: 600, textDecoration: "none" }}>
              My Ads
            </Link>
          </div>
          <button
            onClick={toggleTheme}
            role="switch"
            aria-checked={isDark}
            aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
            style={{
              position: "relative",
              width: 52,
              height: 28,
              borderRadius: 999,
              border: "1px solid var(--hair-strong)",
              background: "var(--paper-deep)",
              cursor: "pointer",
              padding: 0,
              flex: "0 0 auto",
              transition: "background-color .3s var(--ease)",
            }}
          >
            <span
              style={{
                position: "absolute",
                top: 2,
                left: 2,
                width: 24,
                height: 24,
                borderRadius: "50%",
                background: "var(--paper)",
                boxShadow: "0 1px 3px var(--shadow)",
                transition: "transform .35s var(--ease)",
                transform: `translateX(${isDark ? "24px" : "0"})`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {isDark ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="4" />
                  <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
                </svg>
              )}
            </span>
          </button>
          <Link
            href="/studio"
            className="pch-trycta"
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: 13,
              fontWeight: 700,
              padding: "9px 18px",
              border: "1px solid var(--ink)",
              background: "var(--ink)",
              color: "var(--paper)",
              cursor: "pointer",
              whiteSpace: "nowrap",
              textDecoration: "none",
              display: "inline-block",
              transition: "transform .35s var(--ease), background-color .3s var(--ease)",
            }}
          >
            Try it
          </Link>
        </div>
      </nav>

      {/* HERO */}
      <section data-rid="hero" style={{ position: "relative", padding: "150px 48px 90px", maxWidth: 1180, margin: "0 auto", overflow: "hidden" }}>
        <div
          style={{
            position: "absolute",
            top: -60,
            right: -40,
            width: 260,
            height: 260,
            borderRadius: "50%",
            background: "radial-gradient(circle at 30% 30%, var(--paper-deep), transparent 70%)",
            transform: `translateY(${py}px)`,
            pointerEvents: "none",
          }}
        />
        <div
          style={{
            position: "absolute",
            bottom: -100,
            left: -60,
            width: 200,
            height: 200,
            border: "1px solid var(--hair-strong)",
            transform: `translateY(${-py * 0.6}px) rotate(12deg)`,
            pointerEvents: "none",
          }}
        />
        <div style={{ position: "relative", zIndex: 1 }}>
          <div style={{ fontFamily: "var(--font-sans)", fontSize: 12, fontWeight: 700, letterSpacing: "2px", textTransform: "uppercase", color: "var(--accent)", marginBottom: 26 }}>
            AI product ad videos · built for small makers
          </div>
          <h1
            style={{
              fontFamily: "var(--font-serif)",
              fontWeight: 400,
              fontSize: "clamp(48px, 7vw, 92px)",
              lineHeight: 0.98,
              letterSpacing: "-1.6px",
              margin: "0 0 28px",
              maxWidth: "16ch",
            }}
          >
            Three photos in. One <em style={{ fontStyle: "italic", color: "var(--accent)" }}>honest</em> ad out.
          </h1>
          <p style={{ margin: "0 0 42px", fontSize: 18, lineHeight: 1.6, color: "var(--ink-soft)", maxWidth: "46ch" }}>
            ProductCut reads your real product photos, pins down what&rsquo;s actually true about it, and has a team of
            writing &amp; directing agents script, shoot, and cut a short ad — grounded in facts, not filler.
          </p>
          <Link
            href="/studio"
            className="pch-hero-cta"
            style={{
              display: "inline-block",
              textDecoration: "none",
              fontFamily: "var(--font-sans)",
              fontSize: 16,
              fontWeight: 700,
              padding: "18px 34px",
              border: "none",
              cursor: "pointer",
              background: "var(--ink)",
              color: "var(--paper)",
              transition: "transform .45s var(--ease), box-shadow .45s var(--ease)",
            }}
          >
            Try it — make your own ad →
          </Link>
        </div>
      </section>

      {/* DEMO */}
      <section data-reveal-id="demo" style={{ ...reveal("demo"), maxWidth: 1180, margin: "60px auto 0", padding: "0 48px 120px" }}>
        <div style={{ fontFamily: "var(--font-sans)", fontSize: 12, fontWeight: 700, letterSpacing: "2px", textTransform: "uppercase", color: "var(--faint)", marginBottom: 22 }}>
          See it run
        </div>
        <div style={{ position: "relative", width: "100%", aspectRatio: "16 / 9", background: "var(--inverse-bg)", overflow: "hidden", boxShadow: "0 30px 70px var(--shadow)" }}>
          <div style={{ position: "absolute", inset: 0, padding: "44px 56px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 40, opacity: 0.5 }}>
            <div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "1.5px", textTransform: "uppercase", color: "var(--accent)", marginBottom: 18 }}>
                Truth Agent
              </div>
              {POSTER_TRUTHS.map((pt) => (
                <div key={pt.cat} style={{ display: "flex", gap: 12, padding: "10px 0", borderTop: "1px solid rgba(249,244,234,0.14)" }}>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.6px", textTransform: "uppercase", color: "rgba(249,244,234,0.5)", flex: "0 0 64px" }}>
                    {pt.cat}
                  </span>
                  <span style={{ fontFamily: "var(--font-serif)", fontSize: 14, color: "rgba(249,244,234,0.8)" }}>{pt.fact}</span>
                </div>
              ))}
            </div>
            <div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "1.5px", textTransform: "uppercase", color: "var(--accent)", marginBottom: 18 }}>
                Critic
              </div>
              {POSTER_SCORES.map((ps) => (
                <div key={ps.label} style={{ marginBottom: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "rgba(249,244,234,0.55)", marginBottom: 5 }}>
                    <span>{ps.label}</span>
                    <span>{ps.value}</span>
                  </div>
                  <div style={{ height: 3, background: "rgba(249,244,234,0.14)" }}>
                    <div style={{ height: "100%", width: `${ps.pct}%`, background: "var(--accent)" }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div style={{ position: "absolute", inset: 0, background: "linear-gradient(180deg, rgba(18,52,59,0.55), rgba(18,52,59,0.78))" }} />
          <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 18 }}>
            <button
              onClick={onPlayDemo}
              className="pch-play"
              style={{
                width: 76,
                height: 76,
                borderRadius: "50%",
                background: "var(--accent)",
                border: "none",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "transform .4s var(--ease)",
                boxShadow: "0 12px 30px rgba(0,0,0,0.35)",
              }}
            >
              <span
                style={{
                  width: 0,
                  height: 0,
                  borderTop: "15px solid transparent",
                  borderBottom: "15px solid transparent",
                  borderLeft: "24px solid var(--accent-ink)",
                  marginLeft: 6,
                }}
              />
            </button>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 12, letterSpacing: "0.5px", color: "rgba(249,244,234,0.75)" }}>
              Full pipeline demo — intake → agents thinking → live dashboard
            </span>
          </div>
          <span style={{ position: "absolute", bottom: 16, left: 20, fontFamily: "var(--font-mono)", fontSize: "11.5px", color: "rgba(249,244,234,0.45)" }}>
            demo.mp4 · drop your recording here
          </span>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <div id="how" />
      {STEPS.map((step, i) => {
        const alt = i % 2 === 1;
        const revealId = `step-${i}`;
        const bodyColor = alt ? "rgba(249,244,234,0.65)" : "var(--ink-soft)";
        const hairColor = alt ? "rgba(249,244,234,0.16)" : "var(--hair-strong)";
        const on = !!visible[revealId];
        return (
          <section
            key={revealId}
            data-reveal-id={revealId}
            style={{
              background: alt ? "var(--inverse-bg)" : "var(--paper)",
              color: alt ? "var(--inverse-fg)" : "var(--ink)",
              opacity: on ? 1 : 0,
              transform: on ? "translateY(0)" : "translateY(34px)",
              transition: "opacity .8s var(--ease), transform .8s var(--ease), background-color .5s var(--ease)",
            }}
          >
            <div style={{ maxWidth: 1180, margin: "0 auto", padding: "100px 48px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 64, alignItems: "center" }}>
              <div style={alt ? { order: 2 } : undefined}>
                <div style={{ fontFamily: "var(--font-serif)", fontStyle: "italic", fontSize: 30, color: "var(--accent)", marginBottom: 18 }}>{step.num}</div>
                <h3 style={{ fontFamily: "var(--font-serif)", fontWeight: 400, fontSize: "clamp(28px, 3.4vw, 40px)", lineHeight: 1.15, margin: "0 0 18px", maxWidth: "14ch" }}>
                  {step.title}
                </h3>
                <p style={{ margin: 0, fontSize: "15.5px", lineHeight: 1.65, maxWidth: "40ch", color: bodyColor }}>{step.body}</p>
              </div>
              <div
                style={{
                  order: alt ? 1 : undefined,
                  padding: 36,
                  border: `1px solid ${alt ? "rgba(249,244,234,0.16)" : "var(--hair-strong)"}`,
                  background: alt ? "rgba(249,244,234,0.03)" : "var(--paper)",
                }}
              >
                {step.kind === "upload" && (
                  <div style={{ display: "flex", gap: 14 }}>
                    {uploadTiles(alt).map((tile) => (
                      <div key={tile.key} style={tile.style} />
                    ))}
                  </div>
                )}
                {step.kind === "truths" && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                    {TRUTH_CATS.map((c) => (
                      <div key={c.label} style={{ display: "flex", gap: 12, alignItems: "baseline", paddingBottom: 12, borderBottom: `1px solid ${hairColor}` }}>
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.8px", textTransform: "uppercase", color: "var(--accent)", flex: "0 0 76px" }}>
                          {c.label}
                        </span>
                        <span style={{ fontSize: 13, lineHeight: 1.5, color: bodyColor }}>{c.fact}</span>
                      </div>
                    ))}
                  </div>
                )}
                {step.kind === "agents" && (
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                    {agentRoles(alt).map((r) => (
                      <div key={r.key} style={r.style}>
                        {r.label}
                      </div>
                    ))}
                  </div>
                )}
                {step.kind === "dashboard" && (
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 10 }}>
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.8px", textTransform: "uppercase", color: bodyColor }}>
                        Budget · 108 / 120
                      </span>
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--accent)" }}>within cap</span>
                    </div>
                    <div style={{ height: 4, background: hairColor, marginBottom: 20 }}>
                      <div style={{ height: "100%", width: "90%", background: "var(--accent)" }} />
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
                      {miniShots().map((ms) => (
                        <div key={ms.key} style={ms.style} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </section>
        );
      })}

      {/* WHY DIFFERENT */}
      <div id="different" />
      <section data-reveal-id="diff" style={{ ...reveal("diff"), maxWidth: 1180, margin: "0 auto", padding: "110px 48px 60px" }}>
        <div style={{ fontFamily: "var(--font-sans)", fontSize: 12, fontWeight: 700, letterSpacing: "2px", textTransform: "uppercase", color: "var(--faint)", marginBottom: 60 }}>
          Why it&rsquo;s different
        </div>

        <div data-rid="diff-grid-1" style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr", gap: 70, alignItems: "start", marginBottom: 90 }}>
          <div>
            <h3 style={{ fontFamily: "var(--font-serif)", fontStyle: "italic", fontWeight: 400, fontSize: "clamp(30px, 3.6vw, 46px)", lineHeight: 1.2, margin: "0 0 22px", maxWidth: "15ch" }}>
              Scripts grounded in what&rsquo;s actually true.
            </h3>
            <p style={{ margin: 0, fontSize: "15.5px", lineHeight: 1.65, color: "var(--ink-soft)", maxWidth: "46ch" }}>
              No invented claims, no generic &ldquo;premium quality&rdquo; filler. Every line in the script traces back
              to a fact our Truth Agent verified straight from your photos — material, color, texture, size, the
              maker&rsquo;s mark. If we can&rsquo;t see it, we don&rsquo;t say it.
            </p>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {TRUTH_SAMPLES.map((t) => (
              <div key={t.cat} style={{ display: "flex", gap: 14, alignItems: "baseline", padding: "14px 0", borderTop: "1px solid var(--hair)" }}>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.8px", textTransform: "uppercase", color: "var(--accent)", flex: "0 0 90px" }}>
                  {t.cat}
                </span>
                <span style={{ fontSize: 14, lineHeight: 1.5, color: "var(--ink)" }}>{t.fact}</span>
              </div>
            ))}
          </div>
        </div>

        <div
          data-rid="diff-dark"
          style={{
            background: "var(--inverse-bg)",
            color: "var(--inverse-fg)",
            padding: "70px 56px",
            margin: "0 -8px 90px",
            display: "grid",
            gridTemplateColumns: "1fr 1.3fr",
            gap: 60,
            alignItems: "center",
          }}
        >
          <div>
            <h3 style={{ fontFamily: "var(--font-serif)", fontStyle: "italic", fontWeight: 400, fontSize: "clamp(28px, 3.2vw, 40px)", lineHeight: 1.2, margin: "0 0 20px" }}>
              Nothing happens in a black box.
            </h3>
            <p style={{ margin: 0, fontSize: "14.5px", lineHeight: 1.65, color: "rgba(249,244,234,0.65)", maxWidth: "42ch" }}>
              Watch the critic score every script variant, see the exact budget spent per shot, and review continuity
              drift on every clip — with a real human-review step when something needs your eyes.
            </p>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {TRANSPARENCY_ROWS.map((r) => (
              <div key={r.label}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 6, color: "rgba(249,244,234,0.7)" }}>
                  <span>{r.label}</span>
                  <span style={{ fontFamily: "var(--font-mono)" }}>{r.value}</span>
                </div>
                <div style={{ height: 3, background: "rgba(249,244,234,0.14)" }}>
                  <div style={{ height: "100%", width: `${r.pct}%`, background: "var(--accent)" }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div data-rid="diff-grid-3" style={{ display: "grid", gridTemplateColumns: "1fr 1.3fr", gap: 70, alignItems: "center" }}>
          <div style={{ order: 2 }}>
            <h3 style={{ fontFamily: "var(--font-serif)", fontStyle: "italic", fontWeight: 400, fontSize: "clamp(28px, 3.2vw, 40px)", lineHeight: 1.2, margin: "0 0 20px", maxWidth: "13ch" }}>
              One ad, every format.
            </h3>
            <p style={{ margin: 0, fontSize: "15.5px", lineHeight: 1.65, color: "var(--ink-soft)", maxWidth: "42ch" }}>
              Vertical for Reels and TikTok, square for the feed, landscape for YouTube and your site — exported
              together, framed for how people actually watch.
            </p>
          </div>
          <div data-rid="ratio-tiles" style={{ order: 1, display: "flex", alignItems: "flex-end", gap: 16 }}>
            {RATIO_TILES.map((tile) => (
              <div
                key={tile.id}
                style={{
                  width: tile.width,
                  height: tile.height,
                  border: "1px solid var(--hair-strong)",
                  display: "flex",
                  alignItems: "flex-end",
                  justifyContent: "center",
                  paddingBottom: 6,
                  background: "var(--paper-deep)",
                }}
              >
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "11.5px", color: "var(--faint)" }}>{tile.id}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FINAL CTA */}
      <section data-reveal-id="final" style={{ ...reveal("final"), padding: "140px 48px 150px", textAlign: "center" }}>
        <h2 style={{ fontFamily: "var(--font-serif)", fontWeight: 400, fontStyle: "italic", fontSize: "clamp(40px, 6vw, 72px)", lineHeight: 1.05, margin: "0 0 40px" }}>
          Make your first ad
          <br />
          this afternoon.
        </h2>
        <Link
          href="/studio"
          className="pch-final-cta"
          style={{
            display: "inline-block",
            textDecoration: "none",
            fontFamily: "var(--font-sans)",
            fontSize: 17,
            fontWeight: 700,
            padding: "20px 40px",
            border: "none",
            cursor: "pointer",
            background: "var(--accent)",
            color: "var(--accent-ink)",
            transition: "transform .45s var(--ease), box-shadow .45s var(--ease)",
          }}
        >
          Make your own ad →
        </Link>
      </section>

      {/* FOOTER */}
      <footer data-rid="footer" style={{ borderTop: "1px solid var(--hair)", padding: "30px 48px", display: "flex", alignItems: "center", justifyContent: "space-between", maxWidth: 1180, margin: "0 auto" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <div style={{ position: "relative", width: 15, height: 15, border: "1.2px solid var(--ink)", flex: "0 0 auto" }}>
            <span style={{ position: "absolute", width: "3.5px", height: "3.5px", borderRadius: "50%", background: "var(--accent)", top: "2.5px", left: "2.5px" }} />
          </div>
          <span style={{ fontFamily: "var(--font-serif)", fontStyle: "italic", fontSize: 15, color: "var(--ink-soft)" }}>ProductCut</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 28, fontFamily: "var(--font-sans)", fontSize: "12.5px", color: "var(--faint)" }}>
          <a href="#how" className="pch-footer-link" style={{ textDecoration: "none", color: "var(--faint)" }}>
            How it works
          </a>
          <Link href="/studio" className="pch-footer-link" style={{ fontFamily: "var(--font-sans)", fontSize: "12.5px", color: "var(--faint)", textDecoration: "none" }}>
            Try it
          </Link>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "11.5px" }}>Hackathon build · 2026</span>
        </div>
      </footer>
    </div>
  );
}
