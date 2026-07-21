"use client";

import type { KeyboardEvent } from "react";
import type { MergeValidation, Script, Scores } from "@/lib/types";
import { SCORE_META, SCORE_MAX } from "../shared";

const MERGED_TAB_ID = "__merged__";

// Mirrors meta_critic.py's _AXIS_WEIGHTS exactly — single source of truth is the backend.
const AXIS_WEIGHTS: Record<string, number> = {
  hook: 0.25,
  pacing: 0.20,
  completion: 0.20,
  cta: 0.20,
  tone: 0.15,
};

interface ScriptsPanelProps {
  scripts: Script[];
  activeScriptId: string | null;
  winnerId: string | null;
  merge: MergeValidation | null;
  onSelectScript: (id: string) => void;
  onTabKeyDown: (e: KeyboardEvent<HTMLButtonElement>, index: number) => void;
}

export default function ScriptsPanel({
  scripts,
  activeScriptId,
  winnerId,
  merge,
  onSelectScript,
}: ScriptsPanelProps) {
  const winner = scripts.find((x) => x.id === winnerId);

  // Winning variant's actual scores (the meta-critic picks the single best composite variant).
  const mergedScores: Scores | null = winner?.scores ?? null;

  const mergedTotal = mergedScores
    ? Math.round(
        Object.entries(AXIS_WEIGHTS).reduce(
          (sum, [key, w]) => sum + w * ((mergedScores as Record<string, number>)[key] ?? 0),
          0
        )
      )
    : 0;

  const activeId = activeScriptId ?? MERGED_TAB_ID;
  const isMergedActive = activeId === MERGED_TAB_ID;
  const activeScript = isMergedActive ? null : scripts.find((x) => x.id === activeId) ?? null;

  // All tabs: merged first, then individual variants.
  const allTabIds = [MERGED_TAB_ID, ...scripts.map((s) => s.id)];

  const handleTabKey = (e: KeyboardEvent<HTMLButtonElement>, tabIndex: number) => {
    if (!allTabIds.length) return;
    let next: number | null = null;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (tabIndex + 1) % allTabIds.length;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = (tabIndex - 1 + allTabIds.length) % allTabIds.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = allTabIds.length - 1;
    if (next === null) return;
    e.preventDefault();
    const nextId = allTabIds[next];
    onSelectScript(nextId);
    requestAnimationFrame(() => {
      document.getElementById(`script-tab-${nextId}`)?.focus();
    });
  };

  return (
    <section data-rid="section-pad" style={{ background: "var(--inverse-bg)", color: "var(--inverse-fg)", padding: "80px 48px", animation: "pc-section-in 0.6s var(--ease) both" }}>
      <div style={{ maxWidth: 1180, margin: "0 auto" }}>
        <span style={{ fontFamily: "var(--font-sans)", fontSize: "11.5px", fontWeight: 700, letterSpacing: "1.5px", textTransform: "uppercase", color: "var(--accent)", display: "block", marginBottom: 28 }}>
          Scriptwriter + Critic — winning cut
        </span>

        {/* Winning script score label at top */}
        <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, letterSpacing: "0.5px", color: "rgba(249,244,234,0.75)", marginBottom: 14 }}>
          Winning script · score {mergedTotal}
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 34, maxWidth: 900 }}>
          {winner?.lines.map((line, i) => (
            <div key={i} style={{ fontFamily: "var(--font-serif)", fontStyle: "italic", fontSize: "clamp(24px, 3vw, 34px)", lineHeight: 1.35, textAlign: "left" }}>
              {line}
            </div>
          ))}
        </div>
        <div style={{ width: 48, height: 2, background: "var(--accent)", marginBottom: 34 }} />

        {/* Merge validation */}
        {merge && (
          <div style={{ borderTop: "1px solid rgba(249,244,234,0.16)", paddingTop: 22, marginBottom: 34 }}>
            <div style={{ fontFamily: "var(--font-sans)", fontSize: 11, fontWeight: 700, letterSpacing: "1px", textTransform: "uppercase", color: "rgba(249,244,234,0.72)", marginBottom: 12 }}>
              Merge validation · {merge.status.toUpperCase()} · {merge.repairPath}
            </div>
            <p style={{ margin: "0 0 16px", fontSize: 13, lineHeight: 1.6, color: "rgba(249,244,234,0.72)", maxWidth: "70ch", textAlign: "left" }}>{merge.note}</p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, maxWidth: 900 }}>
              <div>
                <div style={{ fontFamily: "var(--font-sans)", fontSize: 11, fontWeight: 700, letterSpacing: "0.8px", textTransform: "uppercase", color: "rgba(249,244,234,0.7)", marginBottom: 6 }}>
                  Before
                </div>
                <p style={{ margin: 0, fontSize: "12.5px", lineHeight: 1.6, color: "rgba(249,244,234,0.55)", textDecorationLine: "line-through", textDecorationColor: "rgba(249,244,234,0.25)", textAlign: "left" }}>
                  {merge.seam.before}
                </p>
              </div>
              <div>
                <div style={{ fontFamily: "var(--font-sans)", fontSize: 11, fontWeight: 700, letterSpacing: "0.8px", textTransform: "uppercase", color: "var(--accent)", marginBottom: 6 }}>
                  After
                </div>
                <p style={{ margin: 0, fontSize: "12.5px", lineHeight: 1.6, color: "rgba(249,244,234,0.92)", textAlign: "left" }}>{merge.seam.after}</p>
              </div>
            </div>
            <div style={{ marginTop: 14, fontFamily: "var(--font-mono)", fontSize: 11, color: "rgba(249,244,234,0.45)" }}>
              Meta-Critic swap: {merge.metaCriticSwapFired ? "fired" : "not required"}
            </div>
          </div>
        )}

        {/* Tab strip: Winner tab + individual variants */}
        <div style={{ fontFamily: "var(--font-sans)", fontSize: 11, fontWeight: 700, letterSpacing: "1px", textTransform: "uppercase", color: "rgba(249,244,234,0.72)", marginBottom: 14, borderTop: "1px solid rgba(249,244,234,0.16)", paddingTop: 30 }}>
          All variants
        </div>
        <div role="tablist" aria-label="Script variants" style={{ display: "flex", flexWrap: "wrap", gap: 10, marginBottom: 28 }}>
          {/* Winner tab */}
          {mergedScores && (
            <button
              id={`script-tab-${MERGED_TAB_ID}`}
              tabIndex={isMergedActive ? 0 : -1}
              role="tab"
              aria-selected={isMergedActive}
              aria-controls={`script-panel-${MERGED_TAB_ID}`}
              onClick={() => onSelectScript(MERGED_TAB_ID)}
              onKeyDown={(e) => handleTabKey(e, 0)}
              className="pcs-tab"
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                fontFamily: "var(--font-sans)",
                fontSize: 14,
                fontWeight: isMergedActive ? 700 : 500,
                padding: "10px 16px",
                border: isMergedActive ? "1px solid var(--accent)" : "1px solid rgba(249,244,234,0.2)",
                borderBottom: isMergedActive ? "3px solid var(--accent)" : "1px solid rgba(249,244,234,0.2)",
                background: isMergedActive ? "rgba(249,244,234,0.08)" : "transparent",
                cursor: "pointer",
                color: isMergedActive ? "var(--paper)" : "rgba(249,244,234,0.72)",
                minHeight: 44,
              }}
            >
              <span>Winner ✓</span>
              <span style={{ fontFamily: "var(--font-mono)", opacity: 0.55 }}>{mergedTotal}</span>
            </button>
          )}

          {/* Individual variant tabs */}
          {scripts.map((sc, i) => {
            const active = sc.id === activeId;
            return (
              <button
                key={sc.id}
                id={`script-tab-${sc.id}`}
                tabIndex={active ? 0 : -1}
                role="tab"
                aria-selected={active}
                aria-controls={`script-panel-${sc.id}`}
                onClick={() => onSelectScript(sc.id)}
                onKeyDown={(e) => handleTabKey(e, i + 1)}
                className="pcs-tab"
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  fontFamily: "var(--font-sans)",
                  fontSize: 14,
                  fontWeight: active ? 700 : 500,
                  padding: "10px 16px",
                  border: active ? "1px solid var(--accent)" : "1px solid rgba(249,244,234,0.2)",
                  borderBottom: active ? "3px solid var(--accent)" : "1px solid rgba(249,244,234,0.2)",
                  background: active ? "rgba(249,244,234,0.08)" : "transparent",
                  cursor: "pointer",
                  color: active ? "var(--paper)" : "rgba(249,244,234,0.72)",
                  minHeight: 44,
                }}
              >
                <span>{sc.title}</span>
                <span style={{ fontFamily: "var(--font-mono)", opacity: 0.55 }}>{sc.total}</span>
              </button>
            );
          })}
        </div>

        {/* Tabpanel */}
        <div
          id={`script-panel-${activeId}`}
          role="tabpanel"
          aria-labelledby={`script-tab-${activeId}`}
          style={{ borderTop: "1px solid rgba(249,244,234,0.16)", paddingTop: 28 }}
        >
          <div style={{ fontFamily: "var(--font-sans)", fontSize: 11, fontWeight: 700, letterSpacing: "0.8px", textTransform: "uppercase", color: "rgba(249,244,234,0.72)", marginBottom: 20 }}>
            {isMergedActive ? "Winning script · breakdown" : `${activeScript?.title ?? ""} · breakdown`}
          </div>
          <div data-rid="scripts-breakdown-grid" style={{ display: "grid", gridTemplateColumns: "1.1fr 1.4fr", gap: 56, alignItems: "start" }}>
            <div>
              {(isMergedActive ? mergedScores : activeScript) &&
                SCORE_META.map((m) => {
                  const score = isMergedActive
                    ? (mergedScores as Scores)[m.key]
                    : (activeScript as Script).scores[m.key];
                  return (
                    <div key={m.key} style={{ marginBottom: 14 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4, color: "rgba(249,244,234,0.75)" }}>
                        <span>{m.label}</span>
                        <span style={{ fontFamily: "var(--font-mono)" }}>{score}</span>
                      </div>
                      <div style={{ height: 2, background: "rgba(249,244,234,0.14)", overflow: "hidden" }}>
                        <div
                          style={{
                            height: "100%",
                            width: "100%",
                            background: "var(--accent)",
                            transform: `scaleX(${Math.min(score / SCORE_MAX, 1)})`,
                            transformOrigin: "left",
                            transition: "transform .6s var(--ease)",
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
            </div>
            <p style={{ margin: 0, fontSize: "14.5px", lineHeight: 1.7, color: "rgba(249,244,234,0.85)", textAlign: "left" }}>
              {isMergedActive
                ? "The highest-scoring script variant, picked by the Meta-Critic's composite score (hook 25%, pacing 20%, completion 20%, CTA 20%, tone 15%). All four variants were scored independently; the best overall wins."
                : (activeScript?.reasoning ?? "")}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
