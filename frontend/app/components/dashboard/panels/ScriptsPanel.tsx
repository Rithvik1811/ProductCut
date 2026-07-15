"use client";

import type { KeyboardEvent } from "react";
import type { MergeValidation, Script } from "@/lib/types";
import { SCORE_META } from "../shared";

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
  onTabKeyDown,
}: ScriptsPanelProps) {
  const activeId = activeScriptId || scripts[0]?.id;
  const activeScript = scripts.find((x) => x.id === activeId) || scripts[0] || null;
  const winner = scripts.find((x) => x.id === winnerId);

  return (
    <section data-rid="section-pad" style={{ background: "var(--inverse-bg)", color: "var(--inverse-fg)", padding: "80px 48px", animation: "pc-section-in 0.6s var(--ease) both" }}>
      <div style={{ maxWidth: 1180, margin: "0 auto" }}>
        <span style={{ fontFamily: "var(--font-sans)", fontSize: "11.5px", fontWeight: 700, letterSpacing: "1.5px", textTransform: "uppercase", color: "var(--accent)", display: "block", marginBottom: 28 }}>
          Scriptwriter + Critic — winning cut
        </span>
        <div data-rid="scripts-grid" style={{ display: "grid", gridTemplateColumns: "1.6fr 1fr", gap: 64, alignItems: "start" }}>
          <div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, letterSpacing: "0.5px", color: "rgba(249,244,234,0.75)", marginBottom: 14 }}>
              {winner ? winner.title : ""} · score {winner ? winner.total : ""}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 34 }}>
              {winner?.lines.map((line, i) => (
                <div key={i} style={{ fontFamily: "var(--font-serif)", fontStyle: "italic", fontSize: "clamp(24px, 3vw, 34px)", lineHeight: 1.35 }}>
                  {line}
                </div>
              ))}
            </div>
            <div style={{ width: 48, height: 2, background: "var(--accent)", marginBottom: 34 }} />

            {merge && (
              <div style={{ borderTop: "1px solid rgba(249,244,234,0.16)", paddingTop: 22 }}>
                <div style={{ fontFamily: "var(--font-sans)", fontSize: 11, fontWeight: 700, letterSpacing: "1px", textTransform: "uppercase", color: "rgba(249,244,234,0.72)", marginBottom: 12 }}>
                  Merge validation · {merge.status.toUpperCase()} · {merge.repairPath}
                </div>
                <p style={{ margin: "0 0 16px", fontSize: 13, lineHeight: 1.6, color: "rgba(249,244,234,0.72)", maxWidth: "60ch" }}>{merge.note}</p>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                  <div>
                    <div style={{ fontFamily: "var(--font-sans)", fontSize: 11, fontWeight: 700, letterSpacing: "0.8px", textTransform: "uppercase", color: "rgba(249,244,234,0.7)", marginBottom: 6 }}>
                      Before
                    </div>
                    <p style={{ margin: 0, fontSize: "12.5px", lineHeight: 1.6, color: "rgba(249,244,234,0.55)", textDecorationLine: "line-through", textDecorationColor: "rgba(249,244,234,0.25)" }}>
                      {merge.seam.before}
                    </p>
                  </div>
                  <div>
                    <div style={{ fontFamily: "var(--font-sans)", fontSize: 11, fontWeight: 700, letterSpacing: "0.8px", textTransform: "uppercase", color: "var(--accent)", marginBottom: 6 }}>
                      After
                    </div>
                    <p style={{ margin: 0, fontSize: "12.5px", lineHeight: 1.6, color: "rgba(249,244,234,0.92)" }}>{merge.seam.after}</p>
                  </div>
                </div>
                <div style={{ marginTop: 14, fontFamily: "var(--font-mono)", fontSize: 11, color: "rgba(249,244,234,0.45)" }}>
                  Meta-Critic swap: {merge.metaCriticSwapFired ? "fired" : "not required"}
                </div>
              </div>
            )}
          </div>

          <div>
            <div style={{ fontFamily: "var(--font-sans)", fontSize: 11, fontWeight: 700, letterSpacing: "1px", textTransform: "uppercase", color: "rgba(249,244,234,0.72)", marginBottom: 16 }}>
              Other variants considered
            </div>
            <div role="tablist" aria-label="Script variants" style={{ display: "flex", flexDirection: "column" }}>
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
                    onKeyDown={(e) => onTabKeyDown(e, i)}
                    className="pcs-tab"
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      gap: 10,
                      width: "100%",
                      textAlign: "left",
                      fontFamily: "var(--font-sans)",
                      fontSize: 14,
                      fontWeight: active ? 700 : 500,
                      padding: "13px 12px",
                      borderTop: "1px solid rgba(249,244,234,0.14)",
                      borderBottom: "none",
                      borderLeft: active ? "3px solid var(--accent)" : "3px solid transparent",
                      borderRight: "none",
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

            <div id={`script-panel-${activeId || ""}`} role="tabpanel" aria-labelledby={`script-tab-${activeId || ""}`} style={{ marginTop: 26 }}>
              <div style={{ fontFamily: "var(--font-sans)", fontSize: 11, fontWeight: 700, letterSpacing: "0.8px", textTransform: "uppercase", color: "rgba(249,244,234,0.72)", marginBottom: 12 }}>
                {activeScript ? activeScript.title : ""} · breakdown
              </div>
              {activeScript &&
                SCORE_META.map((m) => (
                  <div key={m.key} style={{ marginBottom: 10 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4, color: "rgba(249,244,234,0.75)" }}>
                      <span>{m.label}</span>
                      <span style={{ fontFamily: "var(--font-mono)" }}>{activeScript.scores[m.key]}</span>
                    </div>
                    <div style={{ height: 2, background: "rgba(249,244,234,0.14)" }}>
                      <div style={{ height: "100%", width: `${activeScript.scores[m.key]}%`, background: "var(--accent)", transition: "width .6s var(--ease)" }} />
                    </div>
                  </div>
                ))}
              <p style={{ margin: "14px 0 0", fontSize: 12, lineHeight: 1.6, color: "rgba(249,244,234,0.72)" }}>
                {activeScript ? activeScript.reasoning : ""}
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
