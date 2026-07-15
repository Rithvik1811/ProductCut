"use client";

import type { CSSProperties } from "react";
import type { Shot, ShotStatus } from "@/lib/types";

interface ShotsPanelProps {
  shots: Shot[];
  shotOpenId: string | null;
  onToggle: (id: string) => void;
}

function statusMeta(status: ShotStatus | undefined) {
  if (status === "done") return { label: "done", color: "var(--ink-soft)" };
  if (status === "fallback") return { label: "fallback", color: "var(--accent)" };
  if (status === "generating") return { label: "generating", color: "var(--ink)" };
  if (status === "retrying") return { label: "retry", color: "var(--warn)" };
  return { label: "queued", color: "var(--faint)" };
}

export default function ShotsPanel({ shots, shotOpenId, onToggle }: ShotsPanelProps) {
  const shotsDone = shots.filter((x) => x.status === "done" || x.status === "fallback").length;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 12 }}>
        <span style={{ fontFamily: "var(--font-sans)", fontSize: "11.5px", fontWeight: 700, letterSpacing: "0.8px", textTransform: "uppercase", color: "var(--ink-soft)" }}>
          Shots
        </span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--ink)" }}>
          {shotsDone} / {shots.length}
        </span>
      </div>
      <div data-rid="shots-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        {shots.map((sh) => {
          const m = statusMeta(sh.status);
          const isFallback = sh.status === "fallback";
          const isReal = sh.status === "done";
          const isGenerating = sh.status === "generating" || sh.status === "retrying";
          const open = shotOpenId === sh.id;
          const thumbBg: CSSProperties = isFallback
            ? { backgroundColor: "transparent", backgroundImage: "repeating-linear-gradient(45deg, var(--accent) 0 6px, var(--paper) 6px 12px)" }
            : isReal
              ? { backgroundColor: "var(--ink)", backgroundImage: "none" }
              : { backgroundColor: "rgba(18,52,59,0.06)", backgroundImage: "none" };

          return (
            <div key={sh.id} onClick={() => onToggle(sh.id)} style={{ border: "1px solid var(--hair-strong)", cursor: "pointer", background: "var(--paper)" }}>
              <div style={{ position: "relative", width: "100%", aspectRatio: "16 / 10", display: "flex", alignItems: "center", justifyContent: "center", borderBottom: "1px solid var(--hair)", ...thumbBg }}>
                {isGenerating && (
                  <span style={{ width: 14, height: 14, borderRadius: "50%", border: "2px solid rgba(18,52,59,0.2)", borderTopColor: "var(--accent)", display: "block", animation: "pc-spin 0.75s linear infinite" }} />
                )}
                {isFallback && (
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.4px", color: "var(--ink)", position: "absolute", top: 5, left: 6, background: "var(--paper)", padding: "1px 4px" }}>
                    KEN-BURNS
                  </span>
                )}
              </div>
              <div style={{ padding: "7px 8px" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 4 }}>
                  <span style={{ fontSize: "11.5px", fontWeight: 600, color: "var(--ink)" }}>{sh.label}</span>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: m.color }}>{m.label}</span>
                </div>
                {open && (
                  <div style={{ marginTop: 6, fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--ink-soft)", lineHeight: 1.5 }}>
                    {sh.camera} · {sh.move} · {sh.duration}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
