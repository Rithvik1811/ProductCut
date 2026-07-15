"use client";

import type { Interrupt, InterruptResolution, Shot } from "@/lib/types";

interface ContinuityPanelProps {
  shots: Shot[];
  drift: Record<string, number>;
  driftThreshold: number;
  interrupt: Interrupt | null;
  interruptResolution: InterruptResolution | null;
  onApprove: () => void;
  onRetry: () => void;
  onFallback: () => void;
}

const candThumbStyle = { width: "100%", aspectRatio: "1 / 1", background: "rgba(18,52,59,0.05)" };

const actionBtnBase = {
  fontFamily: "var(--font-sans)",
  fontSize: "11.5px",
  fontWeight: 600,
  padding: "8px 12px",
  border: "1px solid var(--hair-strong)",
  background: "transparent",
  color: "var(--ink)",
  cursor: "pointer",
};

export default function ContinuityPanel({
  shots,
  drift,
  driftThreshold,
  interrupt,
  interruptResolution,
  onApprove,
  onRetry,
  onFallback,
}: ContinuityPanelProps) {
  const driftRows = shots
    .filter((sh) => drift[sh.id] != null)
    .map((sh) => {
      const score = drift[sh.id];
      const ok = score <= driftThreshold;
      return { id: sh.id, label: sh.label, score, ok };
    });
  const interruptResolvedShown = !!interruptResolution && !interrupt;

  return (
    <div>
      <span style={{ fontFamily: "var(--font-sans)", fontSize: "11.5px", fontWeight: 700, letterSpacing: "0.8px", textTransform: "uppercase", color: "var(--ink-soft)", display: "block", marginBottom: 12 }}>
        Continuity
      </span>

      {interrupt && (
        <div style={{ border: "1.5px solid var(--accent)", padding: 14, marginBottom: 14, background: "var(--paper)" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "11.5px", letterSpacing: "0.5px", color: "var(--accent)", fontWeight: 700, marginBottom: 8 }}>
            ⚠ Human review · {interrupt.label} · drift {interrupt.driftScore.toFixed(2)}
          </div>
          <p style={{ margin: "0 0 12px", fontSize: 12, lineHeight: 1.5, color: "var(--ink)" }}>{interrupt.reason}</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 6, marginBottom: 12 }}>
            {interrupt.candidates.map((c) => (
              <div key={c.id} style={{ border: "1px solid var(--hair-strong)" }}>
                <div style={candThumbStyle} />
                <div style={{ padding: "5px 6px", fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--ink-soft)" }}>{c.note}</div>
              </div>
            ))}
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button
              onClick={onApprove}
              style={{ fontFamily: "var(--font-sans)", fontSize: "11.5px", fontWeight: 700, padding: "8px 12px", border: "none", cursor: "pointer", background: "var(--accent)", color: "var(--accent-ink)" }}
            >
              Approve
            </button>
            <button onClick={onRetry} className="pcs-hover-ink" style={actionBtnBase}>
              Retry
            </button>
            <button onClick={onFallback} className="pcs-hover-ink" style={actionBtnBase}>
              Accept fallback
            </button>
          </div>
        </div>
      )}

      {interruptResolvedShown && (
        <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--ink)", marginBottom: 12 }}>
          ✓ Resolved — {interruptResolution}
        </div>
      )}

      <div>
        {driftRows.map((row) => (
          <div key={row.id} style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 0", borderTop: "1px solid var(--hair)" }}>
            <span style={{ fontSize: 12, color: "var(--ink)", flex: 1 }}>{row.label}</span>
            <div style={{ width: 60, height: 3, background: "rgba(18,52,59,0.14)" }}>
              <div style={{ height: "100%", width: `${Math.min((row.score / 0.6) * 100, 100)}%`, background: row.ok ? "var(--ink-soft)" : "var(--over)" }} />
            </div>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, minWidth: 30, textAlign: "right", color: row.ok ? "var(--ink-soft)" : "var(--over)" }}>
              {row.score.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
