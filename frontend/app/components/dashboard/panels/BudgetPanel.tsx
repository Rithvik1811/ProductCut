"use client";

import type { Budget } from "@/lib/types";

interface BudgetPanelProps {
  budget: Budget & { running: number };
  budgetOpenId: string | null;
  onToggle: (id: string) => void;
}

export default function BudgetPanel({ budget, budgetOpenId, onToggle }: BudgetPanelProps) {
  const bCap = budget.cap || 1;
  const bRun = budget.running;
  const bPct = Math.min(bRun / bCap, 1) * 100;
  const overRatio = bRun / bCap;
  const budgetColor = overRatio > 1 ? "var(--over)" : overRatio > 0.85 ? "var(--warn)" : "var(--ink)";
  const budgetStateLabel = overRatio > 1 ? "over cap" : overRatio > 0.85 ? "approaching cap" : "within budget";

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 12 }}>
        <span style={{ fontFamily: "var(--font-sans)", fontSize: "11.5px", fontWeight: 700, letterSpacing: "0.8px", textTransform: "uppercase", color: "var(--ink-soft)" }}>
          Budget
        </span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--ink)" }}>
          ${bRun.toFixed(2)} / ${(budget.cap || 0).toFixed(2)}
        </span>
      </div>
      <div style={{ height: 3, background: "rgba(18,52,59,0.14)", marginBottom: 4 }}>
        <div style={{ height: "100%", width: "100%", background: budgetColor, transform: `scaleX(${bPct / 100})`, transformOrigin: "left", transition: "transform .6s var(--ease)" }} />
      </div>
      <div style={{ fontFamily: "var(--font-sans)", fontSize: 11, fontWeight: 600, color: "var(--ink-soft)", marginBottom: 16 }}>
        {budgetStateLabel}
      </div>
      <div>
        {budget.shots.map((sh) => {
          const open = budgetOpenId === sh.id;
          return (
            <div
              key={sh.id}
              onClick={() => onToggle(sh.id)}
              style={{ padding: "12px 4px", borderTop: "1px solid var(--hair)", cursor: "pointer", minHeight: 44 }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 12, color: "var(--ink)", flex: 1 }}>{sh.label}</span>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--ink-soft)" }}>{sh.alloc}</span>
              </div>
              {open && (
                <p style={{ margin: "6px 0 0", fontSize: "11.5px", lineHeight: 1.5, color: "var(--ink-soft)" }}>{sh.justification}</p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
