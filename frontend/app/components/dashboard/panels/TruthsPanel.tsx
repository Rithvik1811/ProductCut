"use client";

import type { Truth } from "@/lib/types";
import { categoryLabel } from "../shared";

interface TruthsPanelProps {
  truths: Truth[];
  hoveredTruthId: string | null;
  onHoverTruth: (id: string | null) => void;
}

export default function TruthsPanel({ truths, hoveredTruthId, onHoverTruth }: TruthsPanelProps) {
  return (
    <section data-rid="section-pad" style={{ maxWidth: 1180, margin: "0 auto", padding: "72px 48px 56px", animation: "pc-section-in 0.6s var(--ease) both" }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 40 }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "2px", textTransform: "uppercase", color: "var(--faint)" }}>
          Truth Agent — grounding facts
        </span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--faint)" }}>{truths.length} facts</span>
      </div>
      <div>
        {truths.map((truth, i) => (
          <div
            key={truth.id}
            onMouseEnter={() => onHoverTruth(truth.id)}
            onMouseLeave={() => onHoverTruth(null)}
            style={{
              display: "flex",
              gap: 24,
              alignItems: "flex-start",
              padding: "26px 12px",
              borderTop: i === 0 ? "1px solid var(--hair-strong)" : "1px solid var(--hair)",
              transition: "background-color .3s var(--ease)",
              background: hoveredTruthId === truth.id ? "var(--paper-deep)" : "transparent",
            }}
          >
            <span style={{ fontFamily: "var(--font-serif)", fontStyle: "italic", fontSize: 38, color: "var(--faint)", flex: "0 0 68px", lineHeight: 1 }}>
              {String(i + 1).padStart(2, "0")}
            </span>
            <div style={{ flex: 1 }}>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "1.2px", textTransform: "uppercase", color: "var(--ink-soft)", marginBottom: 6 }}>
                {categoryLabel(truth.category)}
              </div>
              <p style={{ margin: 0, fontFamily: "var(--font-serif)", fontSize: 23, lineHeight: 1.4, color: "var(--ink)", maxWidth: "52ch" }}>
                {truth.fact_text}
              </p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
