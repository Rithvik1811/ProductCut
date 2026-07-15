"use client";

import type { Treatment, Truth } from "@/lib/types";
import { categoryLabel } from "../shared";

interface TreatmentPanelProps {
  treatment: Treatment;
  truths: Truth[];
  hoveredTruthId: string | null;
  onHoverTruth: (id: string | null) => void;
}

export default function TreatmentPanel({ treatment, truths, hoveredTruthId, onHoverTruth }: TreatmentPanelProps) {
  const facets = [
    { label: "Persona", value: treatment.director_persona },
    { label: "Color story", value: treatment.color_story },
    { label: "Pacing", value: treatment.pacing_philosophy },
  ];

  return (
    <section data-rid="section-pad" style={{ maxWidth: 1180, margin: "0 auto", padding: "76px 48px 60px", animation: "pc-section-in 0.6s var(--ease) both" }}>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "2px", textTransform: "uppercase", color: "var(--faint)", display: "block", marginBottom: 40 }}>
        Director — the treatment
      </span>
      <div data-rid="facet-grid-wrap" style={{ display: "flex", flexDirection: "column", gap: 22, marginBottom: 52, maxWidth: 760 }}>
        {facets.map((f) => (
          <div key={f.label} data-rid="facet-grid" style={{ display: "grid", gridTemplateColumns: "130px 1fr", gap: 24 }}>
            <span style={{ fontFamily: "var(--font-sans)", fontSize: 12, fontWeight: 700, letterSpacing: "1px", textTransform: "uppercase", color: "var(--ink-soft)", paddingTop: 3 }}>
              {f.label}
            </span>
            <p style={{ margin: 0, fontFamily: "var(--font-serif)", fontStyle: "italic", fontSize: 19, lineHeight: 1.5, color: "var(--ink)" }}>{f.value}</p>
          </div>
        ))}
      </div>
      <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "1px", textTransform: "uppercase", color: "var(--faint)", marginBottom: 18 }}>
        Beat by beat — hover to trace the truth
      </div>
      <div style={{ display: "flex", flexDirection: "column" }}>
        {treatment.beats.map((beat) => {
          const truth = truths.find((t) => t.id === beat.truth_fact_id);
          const label = truth ? categoryLabel(truth.category) : beat.truth_fact_id;
          const activeLink = hoveredTruthId === beat.truth_fact_id;
          return (
            <div
              key={beat.id}
              data-rid="beat-row"
              onMouseEnter={() => onHoverTruth(beat.truth_fact_id)}
              onMouseLeave={() => onHoverTruth(null)}
              style={{ display: "flex", gap: 16, alignItems: "baseline", justifyContent: "space-between", padding: "16px 4px", borderTop: "1px solid var(--hair)" }}
            >
              <span style={{ fontFamily: "var(--font-serif)", fontStyle: "italic", fontSize: 17, color: "var(--ink)", flex: 1 }}>
                &quot;{beat.script_quote}&quot;
              </span>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "11.5px", letterSpacing: "0.5px", textTransform: "uppercase", whiteSpace: "nowrap", color: activeLink ? "var(--accent)" : "var(--faint)" }}>
                → {label}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}
