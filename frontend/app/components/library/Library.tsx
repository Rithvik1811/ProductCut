"use client";

import type { HistoryEntry } from "@/lib/types";

interface LibraryProps {
  history: HistoryEntry[];
  onClose: () => void;
  onOpen: (entry: HistoryEntry) => void;
}

export default function Library({ history, onClose, onOpen }: LibraryProps) {
  return (
    <main style={{ maxWidth: 1020, margin: "0 auto", padding: "56px 56px 90px" }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 44 }}>
        <h1 style={{ fontFamily: "var(--font-serif)", fontStyle: "italic", fontWeight: 400, fontSize: "clamp(32px, 4.4vw, 48px)", margin: 0 }}>
          My ads
        </h1>
        <button
          onClick={onClose}
          className="pcs-hover-ink"
          style={{ fontFamily: "var(--font-sans)", fontSize: 13, fontWeight: 600, color: "var(--ink-soft)", background: "transparent", border: "none", borderBottom: "1px solid var(--hair-strong)", cursor: "pointer", padding: "2px 0" }}
        >
          ← Back
        </button>
      </div>

      {history.length === 0 && (
        <p style={{ fontSize: "14.5px", color: "var(--ink-soft)", fontStyle: "italic" }}>
          No finished ads yet — completed jobs will show up here.
        </p>
      )}

      <div style={{ display: "flex", flexDirection: "column" }}>
        {history.map((job, i) => (
          <div
            key={`${job.productName}-${job.date}-${i}`}
            data-rid="library-row"
            onClick={() => onOpen(job)}
            style={{ display: "grid", gridTemplateColumns: "84px 1fr auto", gap: 20, alignItems: "center", padding: "18px 4px", borderTop: "1px solid var(--hair)", cursor: "pointer" }}
          >
            <div style={{ width: 84, aspectRatio: "16 / 9", background: "var(--ink)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <span style={{ width: 0, height: 0, borderTop: "6px solid transparent", borderBottom: "6px solid transparent", borderLeft: "10px solid var(--accent)", marginLeft: 2 }} />
            </div>
            <div>
              <div style={{ fontFamily: "var(--font-serif)", fontStyle: "italic", fontSize: 18, color: "var(--ink)" }}>{job.productName}</div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--faint)", marginTop: 3 }}>
                {new Date(job.date).toLocaleDateString(undefined, { month: "short", day: "numeric" })} · {job.truths.length} facts
                {job.final ? ` · ${job.final.duration}` : ""}
              </div>
            </div>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "11.5px", color: "var(--accent)" }}>View →</span>
          </div>
        ))}
      </div>
    </main>
  );
}
