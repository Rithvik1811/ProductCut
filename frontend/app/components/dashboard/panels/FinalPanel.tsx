"use client";

import type { Final } from "@/lib/types";

interface FinalPanelProps {
  final: Final;
}

export default function FinalPanel({ final }: FinalPanelProps) {
  const maxH = 40;
  const ratioTiles = final.ratios.map((r) => {
    const ratio = r.w / r.h;
    const w = ratio >= 1 ? maxH : maxH * ratio;
    const h = ratio >= 1 ? maxH / ratio : maxH;
    return { ...r, frameW: Math.round(w), frameH: Math.round(h) };
  });

  return (
    <section data-rid="section-pad" style={{ background: "var(--inverse-bg)", color: "var(--inverse-fg)", padding: "90px 48px 100px", animation: "pc-section-in 0.6s var(--ease) both" }}>
      <div style={{ maxWidth: 1180, margin: "0 auto" }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "2px", textTransform: "uppercase", color: "var(--accent)", display: "block", marginBottom: 20 }}>
          Delivery · {final.duration}
        </span>
        <h2 style={{ fontFamily: "var(--font-serif)", fontStyle: "italic", fontWeight: 400, fontSize: "clamp(38px, 5.2vw, 66px)", margin: "0 0 48px" }}>
          Your ad is ready.
        </h2>
        <div data-rid="final-grid" style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 48, alignItems: "start" }}>
          <div style={{ position: "relative", width: "100%", aspectRatio: "16 / 9", border: "1px solid rgba(249,244,234,0.2)", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <div style={{ width: 56, height: 56, borderRadius: "50%", background: "var(--accent)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <span style={{ width: 0, height: 0, borderTop: "10px solid transparent", borderBottom: "10px solid transparent", borderLeft: "16px solid var(--accent-ink)", marginLeft: 4 }} />
            </div>
            <span style={{ position: "absolute", bottom: 12, left: 14, fontFamily: "var(--font-mono)", fontSize: 11, color: "rgba(249,244,234,0.5)" }}>
              preview · master.mp4
            </span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {ratioTiles.map((tile) => (
              <div key={tile.id} style={{ display: "flex", alignItems: "center", gap: 14, borderTop: "1px solid rgba(249,244,234,0.16)", paddingTop: 14 }}>
                <div style={{ width: tile.frameW, height: tile.frameH, border: "1px solid rgba(249,244,234,0.3)", flex: "0 0 auto" }} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: "13.5px", fontWeight: 600 }}>
                    {tile.id} · {tile.label}
                  </div>
                  <div style={{ fontSize: 11, color: "rgba(249,244,234,0.72)", marginTop: 2 }}>{tile.use}</div>
                </div>
                <button
                  className="pcs-final-dl"
                  style={{
                    fontFamily: "var(--font-sans)",
                    fontSize: "11.5px",
                    fontWeight: 600,
                    padding: "8px 12px",
                    border: "1px solid var(--accent)",
                    background: "transparent",
                    color: "var(--accent)",
                    cursor: "pointer",
                  }}
                >
                  ↓
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
