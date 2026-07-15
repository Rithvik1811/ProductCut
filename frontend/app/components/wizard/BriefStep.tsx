"use client";

import type { ChangeEvent, KeyboardEvent } from "react";

interface BriefStepProps {
  brief: string;
  onBriefInput: (e: ChangeEvent<HTMLInputElement>) => void;
  onBriefKey: (e: KeyboardEvent<HTMLInputElement>) => void;
}

export default function BriefStep({ brief, onBriefInput, onBriefKey }: BriefStepProps) {
  return (
    <div style={{ paddingTop: 24 }}>
      <p style={{ margin: "0 0 24px", fontFamily: "var(--font-mono)", fontSize: 12, letterSpacing: "0.5px", color: "var(--faint)", maxWidth: "40ch" }}>
        What is it, and what should the ad feel like? One honest sentence is plenty.
      </p>
      <input
        type="text"
        value={brief}
        onChange={onBriefInput}
        onKeyDown={onBriefKey}
        placeholder="handmade ceramic mugs, cozy autumn vibe"
        className="pcs-underline-input"
        style={{
          width: "100%",
          padding: "0 0 20px",
          fontFamily: "var(--font-serif)",
          fontStyle: "italic",
          fontSize: "clamp(30px, 4.8vw, 52px)",
          lineHeight: 1.2,
          color: "var(--ink)",
          background: "transparent",
          border: "none",
          borderBottom: "2px solid var(--hair-strong)",
        }}
      />
      <p style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.3px", color: "var(--faint)", margin: "18px 0 0" }}>
        e.g. &quot;small-batch soy candles, quiet Sunday mornings&quot;
      </p>
    </div>
  );
}
