"use client";

import type { Photo } from "./types";

interface SummaryStepProps {
  photos: Photo[];
  brief: string;
  moodWords: string[];
  neverList: string[];
  notes: string;
  onEditPhotos: () => void;
  onEditBrief: () => void;
  onEditDirection: () => void;
}

export default function SummaryStep({
  photos,
  brief,
  moodWords,
  neverList,
  notes,
  onEditPhotos,
  onEditBrief,
  onEditDirection,
}: SummaryStepProps) {
  const photoCountLabel = photos.length ? `${photos.length} / 3 added` : "up to 3";
  const hasMood = moodWords.length > 0;
  const hasNever = neverList.length > 0;
  const hasNotes = !!notes.trim();
  const hasDirection = hasMood || hasNever || hasNotes;

  const sectionHeadStyle = { display: "flex", alignItems: "center", justifyContent: "space-between" };
  const labelStyle = {
    fontFamily: "var(--font-sans)",
    fontSize: 12,
    fontWeight: 700,
    letterSpacing: "1px",
    textTransform: "uppercase" as const,
    color: "var(--ink-soft)",
  };
  const editBtnStyle = {
    fontFamily: "var(--font-sans)",
    fontSize: "12.5px",
    fontWeight: 600,
    color: "var(--ink)",
    background: "none",
    border: "none",
    borderBottom: "1px solid var(--ink)",
    cursor: "pointer",
    padding: 0,
  };
  const rowLabelStyle = { fontSize: 12, color: "var(--ink-soft)", minWidth: 68 };

  return (
    <div data-rid="step4-summary" style={{ background: "var(--paper-deep)", margin: "0 -40px", padding: "44px 40px" }}>
      <h1 style={{ fontFamily: "var(--font-serif)", fontWeight: 400, fontStyle: "italic", fontSize: 36, margin: "0 0 36px" }}>
        Here&apos;s your brief.
      </h1>
      <div style={{ display: "flex", flexDirection: "column", gap: 30 }}>
        <div>
          <div style={{ ...sectionHeadStyle, marginBottom: 14 }}>
            <span style={labelStyle}>Photos · {photoCountLabel}</span>
            <button onClick={onEditPhotos} style={editBtnStyle}>
              Edit
            </button>
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            {photos.map((photo) => (
              <div key={photo.url} style={{ width: 74, height: 74, overflow: "hidden", border: "1px solid var(--ink)" }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={photo.url} alt={photo.name} style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} />
              </div>
            ))}
          </div>
        </div>

        <div>
          <div style={{ ...sectionHeadStyle, marginBottom: 12 }}>
            <span style={labelStyle}>The brief</span>
            <button onClick={onEditBrief} style={editBtnStyle}>
              Edit
            </button>
          </div>
          <p style={{ margin: 0, fontFamily: "var(--font-serif)", fontStyle: "italic", fontSize: 26, lineHeight: 1.35, color: "var(--ink)" }}>
            &quot;{brief}&quot;
          </p>
        </div>

        <div>
          <div style={{ ...sectionHeadStyle, marginBottom: 12 }}>
            <span style={labelStyle}>Seller direction</span>
            <button onClick={onEditDirection} style={editBtnStyle}>
              Edit
            </button>
          </div>
          {hasDirection ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {hasMood && (
                <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
                  <span style={rowLabelStyle}>Mood</span>
                  <span style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {moodWords.map((tag, i) => (
                      <span key={i} style={{ padding: "2px 9px", border: "1px solid var(--ink)", color: "var(--ink)", fontSize: 12 }}>
                        {tag}
                      </span>
                    ))}
                  </span>
                </div>
              )}
              {hasNever && (
                <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
                  <span style={rowLabelStyle}>Never</span>
                  <span style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {neverList.map((tag, i) => (
                      <span key={i} style={{ padding: "2px 9px", border: "1px solid var(--accent)", color: "var(--ink)", fontSize: 12 }}>
                        {tag}
                      </span>
                    ))}
                  </span>
                </div>
              )}
              {hasNotes && (
                <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
                  <span style={rowLabelStyle}>Notes</span>
                  <span style={{ fontSize: "13.5px", color: "var(--ink)", lineHeight: 1.5 }}>{notes}</span>
                </div>
              )}
            </div>
          ) : (
            <p style={{ margin: 0, fontSize: 14, color: "var(--ink-soft)", fontStyle: "italic" }}>No additional direction provided.</p>
          )}
        </div>
      </div>
    </div>
  );
}
