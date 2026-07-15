"use client";

import type { ChangeEvent, DragEvent, RefObject } from "react";
import type { Photo } from "./types";

interface PhotosStepProps {
  photos: Photo[];
  hoveredPhoto: number | null;
  dragOver: boolean;
  fileRef: RefObject<HTMLInputElement | null>;
  onPickClick: () => void;
  onDragOver: (e: DragEvent<HTMLDivElement>) => void;
  onDragLeave: (e: DragEvent<HTMLDivElement>) => void;
  onDrop: (e: DragEvent<HTMLDivElement>) => void;
  onFileChange: (e: ChangeEvent<HTMLInputElement>) => void;
  onPhotoEnter: (i: number) => void;
  onPhotoLeave: () => void;
  onRemovePhoto: (i: number) => void;
}

export default function PhotosStep({
  photos,
  hoveredPhoto,
  dragOver,
  fileRef,
  onPickClick,
  onDragOver,
  onDragLeave,
  onDrop,
  onFileChange,
  onPhotoEnter,
  onPhotoLeave,
  onRemovePhoto,
}: PhotosStepProps) {
  const photoCountLabel = photos.length ? `${photos.length} / 3 added` : "up to 3";
  const ruleColor = dragOver ? "var(--accent)" : "var(--hair-strong)";

  return (
    <div>
      <h1
        style={{
          fontFamily: "var(--font-serif)",
          fontWeight: 400,
          fontSize: "clamp(42px, 6.4vw, 84px)",
          lineHeight: 0.98,
          letterSpacing: "-1.5px",
          margin: "0 0 20px",
          maxWidth: "15ch",
        }}
      >
        Show us the <em style={{ fontStyle: "italic", color: "var(--accent)" }}>real</em> thing.
      </h1>
      <p style={{ margin: "0 0 40px", fontSize: "16.5px", lineHeight: 1.6, color: "var(--ink-soft)", maxWidth: "42ch" }}>
        Upload two or three clear photos. These are what we read — no stock, no rendering, just what you actually
        made.
      </p>

      <div
        onClick={onPickClick}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className="pcs-hover-ink"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 24,
          padding: "38px 0",
          cursor: "pointer",
          borderTop: `1px solid ${ruleColor}`,
          borderBottom: `1px solid ${ruleColor}`,
          transition: "border-color .2s ease",
        }}
      >
        <div>
          <div style={{ fontFamily: "var(--font-serif)", fontStyle: "italic", fontSize: 23, color: "var(--ink)" }}>
            Drop your product photos here
          </div>
          <div style={{ fontSize: 13, color: "var(--faint)", marginTop: 6, letterSpacing: "0.2px" }}>
            or browse your files — clear, well-lit shots read best
          </div>
        </div>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, letterSpacing: "0.5px", color: "var(--faint)", whiteSpace: "nowrap" }}>
          {photoCountLabel}
        </div>
        <input type="file" ref={fileRef} accept="image/*" multiple onChange={onFileChange} style={{ display: "none" }} />
      </div>

      <div style={{ display: photos.length ? "flex" : "none", gap: 10, marginTop: 20, flexWrap: "wrap" }}>
        {photos.map((photo, i) => (
          <div
            key={photo.url}
            onMouseEnter={() => onPhotoEnter(i)}
            onMouseLeave={onPhotoLeave}
            style={{
              position: "relative",
              width: 84,
              height: 84,
              overflow: "hidden",
              border: "1px solid var(--hair-strong)",
              flex: "0 0 auto",
              animation: "pc-rise 0.4s var(--ease)",
            }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={photo.url}
              alt={photo.name}
              style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
            />
            <div
              onClick={() => onRemovePhoto(i)}
              style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                background: "rgba(18,52,59,0.6)",
                opacity: hoveredPhoto === i ? 1 : 0,
                transition: "opacity .3s var(--ease)",
              }}
            >
              <span
                style={{
                  fontFamily: "var(--font-sans)",
                  fontSize: 11,
                  fontWeight: 700,
                  letterSpacing: "0.5px",
                  textTransform: "uppercase",
                  color: "#f9f4ea",
                }}
              >
                Remove
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
