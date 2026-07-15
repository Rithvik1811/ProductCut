"use client";

import type { ChangeEvent, KeyboardEvent } from "react";
import TagField from "./TagField";

interface DirectionStepProps {
  moodWords: string[];
  moodInput: string;
  onMoodInput: (value: string) => void;
  onMoodKey: (e: KeyboardEvent<HTMLInputElement>) => void;
  onRemoveMood: (i: number) => void;
  refLink: string;
  onRefInput: (e: ChangeEvent<HTMLInputElement>) => void;
  neverList: string[];
  neverInput: string;
  onNeverInput: (value: string) => void;
  onNeverKey: (e: KeyboardEvent<HTMLInputElement>) => void;
  onRemoveNever: (i: number) => void;
  notes: string;
  onNotesInput: (e: ChangeEvent<HTMLTextAreaElement>) => void;
}

export default function DirectionStep({
  moodWords,
  moodInput,
  onMoodInput,
  onMoodKey,
  onRemoveMood,
  refLink,
  onRefInput,
  neverList,
  neverInput,
  onNeverInput,
  onNeverKey,
  onRemoveNever,
  notes,
  onNotesInput,
}: DirectionStepProps) {
  const labelStyle = {
    display: "block",
    fontFamily: "var(--font-sans)",
    fontSize: 12,
    fontWeight: 700,
    letterSpacing: "1.2px",
    textTransform: "uppercase" as const,
    color: "var(--faint)",
    marginBottom: 12,
  };

  return (
    <div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 44 }}>
        <h1 style={{ fontFamily: "var(--font-serif)", fontWeight: 400, fontStyle: "italic", fontSize: 34, margin: 0 }}>
          Any direction for us?
        </h1>
        <span style={{ fontFamily: "var(--font-sans)", fontSize: 12, fontWeight: 600, letterSpacing: "1.5px", textTransform: "uppercase", color: "var(--faint)" }}>
          optional
        </span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 34, maxWidth: 640 }}>
        <div>
          <label style={labelStyle}>Mood words</label>
          <TagField
            tags={moodWords}
            variant="solid"
            inputValue={moodInput}
            placeholder="type a word, press enter…"
            onInputChange={onMoodInput}
            onKeyDown={onMoodKey}
            onRemove={onRemoveMood}
          />
        </div>
        <div>
          <label style={labelStyle}>Reference ad</label>
          <input
            type="url"
            value={refLink}
            onChange={onRefInput}
            placeholder="https://…"
            className="pcs-underline-input"
            style={{
              width: "100%",
              padding: "0 0 10px",
              fontFamily: "var(--font-sans)",
              fontSize: 15,
              color: "var(--ink)",
              background: "transparent",
              border: "none",
              borderBottom: "1px solid var(--hair-strong)",
            }}
          />
        </div>
        <div>
          <label style={labelStyle}>Never do</label>
          <TagField
            tags={neverList}
            variant="outline"
            inputValue={neverInput}
            placeholder="type a rule, press enter…"
            onInputChange={onNeverInput}
            onKeyDown={onNeverKey}
            onRemove={onRemoveNever}
          />
        </div>
        <div>
          <label style={labelStyle}>Freeform notes</label>
          <textarea
            value={notes}
            onChange={onNotesInput}
            rows={2}
            placeholder="Your story, the buyer you picture, a line you always say…"
            className="pcs-underline-input"
            style={{
              width: "100%",
              padding: "0 0 10px",
              fontFamily: "var(--font-sans)",
              fontSize: 15,
              color: "var(--ink)",
              background: "transparent",
              border: "none",
              borderBottom: "1px solid var(--hair-strong)",
              resize: "vertical",
              lineHeight: 1.6,
            }}
          />
        </div>
      </div>
    </div>
  );
}
