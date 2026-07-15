"use client";

import type { KeyboardEvent } from "react";

interface TagFieldProps {
  tags: string[];
  variant: "solid" | "outline";
  inputValue: string;
  placeholder: string;
  onInputChange: (value: string) => void;
  onKeyDown: (e: KeyboardEvent<HTMLInputElement>) => void;
  onRemove: (index: number) => void;
}

export default function TagField({
  tags,
  variant,
  inputValue,
  placeholder,
  onInputChange,
  onKeyDown,
  onRemove,
}: TagFieldProps) {
  const tagBorder = variant === "outline" ? "1px solid var(--accent)" : "1px solid var(--hair-strong)";
  const removeColor = variant === "outline" ? "var(--accent)" : "var(--faint)";

  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
      {tags.map((tag, i) => (
        <span
          key={i}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            padding: "4px 5px 4px 11px",
            border: tagBorder,
            color: "var(--ink)",
            fontSize: 13,
          }}
        >
          {tag}
          <button
            onClick={() => onRemove(i)}
            style={{ border: "none", background: "none", cursor: "pointer", color: removeColor, fontSize: 14, lineHeight: 1, padding: "0 2px" }}
          >
            ×
          </button>
        </span>
      ))}
      <input
        type="text"
        value={inputValue}
        onChange={(e) => onInputChange(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        className="pcs-underline-input"
        style={{
          border: "none",
          borderBottom: "1px solid var(--hair-strong)",
          background: "transparent",
          fontFamily: "var(--font-sans)",
          fontSize: 14,
          color: "var(--ink)",
          padding: "6px 2px",
          minWidth: 160,
          marginTop: 8,
        }}
      />
    </div>
  );
}
