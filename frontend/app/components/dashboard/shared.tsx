import type { ScoreKey } from "@/lib/types";

export const SCORE_META: { key: ScoreKey; label: string }[] = [
  { key: "hook", label: "Hook" },
  { key: "pacing", label: "Pacing" },
  { key: "completion", label: "Completion" },
  { key: "cta", label: "CTA" },
  { key: "tone", label: "Tone" },
];

export const CATEGORY: Record<string, string> = {
  material: "Material",
  color: "Color",
  texture: "Texture",
  distinguishing_mark: "Mark",
  size: "Size",
  condition: "Condition",
  shape: "Shape",
};

export function categoryLabel(cat: string): string {
  return CATEGORY[cat] ?? cat;
}

// Sections alternate this max-width + horizontal-padding wrapper; vertical
// padding differs per section so each caller sets it directly.
export const sectionMaxWidth = 1180;
