import type { ScoreKey } from "@/lib/types";

export const SCORE_META: { key: ScoreKey; label: string }[] = [
  { key: "hook", label: "Hook" },
  { key: "pacing", label: "Pacing" },
  { key: "completion", label: "Completion" },
  { key: "cta", label: "CTA" },
  { key: "tone", label: "Tone" },
];

// Every axis (hook/pacing/completion/cta/tone) and the composite are scored
// by the backend on a shared 1-5 rubric (agents/meta_critic.py), which
// studio/page.tsx's critic_score handler multiplies by 100 for display —
// so the real max a displayed score can reach is 5 * 100 = 500, NOT 100.
// Bar widths must divide by this, not treat the raw number as a percentage.
export const SCORE_MAX = 500;

export const CATEGORY: Record<string, string> = {
  // Real backend categories (graph/state.py v11)
  color: "Color",
  material: "Material",
  texture: "Texture",
  construction_detail: "Construction",
  material_character: "Character",
  scale_cue: "Scale",
  brief_or_intake_fact: "Brief",
  form_factor: "Form factor",
};

export function categoryLabel(cat: string): string {
  return CATEGORY[cat] ?? cat;
}

// Sections alternate this max-width + horizontal-padding wrapper; vertical
// padding differs per section so each caller sets it directly.
export const sectionMaxWidth = 1180;
