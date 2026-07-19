// The 8 pipeline stages shown in the dashboard's phase nav, in graph order.
// Index into this array is the single source of truth for "how far the real
// backend has gotten" -- see studio/page.tsx's maxPhaseIdx (bumped only by
// real C2 events actually received, never guessed ahead) and Dashboard.tsx
// (renders done/active/not-started off that same index).
export const PHASES = [
  "Ingest",
  "Truths",
  "Scripts",
  "Treatment",
  "Budget",
  "Shots",
  "Continuity",
  "Delivery",
] as const;

export type Phase = (typeof PHASES)[number];

// Human-readable "what's happening right now" text for the status line,
// keyed by the phase currently in progress (PHASES[curPhaseIdx]).
export const PHASE_RUNNING_LABEL: Record<Phase, string> = {
  Ingest: "Reading your photos & brief…",
  Truths: "Extracting product truths…",
  Scripts: "Writing & scoring scripts…",
  Treatment: "Directing visual treatment…",
  Budget: "Allocating shot budget…",
  Shots: "Generating shots…",
  Continuity: "Checking continuity…",
  Delivery: "Assembling the final cut…",
};

// LangGraph node name -> phase, for the (currently unwired but frozen-C2)
// "node_started" event -- see graph/events.py's NodeStartedPayload. Kept so
// handleEvent can respect it correctly if a future backend change starts
// emitting it, without guessing a sequence in the meantime.
export const NODE_TO_PHASE: Record<string, Phase> = {
  ingest_node: "Ingest",
  brand_research_node: "Ingest",
  product_truth_extractor: "Truths",
  concept_agent: "Scripts",
  hook_checker: "Scripts",
  pacing_checker: "Scripts",
  body_checker: "Scripts",
  cta_checker: "Scripts",
  tone_checker: "Scripts",
  meta_critic: "Scripts",
  merge_validator: "Scripts",
  copy_editor: "Scripts",
  visual_direction_agent: "Treatment",
  treatment_agent: "Treatment",
  shot_list_agent: "Budget",
  budget_gate: "Budget",
  video_gen_node: "Shots",
  ken_burns_fallback_node: "Shots",
  continuity_agent: "Continuity",
  continuity_gate: "Continuity",
  voiceover_caption_agent: "Delivery",
  assembly_agent: "Delivery",
  format_export_node: "Delivery",
};

// Realistic wall-clock estimate for the whole run, grounded in the derisk
// numbers already collected in this codebase: video-gen alone runs 42-99s
// per clip (docs/DERISK_VIDEO_GEN_RESULT.md), and full real pipeline runs
// have taken 20-45+ minutes end to end in testing. Before the shot count is
// known (Budget phase hasn't settled yet) we can't do better than that
// generic range; once it is, refine using the per-clip range plus a fixed
// overhead for every non-video-gen phase (ingest, scripts, treatment,
// budget, continuity, assembly, export).
const FIXED_OVERHEAD_MIN = 6;
const CLIP_MIN_SEC = 42;
const CLIP_MAX_SEC = 99;

export function estimateDuration(shotCount: number): string {
  if (!shotCount) return "~15–30 min";
  const lowMin = Math.round((shotCount * CLIP_MIN_SEC) / 60 + FIXED_OVERHEAD_MIN);
  const highMin = Math.round((shotCount * CLIP_MAX_SEC) / 60 + FIXED_OVERHEAD_MIN);
  return `~${lowMin}–${highMin} min`;
}
