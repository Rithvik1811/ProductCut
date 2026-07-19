// ProductCut — shared types aligned with the real backend C2 contract (graph/events.py v5).
// These shapes are the frozen contract between the WebSocket stream and the dashboard UI.

export interface Product {
  name: string;
  seller: string;
}

// ---------------------------------------------------------------------------
// Truth (matches backend ProductTruth — graph/state.py v11)
// ---------------------------------------------------------------------------

export type TruthCategory =
  | "color"
  | "material"
  | "texture"
  | "construction_detail"
  | "material_character"
  | "scale_cue"
  | "brief_or_intake_fact"
  | "form_factor";

// Frontend state shape — components read .id and .fact_text; handleEvent adapts from backend.
export interface Truth {
  id: string;        // maps to ProductTruth.truth_id
  category: TruthCategory;
  fact_text: string; // maps to ProductTruth.fact
}

// ---------------------------------------------------------------------------
// Scripts / Critic scores
// ---------------------------------------------------------------------------

export type ScoreKey = "hook" | "pacing" | "completion" | "cta" | "tone";

export type Scores = Record<ScoreKey, number>;

export interface Script {
  id: string;
  title: string;
  winner?: boolean;
  scores: Scores;
  reasoning: string;
  lines: string[];
  total: number;
}

// ---------------------------------------------------------------------------
// Merge validation (ScriptsPanel)
// ---------------------------------------------------------------------------

export interface MergeValidation {
  status: "pass" | "fail";
  repairPath: string;
  metaCriticSwapFired: boolean;
  note: string;
  seam: {
    location: string;
    before: string;
    after: string;
  };
}

// ---------------------------------------------------------------------------
// Treatment (TreatmentPanel)
// ---------------------------------------------------------------------------

export interface TreatmentBeat {
  id: string;            // maps to beat_index
  script_quote: string;
  truth_fact_id: string;
  why_not_generic: string;
}

export interface Treatment {
  director_persona: string;
  color_story: string;
  pacing_philosophy: string;
  beats: TreatmentBeat[];
}

// ---------------------------------------------------------------------------
// Budget (BudgetPanel)
// ---------------------------------------------------------------------------

export interface BudgetShot {
  id: string;
  label: string;
  alloc: number;
  justification: string;
}

export interface Budget {
  unit: string;
  cap: number;
  shots: BudgetShot[];
}

// ---------------------------------------------------------------------------
// Shots (ShotsPanel / ContinuityPanel)
// ---------------------------------------------------------------------------

// Matches backend Shot.status (graph/state.py v6 + v3 events.py "fallback_requested")
export type ShotStatus =
  | "pending"
  | "generating"
  | "passed"
  | "fallback"
  | "review"
  | "fallback_requested";

export interface Shot {
  id: string;      // maps to Shot.shot_id
  label: string;   // display label, e.g. "Shot 1"
  camera: string;
  move: string;
  duration: string;
  fallback: boolean;
  status?: ShotStatus;
  videoUri?: string;
}

// ---------------------------------------------------------------------------
// Interrupt / human review (ContinuityPanel)
// ---------------------------------------------------------------------------

export interface InterruptCandidate {
  id: string;
  note: string;
}

export interface Interrupt {
  shotId: string;
  label: string;
  driftScore: number;
  reason: string;
  candidates: InterruptCandidate[];
}

// Matches HumanReviewEntry.resolution (graph/state.py)
export type InterruptResolution = "approve" | "retry_with_edit" | "accept_fallback";

// ---------------------------------------------------------------------------
// Final deliverables (FinalPanel)
// ---------------------------------------------------------------------------

export interface FinalRatio {
  id: string;
  label: string;
  use: string;
  w: number;
  h: number;
  url?: string;
}

export interface Final {
  duration: string;
  masterCutUri?: string;
  ratios: FinalRatio[];
}

// ---------------------------------------------------------------------------
// C2 WebSocket event stream (graph/events.py v5)
//
// Only named business events are listed here. Transport lifecycle events
// (run.started / run.completed / run.error) are handled in the WS wrapper
// in studio/page.tsx and never reach handleEvent.
// ---------------------------------------------------------------------------

export type JobEvent =
  | {
      type: "node_started";
      payload: { node: string; label?: string; phase?: number };
    }
  | {
      type: "truth_extracted";
      payload: {
        truths: Array<{ truth_id: string; fact: string; category: TruthCategory; source: string }>;
        count: number;
      };
    }
  | {
      type: "critic_score";
      payload: {
        scores: Record<
          string,
          {
            hook: number;
            pacing: number;
            completion: number;
            cta: number;
            tone: number;
            composite: number;
            justification: string;
            never_do_violation: boolean;
          }
        >;
        winning_variant_ids?: string[];
      };
    }
  | {
      type: "treatment_ready";
      payload: {
        treatment: {
          director_persona: string;
          color_story: string;
          pacing_philosophy: string;
          beat_treatments: Array<{
            beat_index: number;
            beat_function: string;
            script_quote: string;
            truth_fact_id: string;
            visual_approach: string;
            why_not_generic: string;
          }>;
          character_anchor?: string;
        };
      };
    }
  | {
      type: "budget_updated";
      payload: {
        ledger: { cap: number; spent: number; per_shot: Record<string, number> };
        over_cap: boolean;
      };
    }
  | {
      type: "shot_generated";
      payload: {
        shot_id: string;
        generated?: { video_uri: string; drift_score?: number; attempt: number };
        status: ShotStatus;
        is_fallback: boolean;
      };
    }
  | {
      type: "drift_scored";
      payload: {
        shot_id: string;
        drift_score: number;
        threshold: number;
        passed: boolean;
        attempt: number;
      };
    }
  | {
      type: "interrupt_requested";
      payload: {
        review: { shot_id: string; drift_score: number; candidate_frame_uris: string[] };
        queue_position?: number;
      };
    }
  | {
      type: "edit_routed";
      payload: { edit_id: string; router_output: Record<string, unknown> };
    }
  | {
      type: "job_complete";
      payload: {
        master_cut_uri: string;
        exports: { aspect_9x16: string; aspect_1x1: string; aspect_16x9: string };
        voiceover?: { audio_uri: string; caption_track_uri: string };
      };
    }
  | {
      type: "merge_validated";
      payload: { result: Record<string, unknown>; attempt_number: number };
    }
  | {
      type: "vo_ready";
      payload: {
        voiceover: { audio_uri: string; caption_track_uri: string };
        caption_count: number;
        degraded: boolean;
      };
    }
  | {
      type: "master_cut_ready";
      payload: { uri: string; shot_count: number; total_duration_sec: number };
    };

// ---------------------------------------------------------------------------
// Job history (Library / "My Ads") — localStorage-backed
// ---------------------------------------------------------------------------

export interface HistoryEntry {
  productName: string;
  date: number;
  truths: Truth[];
  final: Final | null;
  jobId?: string;
}
