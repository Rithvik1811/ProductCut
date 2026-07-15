// ProductCut — shared types for the mock job pipeline.
// These shapes are the contract with the real backend C1 job-state schema.

export interface Product {
  name: string;
  seller: string;
}

export type TruthCategory =
  | "material"
  | "color"
  | "texture"
  | "distinguishing_mark"
  | "size"
  | "condition"
  | "shape";

export interface Truth {
  id: string;
  category: TruthCategory;
  fact_text: string;
}

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

export interface TreatmentBeat {
  id: string;
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

export type ShotStatus = "queued" | "generating" | "retrying" | "done" | "fallback";

export interface Shot {
  id: string;
  label: string;
  camera: string;
  move: string;
  duration: string;
  fallback: boolean;
  status?: ShotStatus;
}

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

export interface FinalRatio {
  id: string;
  label: string;
  use: string;
  w: number;
  h: number;
}

export interface Final {
  duration: string;
  ratios: FinalRatio[];
}

export type InterruptResolution = "approve" | "retry" | "fallback";

// ---- streamed job events -----------------------------------------------------

export type JobEvent =
  | { type: "node_started"; payload: { phase: string; label: string } }
  | { type: "truth_extracted"; payload: { truth: Truth } }
  | { type: "critic_score"; payload: { script: Script; index: number; total: number } }
  | { type: "critic_done"; payload: { winnerId: string; merge: MergeValidation } }
  | { type: "treatment_ready"; payload: { treatment: Treatment } }
  | {
      type: "budget_updated";
      payload: { shot: BudgetShot; running: number; cap: number; unit: string; index: number };
    }
  | { type: "shots_init"; payload: { shots: Shot[] } }
  | { type: "shot_generated"; payload: { id: string; status: ShotStatus; attempt?: number } }
  | { type: "drift_scored"; payload: { shotId: string; score: number } }
  | { type: "interrupt_requested"; payload: { interrupt: Interrupt } }
  | {
      type: "interrupt_resolved";
      payload: { shotId: string; resolution: InterruptResolution; status: ShotStatus };
    }
  | { type: "job_complete"; payload: { final: Final; product: Product } };

export interface MockJob {
  on(handler: (e: JobEvent) => void): () => void;
  start(): void;
  resume(resolution: InterruptResolution): void;
  stop(): void;
  readonly paused: boolean;
}

// ---- job history (Library / "My Ads") ----------------------------------------

export interface HistoryEntry {
  productName: string;
  date: number;
  truths: Truth[];
  final: Final | null;
}
