"use client";

import { useCallback, useEffect, useRef } from "react";
import type { ChangeEvent, DragEvent, KeyboardEvent } from "react";
import Header from "../components/Header";
import Wizard from "../components/wizard/Wizard";
import type { Photo } from "../components/wizard/types";
import Dashboard from "../components/dashboard/Dashboard";
import Library from "../components/library/Library";
import type {
  Budget,
  Final,
  HistoryEntry,
  Interrupt,
  InterruptResolution,
  JobEvent,
  MergeValidation,
  Script,
  Shot,
  Treatment,
  Truth,
} from "@/lib/types";
import { useMergeState } from "@/lib/useMergeState";
import "./studio.css";

const STEP_MS = 340;
const HISTORY_KEY = "pc-job-history";

// Map LangGraph node names → pipeline phase display strings
const NODE_TO_PHASE: Record<string, string> = {
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

const FINAL_RATIOS: Final["ratios"] = [
  { id: "9x16", label: "9:16", use: "TikTok / Reels", w: 9, h: 16 },
  { id: "1x1", label: "1:1", use: "Instagram Feed", w: 1, h: 1 },
  { id: "16x9", label: "16:9", use: "YouTube / Web", w: 16, h: 9 },
];

type Theme = "light" | "dark";
type Status = "wizard" | "dashboard" | "library";

interface State {
  theme: Theme;
  status: Status;
  step: 1 | 2 | 3 | 4;
  transitioning: boolean;

  photos: Photo[];
  hoveredPhoto: number | null;
  brief: string;
  moodWords: string[];
  moodInput: string;
  neverList: string[];
  neverInput: string;
  notes: string;
  dragOver: boolean;

  jobId: string | null;
  phase: string;
  phaseLabel: string;
  elapsed: number;
  jobDone: boolean;

  truths: Truth[];
  scripts: Script[];
  activeScriptId: string | null;
  winnerId: string | null;
  merge: MergeValidation | null;

  treatment: Treatment | null;
  budget: Budget & { running: number };
  shots: Shot[];
  drift: Record<string, number>;
  driftThreshold: number;

  interrupt: Interrupt | null;
  interruptResolution: InterruptResolution | null;
  final: Final | null;

  hoveredTruthId: string | null;
  budgetOpenId: string | null;
  shotOpenId: string | null;

  history: HistoryEntry[];
}

function initialState(): State {
  return {
    theme: "light",
    status: "wizard",
    step: 1,
    transitioning: false,

    photos: [],
    hoveredPhoto: null,
    brief: "",
    moodWords: [],
    moodInput: "",
    neverList: [],
    neverInput: "",
    notes: "",
    dragOver: false,

    jobId: null,
    phase: "",
    phaseLabel: "",
    elapsed: 0,
    jobDone: false,

    truths: [],
    scripts: [],
    activeScriptId: null,
    winnerId: null,
    merge: null,

    treatment: null,
    budget: { shots: [], running: 0, cap: 0, unit: "credits" },
    shots: [],
    drift: {},
    driftThreshold: 0.25,

    interrupt: null,
    interruptResolution: null,
    final: null,

    hoveredTruthId: null,
    budgetOpenId: null,
    shotOpenId: null,

    history: [],
  };
}

// Derive API base URL from the WS base URL env var
function getApiBase(): string {
  const wsBase = process.env.NEXT_PUBLIC_WS_BASE_URL ?? "ws://localhost:8000";
  return wsBase.replace(/^wss:\/\//, "https://").replace(/^ws:\/\//, "http://");
}
function getWsBase(): string {
  return process.env.NEXT_PUBLIC_WS_BASE_URL ?? "ws://localhost:8000";
}

export default function StudioPage() {
  const [state, setState] = useMergeState<State>(initialState);

  const fileRef = useRef<HTMLInputElement | null>(null);
  const timersRef = useRef<Array<ReturnType<typeof setTimeout>>>([]);
  const jobRef = useRef<WebSocket | null>(null);
  const jobIdRef = useRef<string | null>(null);   // mirror of state.jobId for callbacks
  const startedAtRef = useRef(0);
  const elapsedIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const pushTimer = (t: ReturnType<typeof setTimeout>) => {
    timersRef.current.push(t);
  };
  const clearTimers = () => {
    timersRef.current.forEach((t) => {
      clearTimeout(t);
      clearInterval(t);
    });
    timersRef.current = [];
    if (elapsedIntervalRef.current) {
      clearInterval(elapsedIntervalRef.current);
      elapsedIntervalRef.current = null;
    }
  };

  useEffect(() => {
    // Load theme preference
    try {
      const saved = localStorage.getItem("pc-theme");
      if (saved === "light" || saved === "dark") setState({ theme: saved });
    } catch { /* ignore */ }

    // Load history
    try {
      const history: HistoryEntry[] = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
      setState({ history });
    } catch { /* ignore */ }

    // Check ?view=library URL param
    try {
      const params = new URLSearchParams(window.location.search);
      if (params.get("view") === "library") setState({ status: "library" });
    } catch { /* ignore */ }

    return () => {
      clearTimers();
      jobRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-reconnect: if HMR reloads the component while we're mid-job (state preserved
  // but jobRef.current is null), reopen the WS so run.completed fires the recovery fetch.
  useEffect(() => {
    if (state.status === "dashboard" && state.jobId && !state.jobDone && !jobRef.current) {
      openWebSocket(state.jobId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.status, state.jobId, state.jobDone]);

  const toggleTheme = useCallback(() => {
    setState((s) => {
      const theme: Theme = s.theme === "dark" ? "light" : "dark";
      try { localStorage.setItem("pc-theme", theme); } catch { /* ignore */ }
      return { theme };
    });
  }, [setState]);

  // ---- wizard nav ----
  const goStep = useCallback(
    (n: 1 | 2 | 3 | 4) => {
      if (state.transitioning || n === state.step || n < 1 || n > 4) return;
      setState({ transitioning: true });
      const t = setTimeout(() => setState({ transitioning: false, step: n }), STEP_MS);
      pushTimer(t);
    },
    [state.transitioning, state.step, setState],
  );
  const goNext = useCallback(() => {
    if (state.step === 1 && state.photos.length < 1) return;
    if (state.step === 2 && !state.brief.trim()) return;
    goStep(Math.min(state.step + 1, 4) as 1 | 2 | 3 | 4);
  }, [state.step, state.photos.length, state.brief, goStep]);
  const goBack = useCallback(() => {
    goStep(Math.max(state.step - 1, 1) as 1 | 2 | 3 | 4);
  }, [state.step, goStep]);

  // ---- files ----
  const onPickClick = useCallback(() => { fileRef.current?.click(); }, []);

  const addFiles = useCallback(
    (fileList: FileList | null) => {
      if (!fileList) return;
      // Snapshot into a plain array now: fileList (e.g. e.target.files) is a live
      // reference, and callers reset input.value right after calling this, which
      // clears that same live FileList before the setState updater below runs.
      const incoming = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
      if (!incoming.length) return;
      setState((s) => {
        const room = 3 - s.photos.length;
        if (room <= 0) return {};
        const mapped: Photo[] = incoming
          .slice(0, room)
          .map((f) => ({ name: f.name, url: URL.createObjectURL(f), file: f }));
        return { photos: [...s.photos, ...mapped].slice(0, 3) };
      });
    },
    [setState],
  );

  const onFileChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => { addFiles(e.target.files); e.target.value = ""; },
    [addFiles],
  );
  const onDragOver = useCallback(
    (e: DragEvent<HTMLDivElement>) => { e.preventDefault(); setState((s) => (s.dragOver ? {} : { dragOver: true })); },
    [setState],
  );
  const onDragLeave = useCallback(
    (e: DragEvent<HTMLDivElement>) => { e.preventDefault(); setState({ dragOver: false }); },
    [setState],
  );
  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => { e.preventDefault(); setState({ dragOver: false }); addFiles(e.dataTransfer.files); },
    [addFiles, setState],
  );
  const removePhoto = useCallback(
    (i: number) => {
      setState((s) => ({ photos: s.photos.filter((_, idx) => idx !== i), hoveredPhoto: null }));
    },
    [setState],
  );

  // ---- text / tags ----
  const onBriefInput = useCallback((e: ChangeEvent<HTMLInputElement>) => setState({ brief: e.target.value }), [setState]);
  const onBriefKey = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter" && state.brief.trim()) { e.preventDefault(); goStep(3); }
    },
    [state.brief, goStep],
  );
  const onNotesInput = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => setState({ notes: e.target.value }),
    [setState],
  );
  const onMoodInput = useCallback((value: string) => setState({ moodInput: value }), [setState]);
  const onNeverInput = useCallback((value: string) => setState({ neverInput: value }), [setState]);

  const addTag = useCallback(
    (listKey: "moodWords" | "neverList", inputKey: "moodInput" | "neverInput") => {
      setState((s) => {
        const val = s[inputKey].trim();
        if (!val) return {};
        if (s[listKey].includes(val)) return { [inputKey]: "" } as Partial<State>;
        return { [listKey]: [...s[listKey], val], [inputKey]: "" } as Partial<State>;
      });
    },
    [setState],
  );
  const onMoodKey = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => { if (e.key === "Enter") { e.preventDefault(); addTag("moodWords", "moodInput"); } },
    [addTag],
  );
  const onNeverKey = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => { if (e.key === "Enter") { e.preventDefault(); addTag("neverList", "neverInput"); } },
    [addTag],
  );
  const removeTag = useCallback(
    (listKey: "moodWords" | "neverList", i: number) => {
      setState((s) => ({ [listKey]: s[listKey].filter((_, idx) => idx !== i) }) as Partial<State>);
    },
    [setState],
  );

  // ---- C2 event handler ----
  // Adapts real backend C2 payloads into the State shape that Dashboard components render.
  const handleEvent = useCallback(
    (e: JobEvent) => {
      const { type, payload } = e;
      switch (type) {
        case "node_started": {
          const phase = NODE_TO_PHASE[payload.node] ?? payload.node;
          const phaseLabel = payload.label ?? phase;
          setState({ phase, phaseLabel });
          break;
        }

        case "truth_extracted": {
          // C2: {truths: [{truth_id, fact, category, source}], count}
          const truths: Truth[] = payload.truths.map((t) => ({
            id: t.truth_id,
            category: t.category,
            fact_text: t.fact,
          }));
          setState((s) => ({ truths: [...s.truths, ...truths] }));
          break;
        }

        case "critic_score": {
          // C2: {scores: Record<variant_id, CriticScore>, winning_variant_ids?}
          const newScripts: Script[] = Object.entries(payload.scores).map(([id, score], i) => ({
            id,
            title: `Variant ${String(i + 1).padStart(2, "0")}`,
            winner: payload.winning_variant_ids?.includes(id),
            scores: {
              hook: Math.round(score.hook * 100),
              pacing: Math.round(score.pacing * 100),
              completion: Math.round(score.completion * 100),
              cta: Math.round(score.cta * 100),
              tone: Math.round(score.tone * 100),
            },
            reasoning: score.justification,
            lines: [],
            total: Math.round(score.composite * 100),
          }));
          const firstWinner = payload.winning_variant_ids?.[0] ?? null;
          setState((s) => ({
            scripts: [
              ...s.scripts,
              ...newScripts.filter((ns) => !s.scripts.some((x) => x.id === ns.id)),
            ],
            winnerId: firstWinner ?? s.winnerId,
            activeScriptId: firstWinner ?? s.activeScriptId,
          }));
          break;
        }

        case "merge_validated": {
          // C2: {result: dict, attempt_number: number}
          const result = payload.result as Record<string, unknown>;
          const passed = result["passed"] === true || result["status"] === "pass";
          setState({
            merge: {
              status: passed ? "pass" : "fail",
              repairPath: String(result["repair_path"] ?? ""),
              metaCriticSwapFired: Boolean(result["meta_critic_swap_fired"]),
              note: String(result["note"] ?? result["justification"] ?? ""),
              seam: {
                location: String((result["seam"] as Record<string, unknown>)?.["location"] ?? ""),
                before: String((result["seam"] as Record<string, unknown>)?.["before"] ?? ""),
                after: String((result["seam"] as Record<string, unknown>)?.["after"] ?? ""),
              },
            },
          });
          break;
        }

        case "treatment_ready": {
          // C2: {treatment: {director_persona, color_story, pacing_philosophy, beat_treatments}}
          const t = payload.treatment;
          setState({
            treatment: {
              director_persona: t.director_persona,
              color_story: t.color_story,
              pacing_philosophy: t.pacing_philosophy,
              beats: t.beat_treatments.map((b) => ({
                id: String(b.beat_index),
                script_quote: b.script_quote,
                truth_fact_id: b.truth_fact_id,
                why_not_generic: b.why_not_generic,
              })),
            },
          });
          break;
        }

        case "budget_updated": {
          // C2: {ledger: {cap, spent, per_shot: Record<shot_id, amount>}, over_cap}
          const ledger = payload.ledger;
          setState((s) => {
            const budgetShots = Object.entries(ledger.per_shot).map(([id, alloc], i) => ({
              id,
              label: s.shots.find((sh) => sh.id === id)?.label ?? `Shot ${i + 1}`,
              alloc,
              justification: "",
            }));
            return {
              budget: {
                cap: ledger.cap,
                unit: s.budget.unit || "credits",
                running: ledger.spent,
                shots: budgetShots,
              },
            };
          });
          break;
        }

        case "shot_generated": {
          // C2: {shot_id, generated?, status, is_fallback}
          const { shot_id, status, is_fallback } = payload;
          const videoUri = payload.generated?.video_uri;
          setState((s) => {
            const exists = s.shots.find((sh) => sh.id === shot_id);
            if (exists) {
              return {
                shots: s.shots.map((sh) =>
                  sh.id === shot_id
                    ? { ...sh, status, fallback: is_fallback, ...(videoUri ? { videoUri } : {}) }
                    : sh
                ),
              };
            }
            const newShot: Shot = {
              id: shot_id,
              label: `Shot ${s.shots.length + 1}`,
              camera: "",
              move: "",
              duration: "",
              fallback: is_fallback,
              status,
              ...(videoUri ? { videoUri } : {}),
            };
            return { shots: [...s.shots, newShot] };
          });
          break;
        }

        case "drift_scored": {
          // C2: {shot_id, drift_score, threshold, passed, attempt}
          setState((s) => ({
            drift: { ...s.drift, [payload.shot_id]: payload.drift_score },
            driftThreshold: payload.threshold,
          }));
          break;
        }

        case "interrupt_requested": {
          // C2: {review: {shot_id, drift_score, candidate_frame_uris}, queue_position?}
          const review = payload.review;
          const interrupt: Interrupt = {
            shotId: review.shot_id,
            label: `Shot ${review.shot_id.slice(-6)}`,
            driftScore: review.drift_score,
            reason: "Continuity drift exceeded retry cap. Please review the generated candidates.",
            candidates: review.candidate_frame_uris.length
              ? review.candidate_frame_uris.map((uri, i) => ({ id: `c${i}`, note: uri }))
              : [{ id: "c0", note: "No candidate frames available" }],
          };
          setState({ interrupt });
          break;
        }

        case "job_complete": {
          // C2: {master_cut_uri, exports, voiceover?}
          if (elapsedIntervalRef.current) {
            clearInterval(elapsedIntervalRef.current);
            elapsedIntervalRef.current = null;
          }
          const { master_cut_uri, exports: ex } = payload;
          const ratiosWithUrls: Final["ratios"] = [
            { ...FINAL_RATIOS[0], url: ex?.aspect_9x16 },
            { ...FINAL_RATIOS[1], url: ex?.aspect_1x1 },
            { ...FINAL_RATIOS[2], url: ex?.aspect_16x9 },
          ];
          const final: Final = { duration: "18s", masterCutUri: master_cut_uri, ratios: ratiosWithUrls };
          setState((s) => {
            const entry: HistoryEntry = {
              productName: s.brief || "Product Ad",
              date: Date.now(),
              truths: s.truths,
              final,
            };
            const history = [entry, ...s.history].slice(0, 20);
            try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); } catch { /* ignore */ }
            return { final, jobDone: true, phase: "Delivery", history };
          });
          break;
        }

        // Informational / future panels — no state update needed yet
        case "vo_ready":
        case "master_cut_ready":
        case "edit_routed":
          break;

        default:
          break;
      }
    },
    [setState],
  );

  // ---- WebSocket connection ----
  const openWebSocket = useCallback(
    (jobId: string, resolution?: string) => {
      // Close existing connection
      if (jobRef.current) {
        jobRef.current.onmessage = null;
        jobRef.current.close();
        jobRef.current = null;
      }

      const wsBase = getWsBase();
      const url = resolution
        ? `${wsBase}/ws/${jobId}?resolution=${encodeURIComponent(resolution)}`
        : `${wsBase}/ws/${jobId}`;

      const ws = new WebSocket(url);
      jobRef.current = ws;

      ws.onmessage = (event: MessageEvent) => {
        try {
          const msg: { type: string; payload: unknown } = JSON.parse(event.data as string);
          if (msg.type === "run.started") return;
          if (msg.type === "run.completed") {
            // If the pipeline finished before we connected (e.g. page reload / WS reconnect),
            // the graph emits run.completed immediately with no business events. Fall back to
            // the REST state endpoint to reconstruct the delivery section.
            setState((s) => {
              if (s.jobDone) return {};  // already have delivery data
              return {};
            });
            const apiBase = getApiBase();
            fetch(`${apiBase}/jobs/${jobId}/state`)
              .then((r) => r.json())
              .then((data: { state?: Record<string, unknown>; next?: string[] }) => {
                const st = data.state;
                if (!st) return;
                const masterCutUri = st["master_cut_uri"] as string | undefined;
                const ex = st["exports"] as { aspect_9x16?: string; aspect_1x1?: string; aspect_16x9?: string } | undefined;
                if (masterCutUri && ex) {
                  handleEvent({
                    type: "job_complete",
                    payload: {
                      master_cut_uri: masterCutUri,
                      exports: {
                        aspect_9x16: ex.aspect_9x16 ?? "",
                        aspect_1x1: ex.aspect_1x1 ?? "",
                        aspect_16x9: ex.aspect_16x9 ?? "",
                      },
                    },
                  });
                }
              })
              .catch(() => { /* best-effort */ });
            return;
          }
          if (msg.type === "run.error") {
            console.error("[ProductCut] graph error:", (msg.payload as Record<string, unknown>)?.error);
            return;
          }
          handleEvent(msg as JobEvent);
        } catch (err) {
          console.error("[ProductCut] WS parse error:", err);
        }
      };

      ws.onerror = (err) => console.error("[ProductCut] WebSocket error:", err);
      ws.onclose = () => {
        console.log("[ProductCut] WebSocket closed for job:", jobId);
        jobRef.current = null;
        // If the job isn't done, re-open automatically to resume the graph run.
        setState((s) => {
          if (!s.jobDone) {
            // Delay slightly to avoid hammering the server on rapid close/open cycles.
            setTimeout(() => openWebSocket(jobId), 2000);
          }
          return {};
        });
      };
    },
    [handleEvent],
  );

  // ---- generate -> live dashboard ----
  const onGenerate = useCallback(async () => {
    if (state.transitioning) return;

    const apiBase = getApiBase();

    // Build multipart form
    const formData = new FormData();
    formData.append("brief", state.brief);
    if (state.moodWords.length) formData.append("mood_words", JSON.stringify(state.moodWords));
    if (state.neverList.length) formData.append("never_do", state.neverList.join(", "));
    if (state.notes) formData.append("notes", state.notes);
    state.photos.forEach((photo) => {
      if (photo.file) formData.append("photos", photo.file, photo.name);
    });

    let jobId: string;
    try {
      const res = await fetch(`${apiBase}/jobs`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`POST /jobs returned ${res.status}`);
      const data = (await res.json()) as { job_id: string };
      jobId = data.job_id;
    } catch (err) {
      console.error("[ProductCut] Failed to create job:", err);
      return;
    }

    jobIdRef.current = jobId;
    setState({ transitioning: true, jobId });
    const t = setTimeout(() => {
      setState({ transitioning: false, status: "dashboard", elapsed: 0, jobDone: false });
      startedAtRef.current = Date.now();
      const el = setInterval(() => setState({ elapsed: Date.now() - startedAtRef.current }), 500);
      elapsedIntervalRef.current = el;
      pushTimer(el);
      openWebSocket(jobId);
    }, STEP_MS);
    pushTimer(t);
  }, [state, setState, openWebSocket]);

  const resolveInterrupt = useCallback(
    (resolution: InterruptResolution) => {
      const jobId = jobIdRef.current;
      if (!jobId) return;
      setState({ interrupt: null, interruptResolution: resolution });
      openWebSocket(jobId, resolution);
    },
    [openWebSocket, setState],
  );

  const resetPipeline = useCallback(() => {
    clearTimers();
    jobRef.current?.close();
    jobRef.current = null;
    jobIdRef.current = null;
    setState((s) => ({ ...initialState(), theme: s.theme, history: s.history }));
  }, [setState]);

  // ---- library ----
  const openLibrary = useCallback(() => setState({ status: "library" }), [setState]);
  const closeLibrary = useCallback(() => {
    setState((s) => ({ status: s.truths.length ? "dashboard" : "wizard" }));
  }, [setState]);
  const openHistoryItem = useCallback(
    (entry: HistoryEntry) => {
      setState({
        status: "dashboard",
        jobDone: true,
        phase: "Delivery",
        phaseLabel: "",
        truths: entry.truths,
        scripts: [],
        activeScriptId: null,
        winnerId: null,
        merge: null,
        treatment: null,
        budget: { shots: [], running: 0, cap: 0, unit: "credits" },
        shots: [],
        drift: {},
        interrupt: null,
        interruptResolution: null,
        final: entry.final,
      });
    },
    [setState],
  );

  // ---- script variant tablist: roving-tabindex + arrow-key nav ----
  const handleScriptTabKey = useCallback(
    (e: KeyboardEvent<HTMLButtonElement>, idx: number) => {
      const scripts = state.scripts;
      if (!scripts.length) return;
      let next: number | null = null;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (idx + 1) % scripts.length;
      else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = (idx - 1 + scripts.length) % scripts.length;
      else if (e.key === "Home") next = 0;
      else if (e.key === "End") next = scripts.length - 1;
      if (next === null) return;
      e.preventDefault();
      const nextId = scripts[next].id;
      setState({ activeScriptId: nextId });
      requestAnimationFrame(() => {
        document.getElementById(`script-tab-${nextId}`)?.focus();
      });
    },
    [state.scripts, setState],
  );

  return (
    <div
      className="pc-studio"
      data-theme={state.theme}
      style={{
        minHeight: "100vh",
        background: "var(--paper)",
        color: "var(--ink)",
        fontFamily: "var(--font-sans)",
        WebkitFontSmoothing: "antialiased",
      }}
    >
      <Header theme={state.theme} onToggleTheme={toggleTheme} />

      {state.status === "wizard" && (
        <Wizard
          step={state.step}
          transitioning={state.transitioning}
          photos={state.photos}
          hoveredPhoto={state.hoveredPhoto}
          dragOver={state.dragOver}
          fileRef={fileRef}
          onPickClick={onPickClick}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onFileChange={onFileChange}
          onPhotoEnter={(i) => setState({ hoveredPhoto: i })}
          onPhotoLeave={() => setState({ hoveredPhoto: null })}
          onRemovePhoto={removePhoto}
          brief={state.brief}
          onBriefInput={onBriefInput}
          onBriefKey={onBriefKey}
          moodWords={state.moodWords}
          moodInput={state.moodInput}
          onMoodInput={onMoodInput}
          onMoodKey={onMoodKey}
          onRemoveMood={(i) => removeTag("moodWords", i)}
          neverList={state.neverList}
          neverInput={state.neverInput}
          onNeverInput={onNeverInput}
          onNeverKey={onNeverKey}
          onRemoveNever={(i) => removeTag("neverList", i)}
          notes={state.notes}
          onNotesInput={onNotesInput}
          goNext={goNext}
          goBack={goBack}
          goStep={goStep}
          onGenerate={onGenerate}
        />
      )}

      {state.status === "dashboard" && (
        <Dashboard
          phase={state.phase}
          phaseLabel={state.phaseLabel}
          elapsed={state.elapsed}
          jobDone={state.jobDone}
          onResetPipeline={resetPipeline}
          historyCount={state.history.length}
          onOpenLibrary={openLibrary}
          truths={state.truths}
          hoveredTruthId={state.hoveredTruthId}
          onHoverTruth={(id) => setState({ hoveredTruthId: id })}
          scripts={state.scripts}
          activeScriptId={state.activeScriptId}
          winnerId={state.winnerId}
          merge={state.merge}
          onSelectScript={(id) => setState({ activeScriptId: id })}
          onScriptTabKey={handleScriptTabKey}
          treatment={state.treatment}
          budget={state.budget}
          budgetOpenId={state.budgetOpenId}
          onToggleBudgetRow={(id) => setState((s) => ({ budgetOpenId: s.budgetOpenId === id ? null : id }))}
          shots={state.shots}
          shotOpenId={state.shotOpenId}
          onToggleShot={(id) => setState((s) => ({ shotOpenId: s.shotOpenId === id ? null : id }))}
          drift={state.drift}
          driftThreshold={state.driftThreshold}
          interrupt={state.interrupt}
          interruptResolution={state.interruptResolution}
          onApprove={() => resolveInterrupt("approve")}
          onRetry={() => resolveInterrupt("retry_with_edit")}
          onFallback={() => resolveInterrupt("accept_fallback")}
          final={state.final}
        />
      )}

      {state.status === "library" && (
        <Library history={state.history} onClose={closeLibrary} onOpen={openHistoryItem} />
      )}
    </div>
  );
}
