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
import { PHASES, NODE_TO_PHASE } from "@/lib/phases";
import "./studio.css";

const STEP_MS = 340;
const HISTORY_KEY = "pc-job-history";

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
  // Highest pipeline stage index (into PHASES) any real C2 event has proven
  // reached; -1 = nothing received yet. Never guessed ahead of real events —
  // see handleEvent's phase bumps below.
  maxPhaseIdx: number;
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

  error: string | null;
  lastResolvedShotId: string | null;

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
    maxPhaseIdx: -1,
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

    error: null,
    lastResolvedShotId: null,

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
  const jobIdRef = useRef<string | null>(null);       // always synced to state.jobId via effect
  const noReconnectRef = useRef(false);               // bug 2: suppress onclose reconnect when intentionally closed
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

    // Bug 11: auto-reconnect + seed Library history from DB on startup
    fetch(`${getApiBase()}/jobs`)
      .then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then((jobs: Array<{ job_id: string; status: string; brief?: string; created_at: string }>) => {
        // Merge DB jobs into history so badge count and Library are correct immediately
        setState((s) => {
          const localById = new Map(s.history.map((h) => [h.jobId, h]));
          const merged: HistoryEntry[] = jobs.map((j) => {
            const local = localById.get(j.job_id);
            if (local) return local;
            return {
              productName: j.brief ? j.brief.slice(0, 60) : j.job_id.slice(0, 8),
              date: new Date(j.created_at).getTime(),
              truths: [],
              final: null,
              jobId: j.job_id,
            };
          });
          return { history: merged };
        });
        if (jobIdRef.current) return;  // onGenerate already fired this session
        // N6: only reconnect to "running" — "ingested" means the WS was never opened
        const running = jobs.find((j) => j.status === "running");
        if (running) {
          setState({ status: "dashboard", jobId: running.job_id, jobDone: false });
        }
      })
      .catch(() => { /* best-effort */ });

    return () => {
      clearTimers();
      // Bug 2: null handlers before close so unmount doesn't schedule a ghost reconnect
      if (jobRef.current) {
        jobRef.current.onclose = null;
        jobRef.current.onerror = null;
        jobRef.current.onmessage = null;
        jobRef.current.close();
        jobRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keep jobIdRef in sync with state.jobId so resolveInterrupt always has the current value
  // regardless of whether the job came from onGenerate or a mount-time reconnect (bug 14).
  useEffect(() => {
    jobIdRef.current = state.jobId;
  }, [state.jobId]);

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
  // Monotonic max: only ever advances, and only in response to a real event —
  // never lets a later-arriving event regress an earlier, already-proven stage.
  const bumpPhase = useCallback(
    (idx: number) => setState((s) => (idx > s.maxPhaseIdx ? { maxPhaseIdx: idx } : {})),
    [setState],
  );

  const handleEvent = useCallback(
    (e: JobEvent) => {
      const { type, payload } = e;
      switch (type) {
        case "node_started": {
          // Not currently emitted by the real backend (see lib/phases.ts), but
          // respected if it ever is, via the same real-event-only cascade.
          const phase = NODE_TO_PHASE[payload.node];
          if (phase) bumpPhase(PHASES.indexOf(phase));
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
          bumpPhase(PHASES.indexOf("Truths"));
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
          bumpPhase(PHASES.indexOf("Scripts"));
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
          bumpPhase(PHASES.indexOf("Scripts"));
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
          bumpPhase(PHASES.indexOf("Treatment"));
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
          bumpPhase(PHASES.indexOf("Budget"));
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
          bumpPhase(PHASES.indexOf("Shots"));
          break;
        }

        case "drift_scored": {
          // C2: {shot_id, drift_score, threshold, passed, attempt}
          setState((s) => ({
            drift: { ...s.drift, [payload.shot_id]: payload.drift_score },
            driftThreshold: payload.threshold,
            // N1: clear dedup key once the shot has new activity — a genuine second
            // interrupt for this shot after re-generation must not be suppressed.
            lastResolvedShotId: s.lastResolvedShotId === payload.shot_id ? null : s.lastResolvedShotId,
          }));
          bumpPhase(PHASES.indexOf("Continuity"));
          break;
        }

        case "interrupt_requested": {
          // C2: {review: {shot_id, drift_score, candidate_frame_uris}, queue_position?}
          const review = payload.review;
          setState((s) => {
            // Bug 3: after resolveInterrupt the pipeline resumes and may re-emit this event
            // for the same shot before the next node runs — suppress it.
            if (s.lastResolvedShotId === review.shot_id) return {};
            const interrupt: Interrupt = {
              shotId: review.shot_id,
              label: `Shot ${review.shot_id.slice(-6)}`,
              driftScore: review.drift_score,
              reason: "Continuity drift exceeded retry cap. Please review the generated candidates.",
              candidates: review.candidate_frame_uris.length
                ? review.candidate_frame_uris.map((uri, i) => ({ id: `c${i}`, note: uri }))
                : [{ id: "c0", note: "No candidate frames available" }],
            };
            return { interrupt };
          });
          bumpPhase(PHASES.indexOf("Continuity"));
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
          const final: Final = { duration: "30s", masterCutUri: master_cut_uri, ratios: ratiosWithUrls };
          setState((s) => {
            const entry: HistoryEntry = {
              productName: s.brief || "Product Ad",
              date: Date.now(),
              truths: s.truths,
              final,
              jobId: s.jobId ?? undefined,
            };
            const history = [entry, ...s.history.filter(h => h.jobId !== entry.jobId)].slice(0, 20);
            try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); } catch { /* ignore */ }
            return { final, jobDone: true, history };
          });
          bumpPhase(PHASES.indexOf("Delivery"));
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
    [setState, bumpPhase],
  );

  // ---- WebSocket connection ----
  const openWebSocket = useCallback(
    (jobId: string, resolution?: string) => {
      // Bug 2: null ALL handlers before intentional close so the ghost onclose
      // does not fire and schedule an unwanted reconnect with the old jobId.
      if (jobRef.current) {
        jobRef.current.onclose = null;
        jobRef.current.onerror = null;
        jobRef.current.onmessage = null;
        jobRef.current.close();
        jobRef.current = null;
      }
      // Bug 2: a fresh open means any previous noReconnect flag is now stale.
      noReconnectRef.current = false;

      const apiBase = getApiBase();

      // Bug 10: rehydrate panels from checkpoint on every WS open (handles reconnect/resume).
      // Fires concurrently with WS connect; any live events that arrive will overwrite stale data.
      fetch(`${apiBase}/jobs/${jobId}/state`)
        .then((r) => r.ok ? r.json() : null)
        .then((data: { state?: Record<string, unknown>; next?: string[] } | null) => {
          if (!data?.state) return;
          const st = data.state;
          // If the job already completed (e.g. fast run or page-reload), synthesise job_complete.
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
            return;
          }
          // Rehydrate truths so the panel isn't blank after reconnect.
          const rawTruths = st["truths"] as Array<{ truth_id: string; fact: string; category: string }> | undefined;
          if (rawTruths?.length) {
            setState((s) => {
              if (s.truths.length) return {};
              const truths: Truth[] = rawTruths.map((t) => ({
                id: t.truth_id,
                category: t.category as Truth["category"],
                fact_text: t.fact,
              }));
              return { truths };
            });
          }
        })
        .catch(() => { /* best-effort */ });

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
            noReconnectRef.current = true;
            // Fetch final state in case the pipeline finished before we connected.
            fetch(`${apiBase}/jobs/${jobId}/state`)
              .then((r) => r.ok ? r.json() : null)
              .then((data: { state?: Record<string, unknown>; next?: string[] } | null) => {
                const st = data?.state;
                const masterCutUri = st?.["master_cut_uri"] as string | undefined;
                const ex = st?.["exports"] as { aspect_9x16?: string; aspect_1x1?: string; aspect_16x9?: string } | undefined;
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
                } else {
                  // N3: graph completed but no exports in checkpoint (format_export_node
                  // skipped, or aget_state threw and we fell back). Stop the spinner so
                  // the dashboard doesn't spin forever with noReconnect already true.
                  setState((s) => s.jobDone ? {} : { jobDone: true });
                }
              })
              .catch(() => {
                // N3: fetch failed entirely — still stop the spinner.
                setState((s) => s.jobDone ? {} : { jobDone: true });
              });
            return;
          }

          if (msg.type === "run.error") {
            // Bug 5: surface error in UI and stop reconnecting
            noReconnectRef.current = true;
            const errMsg = String((msg.payload as Record<string, unknown>)?.error ?? "Pipeline failed");
            setState({ error: errMsg, jobDone: true });
            return;
          }

          if (msg.type === "run.interrupted") {
            // Bug 9: pipeline is paused at interrupt() — wait for user to resolve before reconnecting.
            // The interrupt_requested C2 event (handled below) already populated state.interrupt.
            noReconnectRef.current = true;
            return;
          }

          if (msg.type === "run.busy") {
            // N4: don't permanently block — the existing handler may disconnect within
            // seconds. Schedule a retry so the client catches up without user action.
            // (noReconnectRef stays false so onclose's 2s retry still fires naturally,
            // but add an extra 5s delay here in case the close hasn't happened yet.)
            const t = setTimeout(() => openWebSocket(jobId), 5000);
            timersRef.current.push(t);
            return;
          }

          handleEvent(msg as JobEvent);
        } catch (err) {
          console.error("[ProductCut] WS parse error:", err);
        }
      };

      ws.onerror = (err) => console.error("[ProductCut] WebSocket error:", err);

      ws.onclose = () => {
        // Bug 2: ignore close events from a WS we already replaced.
        if (jobRef.current !== ws) return;
        jobRef.current = null;
        // Don't reconnect if we intentionally stopped (error, complete, interrupted, busy).
        if (noReconnectRef.current) return;
        // Bug 2: schedule outside setState (no side effects in updaters); register timer so
        // clearTimers() can cancel it on reset/unmount.
        const t = setTimeout(() => openWebSocket(jobId), 2000);
        timersRef.current.push(t);
      };
    },
    [handleEvent, setState],
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
      // Bug 13: show error in UI instead of silent console.error
      const msg = err instanceof Error ? err.message : "Failed to start job";
      setState({ error: msg });
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
      // Bug 14: jobIdRef is kept in sync with state.jobId via useEffect, so it's correct
      // even when the job was set by mount-time reconnect rather than onGenerate.
      const jobId = jobIdRef.current;
      if (!jobId) return;
      // Bug 3: record which shot was resolved so interrupt_requested for the same shot
      // after resume is suppressed (backend may re-emit it before the next node runs).
      setState((s) => ({
        interrupt: null,
        interruptResolution: resolution,
        lastResolvedShotId: s.interrupt?.shotId ?? null,
      }));
      openWebSocket(jobId, resolution);
    },
    [openWebSocket, setState],
  );

  const resetPipeline = useCallback(() => {
    clearTimers();
    // Bug 2: null handlers before close so ghost onclose doesn't schedule a reconnect.
    if (jobRef.current) {
      jobRef.current.onclose = null;
      jobRef.current.onerror = null;
      jobRef.current.onmessage = null;
      jobRef.current.close();
      jobRef.current = null;
    }
    noReconnectRef.current = false;
    jobIdRef.current = null;
    setState((s) => ({ ...initialState(), theme: s.theme, history: s.history }));
  }, [setState]);

  // ---- library ----
  const openLibrary = useCallback(() => {
    setState({ status: "library" });
    // Fetch all jobs from DB and merge into history (DB is source of truth)
    fetch(`${getApiBase()}/jobs`)
      .then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then((jobs: Array<{ job_id: string; brief?: string; status: string; created_at: string }>) => {
        setState((s) => {
          const localById = new Map(s.history.map((h) => [h.jobId, h]));
          const merged: HistoryEntry[] = jobs.map((j) => {
            const local = localById.get(j.job_id);
            if (local) return local;
            return {
              productName: j.brief ? j.brief.slice(0, 60) : j.job_id.slice(0, 8),
              date: new Date(j.created_at).getTime(),
              truths: [],
              final: null,
              jobId: j.job_id,
            };
          });
          return { history: merged };
        });
      })
      .catch(() => { /* best-effort — show localStorage history */ });
  }, [setState]);
  const closeLibrary = useCallback(() => {
    setState((s) => ({ status: s.truths.length ? "dashboard" : "wizard" }));
  }, [setState]);
  const openHistoryItem = useCallback(
    async (entry: HistoryEntry) => {
      let final = entry.final;
      let truths = entry.truths;

      // DB-only entry: no cached data — fetch full state from checkpoint
      if (entry.jobId && !final) {
        try {
          const res = await fetch(`${getApiBase()}/jobs/${entry.jobId}/state`);
          if (res.ok) {
            const data: { state?: Record<string, unknown> } = await res.json();
            const st = data.state ?? {};
            // Reconstruct truths
            const rawTruths = st["truths"] as Array<{ truth_id: string; fact: string; category: string }> | undefined;
            if (rawTruths?.length) {
              truths = rawTruths.map((t) => ({
                id: t.truth_id,
                fact_text: t.fact,
                category: t.category as import("@/lib/types").TruthCategory,
              }));
            }
            // Reconstruct final
            const masterCutUri = st["master_cut_uri"] as string | undefined;
            const ex = st["exports"] as { aspect_9x16?: string; aspect_1x1?: string; aspect_16x9?: string } | undefined;
            if (ex) {
              final = {
                duration: "30s",
                masterCutUri,
                ratios: [
                  { ...FINAL_RATIOS[0], url: ex.aspect_9x16 },
                  { ...FINAL_RATIOS[1], url: ex.aspect_1x1 },
                  { ...FINAL_RATIOS[2], url: ex.aspect_16x9 },
                ],
              };
            }
          }
        } catch { /* best-effort */ }
      }

      if (entry.jobId && final) {
        try {
          const res = await fetch(`/api/jobs/${entry.jobId}/exports`);
          if (res.ok) {
            const fresh: Record<string, string> = await res.json();
            final = {
              ...final,
              masterCutUri: fresh.master_cut || final.masterCutUri,
              ratios: (final.ratios ?? FINAL_RATIOS).map((r) => ({
                ...r,
                url: fresh[`aspect_${r.id}`] ?? r.url,
              })),
            };
          }
        } catch {
          // OSS not configured or network error — show with stale URLs
        }
      }
      setState({
        status: "dashboard",
        jobDone: true,
        maxPhaseIdx: PHASES.length - 1,
        truths,
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
        final,
        jobId: entry.jobId ?? null,
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

      {state.error && (
        <div
          role="alert"
          style={{
            position: "fixed",
            bottom: "1.5rem",
            left: "50%",
            transform: "translateX(-50%)",
            background: "var(--error, #c0392b)",
            color: "#fff",
            padding: "0.75rem 1.25rem",
            borderRadius: "8px",
            display: "flex",
            alignItems: "center",
            gap: "1rem",
            zIndex: 9999,
            maxWidth: "min(90vw, 480px)",
            boxShadow: "0 4px 16px rgba(0,0,0,0.25)",
            fontSize: "0.875rem",
          }}
        >
          <span style={{ flex: 1 }}>{state.error}</span>
          <button
            onClick={() => setState({ error: null })}
            style={{ background: "none", border: "none", color: "#fff", cursor: "pointer", padding: "0 0.25rem", fontSize: "1rem", lineHeight: 1 }}
            aria-label="Dismiss error"
          >
            ×
          </button>
        </div>
      )}

      {state.status === "dashboard" && (
        <Dashboard
          maxPhaseIdx={state.maxPhaseIdx}
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
          jobId={state.jobId}
        />
      )}

      {state.status === "library" && (
        <Library history={state.history} onClose={closeLibrary} onOpen={openHistoryItem} />
      )}
    </div>
  );
}
