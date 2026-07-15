"use client";

import { useCallback, useEffect, useRef } from "react";
import type { ChangeEvent, DragEvent, KeyboardEvent } from "react";
import Header from "../components/Header";
import Wizard from "../components/wizard/Wizard";
import type { Photo } from "../components/wizard/types";
import Dashboard from "../components/dashboard/Dashboard";
import Library from "../components/library/Library";
import { createMockJob } from "@/lib/mockStream";
import { PRODUCT } from "@/lib/mockData";
import type {
  Budget,
  Final,
  HistoryEntry,
  Interrupt,
  InterruptResolution,
  JobEvent,
  MergeValidation,
  MockJob,
  Script,
  Shot,
  Treatment,
  Truth,
} from "@/lib/types";
import { useMergeState } from "@/lib/useMergeState";
import "./studio.css";

const STEP_MS = 340;
const HISTORY_KEY = "pc-job-history";

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
  refLink: string;
  neverList: string[];
  neverInput: string;
  notes: string;
  dragOver: boolean;

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
    refLink: "",
    neverList: [],
    neverInput: "",
    notes: "",
    dragOver: false,

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
    budget: { shots: [], running: 0, cap: 0, unit: "" },
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

export default function StudioPage() {
  const [state, setState] = useMergeState<State>(initialState);

  const fileRef = useRef<HTMLInputElement | null>(null);
  const timersRef = useRef<Array<ReturnType<typeof setTimeout>>>([]);
  const jobRef = useRef<MockJob | null>(null);
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
    let saved: string | null = null;
    try {
      saved = localStorage.getItem("pc-theme");
    } catch {
      // ignore
    }
    if (saved === "light" || saved === "dark") {
      setState({ theme: saved });
    }

    let history: HistoryEntry[] = [];
    try {
      history = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    } catch {
      // ignore
    }
    setState({ history });

    try {
      const params = new URLSearchParams(window.location.search);
      if (params.get("view") === "library") setState({ status: "library" });
    } catch {
      // ignore
    }

    return () => {
      clearTimers();
      jobRef.current?.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggleTheme = useCallback(() => {
    setState((s) => {
      const theme: Theme = s.theme === "dark" ? "light" : "dark";
      try {
        localStorage.setItem("pc-theme", theme);
      } catch {
        // ignore
      }
      return { theme };
    });
  }, [setState]);

  // ---- wizard nav ----
  // Reads `state` directly (rather than via a setState updater) since goStep
  // itself needs to schedule a timeout side effect — updater functions must stay pure.
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
  const onPickClick = useCallback(() => {
    fileRef.current?.click();
  }, []);

  const addFiles = useCallback(
    (fileList: FileList | null) => {
      if (!fileList) return;
      setState((s) => {
        const room = 3 - s.photos.length;
        if (room <= 0) return {};
        const mapped: Photo[] = Array.from(fileList)
          .filter((f) => f.type.startsWith("image/"))
          .slice(0, room)
          .map((f) => ({ name: f.name, url: URL.createObjectURL(f) }));
        return { photos: [...s.photos, ...mapped].slice(0, 3) };
      });
    },
    [setState],
  );

  const onFileChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      addFiles(e.target.files);
      e.target.value = "";
    },
    [addFiles],
  );
  const onDragOver = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setState((s) => (s.dragOver ? {} : { dragOver: true }));
    },
    [setState],
  );
  const onDragLeave = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setState({ dragOver: false });
    },
    [setState],
  );
  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setState({ dragOver: false });
      addFiles(e.dataTransfer.files);
    },
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
      if (e.key === "Enter" && state.brief.trim()) {
        e.preventDefault();
        goStep(3);
      }
    },
    [state.brief, goStep],
  );
  const onRefInput = useCallback((e: ChangeEvent<HTMLInputElement>) => setState({ refLink: e.target.value }), [setState]);
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
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") {
        e.preventDefault();
        addTag("moodWords", "moodInput");
      }
    },
    [addTag],
  );
  const onNeverKey = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") {
        e.preventDefault();
        addTag("neverList", "neverInput");
      }
    },
    [addTag],
  );
  const removeTag = useCallback(
    (listKey: "moodWords" | "neverList", i: number) => {
      setState((s) => ({ [listKey]: s[listKey].filter((_, idx) => idx !== i) }) as Partial<State>);
    },
    [setState],
  );

  // ---- generate -> live dashboard (mock stream) ----
  const handleEvent = useCallback(
    (e: JobEvent) => {
      const { type, payload } = e;
      switch (type) {
        case "node_started":
          setState({ phase: payload.phase, phaseLabel: payload.label });
          break;
        case "truth_extracted":
          setState((s) => ({ truths: [...s.truths, payload.truth] }));
          break;
        case "critic_score":
          setState((s) =>
            s.scripts.some((x) => x.id === payload.script.id) ? {} : { scripts: [...s.scripts, payload.script] },
          );
          break;
        case "critic_done":
          setState({ winnerId: payload.winnerId, merge: payload.merge, activeScriptId: payload.winnerId });
          break;
        case "treatment_ready":
          setState({ treatment: payload.treatment });
          break;
        case "budget_updated":
          setState((s) => ({
            budget: {
              cap: payload.cap,
              unit: payload.unit,
              running: payload.running,
              shots: s.budget.shots.some((x) => x.id === payload.shot.id) ? s.budget.shots : [...s.budget.shots, payload.shot],
            },
          }));
          break;
        case "shots_init":
          setState({ shots: payload.shots });
          break;
        case "shot_generated":
          setState((s) => ({
            shots: s.shots.map((sh) => (sh.id === payload.id ? { ...sh, status: payload.status } : sh)),
          }));
          break;
        case "drift_scored":
          setState((s) => ({ drift: { ...s.drift, [payload.shotId]: payload.score } }));
          break;
        case "interrupt_requested":
          setState({ interrupt: payload.interrupt });
          break;
        case "interrupt_resolved":
          setState({ interrupt: null, interruptResolution: payload.resolution });
          break;
        case "job_complete":
          if (elapsedIntervalRef.current) {
            clearInterval(elapsedIntervalRef.current);
            elapsedIntervalRef.current = null;
          }
          setState((s) => {
            const entry: HistoryEntry = {
              productName: payload.product?.name || PRODUCT.name,
              date: Date.now(),
              truths: s.truths,
              final: payload.final,
            };
            const history = [entry, ...s.history].slice(0, 20);
            try {
              localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
            } catch {
              // ignore
            }
            return { final: payload.final, jobDone: true, phase: "Delivery", history };
          });
          break;
        default:
          break;
      }
    },
    [setState],
  );

  const onGenerate = useCallback(() => {
    if (state.transitioning) return;
    const payload = {
      brief: state.brief,
      photoCount: state.photos.length,
      direction: { moodWords: state.moodWords, referenceAd: state.refLink, neverDo: state.neverList, notes: state.notes },
    };
    console.log("[ProductCut] POST /api/ingest", payload);
    setState({ transitioning: true });
    const t = setTimeout(() => {
      setState({ transitioning: false, status: "dashboard", elapsed: 0, jobDone: false });
      startedAtRef.current = Date.now();
      const el = setInterval(() => {
        setState({ elapsed: Date.now() - startedAtRef.current });
      }, 500);
      elapsedIntervalRef.current = el;
      pushTimer(el);
      const job = createMockJob();
      jobRef.current = job;
      job.on(handleEvent);
      job.start();
    }, STEP_MS);
    pushTimer(t);
  }, [state, handleEvent, setState]);

  const resolveInterrupt = useCallback((resolution: InterruptResolution) => {
    jobRef.current?.resume(resolution);
  }, []);

  const resetPipeline = useCallback(() => {
    clearTimers();
    jobRef.current?.stop();
    jobRef.current = null;
    setState((s) => ({ ...initialState(), theme: s.theme, history: s.history }));
  }, [setState]);

  // ---- library ("My Ads") ----
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
        budget: { shots: [], running: 0, cap: 0, unit: "" },
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
          refLink={state.refLink}
          onRefInput={onRefInput}
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
          onRetry={() => resolveInterrupt("retry")}
          onFallback={() => resolveInterrupt("fallback")}
          final={state.final}
        />
      )}

      {state.status === "library" && <Library history={state.history} onClose={closeLibrary} onOpen={openHistoryItem} />}
    </div>
  );
}
