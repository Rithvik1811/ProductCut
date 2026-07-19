"use client";

import type { KeyboardEvent } from "react";
import Link from "next/link";
import type {
  Budget,
  Final,
  Interrupt,
  InterruptResolution,
  MergeValidation,
  Script,
  Shot,
  Treatment,
  Truth,
} from "@/lib/types";
import { PHASES, PHASE_RUNNING_LABEL, estimateDuration } from "@/lib/phases";
import TruthsPanel from "./panels/TruthsPanel";
import ScriptsPanel from "./panels/ScriptsPanel";
import TreatmentPanel from "./panels/TreatmentPanel";
import BudgetPanel from "./panels/BudgetPanel";
import ShotsPanel from "./panels/ShotsPanel";
import ContinuityPanel from "./panels/ContinuityPanel";
import FinalPanel from "./panels/FinalPanel";

export interface DashboardProps {
  maxPhaseIdx: number;
  elapsed: number;
  jobDone: boolean;
  onResetPipeline: () => void;

  historyCount: number;
  onOpenLibrary: () => void;

  truths: Truth[];
  hoveredTruthId: string | null;
  onHoverTruth: (id: string | null) => void;

  scripts: Script[];
  activeScriptId: string | null;
  winnerId: string | null;
  merge: MergeValidation | null;
  onSelectScript: (id: string) => void;
  onScriptTabKey: (e: KeyboardEvent<HTMLButtonElement>, index: number) => void;

  treatment: Treatment | null;

  budget: Budget & { running: number };
  budgetOpenId: string | null;
  onToggleBudgetRow: (id: string) => void;

  shots: Shot[];
  shotOpenId: string | null;
  onToggleShot: (id: string) => void;

  drift: Record<string, number>;
  driftThreshold: number;
  interrupt: Interrupt | null;
  interruptResolution: InterruptResolution | null;
  onApprove: () => void;
  onRetry: () => void;
  onFallback: () => void;

  final: Final | null;
  jobId?: string | null;
}

function formatElapsed(ms: number): string {
  const secs = Math.floor(ms / 1000);
  const mm = String(Math.floor(secs / 60)).padStart(2, "0");
  const ss = String(secs % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

export default function Dashboard(props: DashboardProps) {
  const {
    maxPhaseIdx,
    elapsed,
    jobDone,
    onResetPipeline,
    historyCount,
    onOpenLibrary,
    truths,
    hoveredTruthId,
    onHoverTruth,
    scripts,
    activeScriptId,
    winnerId,
    merge,
    onSelectScript,
    onScriptTabKey,
    treatment,
    budget,
    budgetOpenId,
    onToggleBudgetRow,
    shots,
    shotOpenId,
    onToggleShot,
    drift,
    driftThreshold,
    interrupt,
    interruptResolution,
    onApprove,
    onRetry,
    onFallback,
    final,
    jobId,
  } = props;

  // -1 (nothing received yet) reads as "Ingest in progress", the real starting state.
  const curPhaseIdx = Math.max(0, Math.min(maxPhaseIdx, PHASES.length - 1));
  const jobStatusLine = jobDone
    ? `Job complete · ${truths.length} truths · winning cut delivered`
    : PHASE_RUNNING_LABEL[PHASES[curPhaseIdx]];

  const hasTruths = truths.length > 0;
  const hasScripts = scripts.length > 0;
  const hasBudget = budget.shots.length > 0;
  const hasShots = shots.length > 0;
  const hasInterrupt = !!interrupt;
  const hasDriftPanel = hasInterrupt || Object.keys(drift).length > 0 || !!interruptResolution;
  const hasOps = hasBudget || hasShots || hasDriftPanel;
  const historyCountLabel = historyCount ? ` (${historyCount})` : "";

  // Progress: completed-stages/8, refined within the Shots stage using real
  // per-shot status once the total shot count is known from the budget ledger.
  const shotCount = budget.shots.length || shots.length;
  let stageFraction = 0;
  if (!jobDone && curPhaseIdx === PHASES.indexOf("Shots") && shotCount > 0) {
    const doneShots = shots.filter((s) => s.status === "passed" || s.status === "fallback").length;
    stageFraction = Math.min(doneShots / shotCount, 1);
  }
  const progress = jobDone ? 1 : (curPhaseIdx + stageFraction) / PHASES.length;
  const estimateLabel = estimateDuration(shotCount);

  return (
    <div>
      <div
        data-rid="status-bar"
        style={{ position: "sticky", top: 67, zIndex: 20, background: "var(--paper)", borderBottom: "1px solid var(--hair)", padding: "16px 48px" }}
      >
        <div style={{ maxWidth: 1180, margin: "0 auto", display: "flex", alignItems: "center", gap: 24, flexWrap: "wrap" }}>
          <div style={{ lineHeight: 1.25 }}>
            <div style={{ fontFamily: "var(--font-serif)", fontStyle: "italic", fontSize: 17 }}>ProductCut</div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, letterSpacing: "0.4px", color: "var(--faint)" }}>{jobStatusLine}</div>
          </div>
          <div
            data-rid="phase-chips"
            style={{ flex: 1, display: "flex", alignItems: "center", gap: 10, justifyContent: "center", flexWrap: "wrap", fontFamily: "var(--font-sans)", fontSize: 12, letterSpacing: "0.3px", position: "relative" }}
          >
            {PHASES.map((p, i) => {
              // done/active/not-started are derived only from curPhaseIdx, which
              // itself only ever advances on a real C2 event (see studio/page.tsx's
              // maxPhaseIdx) — a stage with no event yet always reads not-started.
              const done = jobDone || i < curPhaseIdx;
              const active = !jobDone && i === curPhaseIdx;
              const notStarted = !done && !active;
              const color = active ? "var(--ink)" : done ? "var(--ink-soft)" : "var(--faint)";
              return (
                <span
                  key={p}
                  style={{
                    color,
                    fontWeight: done ? 700 : 400,
                    textDecorationLine: active ? "underline" : "none",
                    textDecorationColor: "var(--accent)",
                    textUnderlineOffset: 4,
                    opacity: notStarted ? 0.55 : 1,
                  }}
                >
                  {p}
                </span>
              );
            })}
            <div data-rid="phase-fade" style={{ display: "none", position: "absolute", right: 0, top: 0, bottom: 0, width: 28, background: "linear-gradient(90deg, transparent, var(--paper))", pointerEvents: "none" }} />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 5, minWidth: jobDone ? undefined : 150 }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 6, fontFamily: "var(--font-mono)", fontSize: "12.5px", color: "var(--ink-soft)", whiteSpace: "nowrap" }}>
                <span>{formatElapsed(elapsed)}</span>
                {!jobDone && <span style={{ color: "var(--faint)", fontSize: 11 }}>/ {estimateLabel}</span>}
              </div>
              {!jobDone && (
                <div data-rid="progress-bar-track" style={{ width: "100%", height: 3, background: "var(--hair)", overflow: "hidden" }}>
                  <div
                    data-rid="progress-bar-fill"
                    style={{ height: "100%", width: `${Math.round(progress * 100)}%`, background: "var(--accent)", transition: "width .5s var(--ease)" }}
                  />
                </div>
              )}
            </div>
            <Link
              href="/"
              className="pcs-hover-ink"
              style={{ fontFamily: "var(--font-sans)", fontSize: 12, fontWeight: 600, color: "var(--ink-soft)", textDecoration: "none", borderBottom: "1px solid var(--hair-strong)", cursor: "pointer", padding: "8px 4px" }}
            >
              Home
            </Link>
            <button
              onClick={onOpenLibrary}
              className="pcs-hover-ink"
              style={{ fontFamily: "var(--font-sans)", fontSize: 12, fontWeight: 600, color: "var(--ink-soft)", background: "transparent", border: "none", borderBottom: "1px solid var(--hair-strong)", cursor: "pointer", padding: "8px 4px" }}
            >
              My Ads{historyCountLabel}
            </button>
            <button
              onClick={onResetPipeline}
              className="pcs-hover-ink"
              style={{ fontFamily: "var(--font-sans)", fontSize: 12, fontWeight: 600, color: "var(--ink-soft)", background: "transparent", border: "none", borderBottom: "1px solid var(--hair-strong)", cursor: "pointer", padding: "8px 4px" }}
            >
              New job
            </button>
          </div>
        </div>
      </div>

      {hasTruths && <TruthsPanel truths={truths} hoveredTruthId={hoveredTruthId} onHoverTruth={onHoverTruth} />}

      {hasScripts && (
        <ScriptsPanel
          scripts={scripts}
          activeScriptId={activeScriptId}
          winnerId={winnerId}
          merge={merge}
          onSelectScript={onSelectScript}
          onTabKeyDown={onScriptTabKey}
        />
      )}

      {treatment && (
        <TreatmentPanel treatment={treatment} truths={truths} hoveredTruthId={hoveredTruthId} onHoverTruth={onHoverTruth} />
      )}

      {hasOps && (
        <section data-rid="section-pad" style={{ background: "var(--paper-deep)", padding: "60px 48px 68px", animation: "pc-section-in 0.6s var(--ease) both" }}>
          <div style={{ maxWidth: 1180, margin: "0 auto" }}>
            <span style={{ fontFamily: "var(--font-sans)", fontSize: "11.5px", fontWeight: 700, letterSpacing: "1.5px", textTransform: "uppercase", color: "var(--ink-soft)", display: "block", marginBottom: 34 }}>
              Producer · Shot Generator · Continuity Guard
            </span>
            <div data-rid="ops-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 44, alignItems: "start" }}>
              {hasBudget && <BudgetPanel budget={budget} budgetOpenId={budgetOpenId} onToggle={onToggleBudgetRow} />}
              {hasShots && <ShotsPanel shots={shots} shotOpenId={shotOpenId} onToggle={onToggleShot} />}
              {hasDriftPanel && (
                <ContinuityPanel
                  shots={shots}
                  drift={drift}
                  driftThreshold={driftThreshold}
                  interrupt={interrupt}
                  interruptResolution={interruptResolution}
                  onApprove={onApprove}
                  onRetry={onRetry}
                  onFallback={onFallback}
                />
              )}
            </div>
          </div>
        </section>
      )}

      {final && <FinalPanel final={final} jobId={jobId ?? undefined} />}
    </div>
  );
}
