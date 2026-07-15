"use client";

import type { ChangeEvent, DragEvent, KeyboardEvent, RefObject } from "react";
import BriefStep from "./BriefStep";
import DirectionStep from "./DirectionStep";
import PhotosStep from "./PhotosStep";
import SummaryStep from "./SummaryStep";
import type { Photo } from "./types";

const STEP_MS = 340;

export interface WizardProps {
  step: 1 | 2 | 3 | 4;
  transitioning: boolean;

  photos: Photo[];
  hoveredPhoto: number | null;
  dragOver: boolean;
  fileRef: RefObject<HTMLInputElement | null>;
  onPickClick: () => void;
  onDragOver: (e: DragEvent<HTMLDivElement>) => void;
  onDragLeave: (e: DragEvent<HTMLDivElement>) => void;
  onDrop: (e: DragEvent<HTMLDivElement>) => void;
  onFileChange: (e: ChangeEvent<HTMLInputElement>) => void;
  onPhotoEnter: (i: number) => void;
  onPhotoLeave: () => void;
  onRemovePhoto: (i: number) => void;

  brief: string;
  onBriefInput: (e: ChangeEvent<HTMLInputElement>) => void;
  onBriefKey: (e: KeyboardEvent<HTMLInputElement>) => void;

  moodWords: string[];
  moodInput: string;
  onMoodInput: (value: string) => void;
  onMoodKey: (e: KeyboardEvent<HTMLInputElement>) => void;
  onRemoveMood: (i: number) => void;
  refLink: string;
  onRefInput: (e: ChangeEvent<HTMLInputElement>) => void;
  neverList: string[];
  neverInput: string;
  onNeverInput: (value: string) => void;
  onNeverKey: (e: KeyboardEvent<HTMLInputElement>) => void;
  onRemoveNever: (i: number) => void;
  notes: string;
  onNotesInput: (e: ChangeEvent<HTMLTextAreaElement>) => void;

  goNext: () => void;
  goBack: () => void;
  goStep: (n: 1 | 2 | 3 | 4) => void;
  onGenerate: () => void;
}

export default function Wizard(props: WizardProps) {
  const { step, transitioning, photos, brief, goNext, goBack, goStep, onGenerate } = props;

  const nextDisabled = (step === 1 && photos.length < 1) || (step === 2 && !brief.trim());
  const showBack = step > 1;
  const showSkip = step === 3;
  const showNext = step <= 3;
  const showGenerate = step === 4;
  const nextLabel = step === 3 ? "Review →" : "Next →";
  let nextHint = "";
  if (step === 1 && photos.length < 1) nextHint = "Add at least one photo to continue";
  else if (step === 2 && !brief.trim()) nextHint = "Write a one-line brief to continue";

  const stepNum = step <= 3 ? `0${step}` : "✓";
  const stepLabel = step <= 3 ? `Step ${step} / 3` : "Review";

  const wizardWrapStyle = {
    animation: transitioning
      ? `pc-studio-step-out ${STEP_MS}ms var(--ease) forwards`
      : `pc-studio-step-in ${STEP_MS}ms var(--ease)`,
  };

  const nextBtnStyle = {
    fontFamily: "var(--font-sans)",
    fontSize: 14,
    fontWeight: 600,
    padding: "12px 0",
    borderTop: "none",
    borderLeft: "none",
    borderRight: "none",
    borderBottom: nextDisabled ? "1px solid transparent" : "1px solid var(--ink)",
    background: "transparent",
    ...(nextDisabled
      ? { cursor: "not-allowed" as const, color: "var(--faint)" }
      : { cursor: "pointer" as const, color: "var(--ink)" }),
  };
  const generateBtnStyle = {
    fontFamily: "var(--font-sans)",
    fontSize: 15,
    fontWeight: 700,
    padding: "16px 34px",
    border: "none",
    cursor: "pointer",
    background: "var(--accent)",
    color: "var(--accent-ink)",
    transition: "transform .4s var(--ease), box-shadow .4s var(--ease)",
  };

  return (
    <main
      data-rid="wizard-main"
      style={{
        maxWidth: 1020,
        margin: "0 auto",
        padding: "26px 56px 64px",
        position: "relative",
        overflow: "hidden",
        minHeight: "calc(100vh - 160px)",
      }}
    >
      <div
        data-rid="ghost-num"
        style={{
          position: "absolute",
          top: -40,
          right: -10,
          fontFamily: "var(--font-serif)",
          fontStyle: "italic",
          fontWeight: 400,
          fontSize: 280,
          lineHeight: 1,
          color: "var(--hair)",
          pointerEvents: "none",
          userSelect: "none",
          zIndex: 0,
        }}
      >
        {stepNum}
      </div>

      <div style={{ position: "relative", zIndex: 1, display: "flex", flexDirection: "column", minHeight: "calc(100vh - 210px)" }}>
        <div data-rid="wizard-toprow" style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 54 }}>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 12, letterSpacing: "2px", textTransform: "uppercase", color: "var(--faint)" }}>
            {stepLabel}
          </span>
          <span style={{ fontFamily: "var(--font-sans)", fontSize: 12, fontWeight: 600, letterSpacing: "1px", textTransform: "uppercase", color: "var(--faint)" }}>
            For Etsy &amp; Shopify makers
          </span>
        </div>

        <div style={{ ...wizardWrapStyle, flex: 1 }}>
          {step === 1 && (
            <PhotosStep
              photos={props.photos}
              hoveredPhoto={props.hoveredPhoto}
              dragOver={props.dragOver}
              fileRef={props.fileRef}
              onPickClick={props.onPickClick}
              onDragOver={props.onDragOver}
              onDragLeave={props.onDragLeave}
              onDrop={props.onDrop}
              onFileChange={props.onFileChange}
              onPhotoEnter={props.onPhotoEnter}
              onPhotoLeave={props.onPhotoLeave}
              onRemovePhoto={props.onRemovePhoto}
            />
          )}
          {step === 2 && <BriefStep brief={brief} onBriefInput={props.onBriefInput} onBriefKey={props.onBriefKey} />}
          {step === 3 && (
            <DirectionStep
              moodWords={props.moodWords}
              moodInput={props.moodInput}
              onMoodInput={props.onMoodInput}
              onMoodKey={props.onMoodKey}
              onRemoveMood={props.onRemoveMood}
              refLink={props.refLink}
              onRefInput={props.onRefInput}
              neverList={props.neverList}
              neverInput={props.neverInput}
              onNeverInput={props.onNeverInput}
              onNeverKey={props.onNeverKey}
              onRemoveNever={props.onRemoveNever}
              notes={props.notes}
              onNotesInput={props.onNotesInput}
            />
          )}
          {step === 4 && (
            <SummaryStep
              photos={photos}
              brief={brief}
              moodWords={props.moodWords}
              refLink={props.refLink}
              neverList={props.neverList}
              notes={props.notes}
              onEditPhotos={() => goStep(1)}
              onEditBrief={() => goStep(2)}
              onEditDirection={() => goStep(3)}
            />
          )}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 28, marginTop: 48, paddingTop: 22, borderTop: "1px solid var(--hair)" }}>
          {showBack && (
            <button
              onClick={goBack}
              className="pcs-hover-ink"
              style={{ fontFamily: "var(--font-sans)", fontSize: "13.5px", fontWeight: 600, color: "var(--ink-soft)", background: "none", border: "none", cursor: "pointer", padding: "6px 0", letterSpacing: "0.2px" }}
            >
              ← Back
            </button>
          )}
          <div style={{ flex: 1 }} />
          {showSkip && (
            <button
              onClick={goNext}
              className="pcs-hover-ink"
              style={{ fontFamily: "var(--font-sans)", fontSize: "13.5px", fontWeight: 600, color: "var(--faint)", background: "none", border: "none", cursor: "pointer", padding: "6px 0" }}
            >
              Skip for now
            </button>
          )}
          {showNext && (
            <button onClick={goNext} disabled={nextDisabled} style={nextBtnStyle}>
              {nextLabel}
            </button>
          )}
          {showGenerate && (
            <button onClick={onGenerate} style={generateBtnStyle}>
              Generate my ad →
            </button>
          )}
        </div>
        {nextHint && (
          <div style={{ textAlign: "right", marginTop: 10, fontFamily: "var(--font-sans)", fontSize: 12, color: "var(--faint)" }}>
            {nextHint}
          </div>
        )}
      </div>
    </main>
  );
}
