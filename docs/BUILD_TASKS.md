
# ProductCut — Build Tasks (KR & RR)

**Team:** KR (`KRKR1704`) and RR (Rithvik Ramdas — repo owner).
**Source of truth for *why*/*how*:** `docs/PHASE_PLAN.md` (phase-by-phase build plan, estimates, exit criteria) and `docs/TECHNICAL_DOCUMENTATION.md` (agent-by-agent spec). This file is the *who-does-what* checklist derived from those two documents — if a task description here feels thin, the full spec is in those docs.

## How the split works

Both of you are full-stack, so every phase is split **down the middle**, not by fixed lane. Each person gets a mix of "Brain" work (LangGraph/agent logic/Qwen calls/backend) and "Body" work (Next.js frontend/dashboard/ffmpeg/Alibaba Cloud infra) in every phase, so neither of you gets siloed into only-backend or only-frontend for the whole project.

- `[BRAIN]` = orchestration / agent logic / Qwen-DashScope call / backend
- `[BODY]` = frontend / dashboard / ffmpeg / cloud infra
- `[JOINT]` = must be agreed by both before either builds on it (the "contract handshakes")

Assignments below are a **starting split** — swap freely between yourselves as long as (a) both people stay busy in parallel, and (b) the person who builds a thing's producer side and the person who builds its consumer side agree on the shape *before* either starts (that's what the Joint Handshakes are for).

**Already done:** public GitHub repo (`github.com/Rithvik1811/ProductCut`) ✅, MIT LICENSE committed ✅, `PROJECT_PROPOSAL.md` + `TECHNICAL_DOCUMENTATION.md` + `PHASE_PLAN.md` written ✅.

---

## Joint Handshakes (agree before parallel work starts on anything downstream)

These are shared contracts — draft together, commit as versioned files (schema/`.md`), and only extend additively afterward (add fields, don't rename/remove without a 2-minute sync).

- [ ] **C1 — LangGraph shared state schema** (job id, brief, photo refs, seller_direction, product_truths, script variants, critic trace, winning script, treatment, shot list w/ justification, per-shot video/status, drift scores, budget ledger, VO/caption timing, final outputs, chat/edit fields). *Draft: KR. Sign-off: both. Due: Phase 0.*
- [ ] **C2 — WebSocket event schema** (envelope + event types: `node_started`, `truth_extracted`, `critic_score`, `treatment_ready`, `budget_updated`, `drift_scored`, `shot_generated`, `interrupt_requested`, `edit_routed`, `job_complete`; every event `{type, job_id, ts, payload}`). *Draft: RR. Sign-off: both. Due: Phase 0.*
- [ ] **C3 — Shot-list JSON schema** (camera-literate shot object + `justification` sub-object; **no `product_category` field, ever**). *First cut: Phase 0. Frozen: end of Phase 2.*
- [ ] **C4 — Interrupt / human-review contract** (drift review payload + resume payload; reused by Phase 9's preview/confirm gate). *Drafted end of Phase 3, finalized Phase 4.*
- [ ] **C5 — Edit-request / checkpoint-fork contract** (Phase 9 only — Edit Router output schema, patch schema, fork-branch bookkeeping). *Start of Phase 9, only if you get there.*

---

## Phase 0 — De-risk & Environment Setup
*Goal: prove both external APIs work before writing orchestration code. Do not skip.*

### KR
- [x] `[BRAIN]` Get Qwen Cloud/DashScope creds working; trivial smoke-test calls to Qwen-Max, Qwen-VL, Qwen-TTS/CosyVoice; record model IDs/base URL/auth in `.env.example`
- [x] `[BRAIN]` One-off manual test of the Product Truth Extractor prompt against 3-4 real product photos — confirm it returns 4+ *specific*, non-generic facts
- [x] `[BODY]` Provision Alibaba Cloud **ECS instance**; provision **OSS bucket**
- [x] `[JOINT]` Draft **C1** (state schema)

### RR
- [ ] `[BRAIN]` **Critical de-risk task:** Wan/HappyHorse raw quality test — image-to-video only, 3-4 real product photos, 2-3 camera moves (push_in/orbit/static), a couple prompt styles. Save outputs, write an honest **go/no-go verdict**. Note latency + hard-failure behavior.
- [ ] `[BRAIN]` Scaffold FastAPI app + a bare LangGraph graph with one no-op node streaming a test event via `astream_events`; add `langgraph-checkpoint-postgres` dependency now
- [ ] `[BODY]` Provision **managed Postgres-compatible DB** (PolarDB/RDS); confirm ECS can reach OSS, DB, and DashScope+Wan endpoints (latency/firewall/egress check)
- [ ] `[BODY]` Scaffold Next.js/React/Tailwind app; WebSocket client that connects to KR's no-op endpoint and logs events
- [ ] `[BODY]` Sketch optional seller-intake form (mood words, reference-ad link, "never do this," freeform) — pure UI, no backend dependency yet
- [ ] `[JOINT]` Draft **C2** (co-author with KR) and first cut of **C3**

**Exit criteria (both agree before Phase 1):** real Wan/HappyHorse test clips + written verdict; Qwen-Max/VL/TTS calls all return valid responses; Truth Extractor manual test passes; ECS/OSS/DB round-trip confirmed; frontend logs a live WS test event; C1/C2/C3-draft committed.

---

## Phase 1 — Intake + Product Truth Extractor + Concept Agent + Critic Chain
*Goal: a scored, cross-pollinated winning script grounded in real product facts, zero video-gen dependency.*

### KR
- [x] `[BRAIN]` **Product Truth Extractor** node (Qwen-VL, 6-10 specific facts w/ `truth_id`/`category`, reject-and-reprompt heuristic on generic facts) — verified against real photos + automated tests (`backend/tests/test_product_truth_extractor.py`, `test_graph_build.py`)
- [ ] `[BRAIN]` **Concept Agent** (4 script variants, forced-distinct framework/hook/emotional-trigger, beat timestamps, `grounding_truth_ids`)
- [ ] `[BRAIN]` Critic Chain: **Hook-Checker** + **Pacing-Checker** (deterministic timing math)
- [ ] `[BODY]` Wire the intake form (from Phase 0) to the real ingest endpoint
- [ ] `[BODY]` Dashboard panel: `product_truths[]` list (proof grounding is real)

### RR
- [ ] `[BRAIN]` Wire `seller_direction` fields into `jobs`/`seller_direction` tables + C1 state (all fields nullable)
- [ ] `[BRAIN]` Critic Chain: **CTA-Checker** + **Tone-Checker** (incl. `never_do` hard-fail)
- [ ] `[BRAIN]` **Meta-Critic** (weighted composite: hook 25/pacing 20/completion 20/CTA 20/tone 15%, cross-pollinate hook+body+CTA, full reasoning trace); emit `truth_extracted`/`critic_score` events per C2
- [ ] `[BODY]` Minimal job-submission form (photos + one-line brief + optional intake) → ingest stub
- [ ] `[BODY]` Dashboard panel: critic reasoning trace + per-variant score table

**Exit criteria:** real brief → 6+ product truths → 4 valid grounded script variants → one merged winning script with a human-readable scoring trace, visible in the dashboard.

---

## Phase 2 — Treatment Agent + Justification-Forced Shot-List + Budget Gate + Ledger
*Goal: camera-literate, budget-capped shot list where every choice is justified by script + product facts, not category.*

### KR
- [ ] `[BRAIN]` **Treatment Agent** (`director_persona`, `color_story`, `pacing_philosophy`, `beat_treatments[]` with `script_quote`/`truth_fact_id`/`why_not_generic`; "category" word disallowed in output)
- [ ] `[BRAIN]` **Justification Validator** (deterministic: verbatim quote check, `truth_fact_id` exists, `treatment_ref` matches, stoplist reject; one re-prompt then fallback to literal `visual_approach`)
- [ ] `[BODY]` Director's treatment panel in dashboard (persona/color story/pacing/per-beat justification)
- [ ] `[BODY]` DB migrations for ledger + treatment + shot-list-with-justification tables

### RR
- [ ] `[BRAIN]` **Shot-List Agent** (3-7 shots, camera-literate schema per **C3**, `justification` object; freeze C3 here; confirm no `product_category` field anywhere)
- [ ] `[BRAIN]` **Budget Gate** (deterministic cap check, one loop-back to Shot-List Agent if over cap, then enforce); write allocations to Budget Ledger table; emit `treatment_ready`/`budget_updated` events
- [ ] `[BODY]` Live budget ledger panel (per-shot allocations, running total, cap line, over/under state) + per-shot justification tooltip

**Exit criteria:** valid 3-7 shot list conforming to frozen C3, budgets sum within cap (loop-back demonstrably works when seeded over budget), every justification passes the validator (re-prompt demonstrably works when seeded generic), ledger/treatment/shot rows visible in DB + dashboard.

---

## Phase 3 — Video-Gen Node + Graceful Degradation
*Goal: generate every shot in parallel with a hard-failure path that never blocks the pipeline.*

### KR
- [ ] `[BRAIN]` **Video-Gen Node**: LangGraph parallel fan-out via `Send()`; structured prompt formula (Subject→Action→Camera→Lighting→Composition→Mood→Quality) + `negative_prompt` + reference product photo (image-to-video only, never text-to-video)
- [ ] `[BODY]` Dashboard: per-shot generation status grid (queued/generating/done/fallback) with thumbnails

### RR
- [ ] `[BRAIN]` **Ken-Burns Fallback Node** — on hard API failure/timeout, route straight to fallback, **no retry consumed**
- [ ] `[BRAIN]` Upload generated shot assets to OSS; record per-shot status in state; emit `shot_generated` events (real vs. fallback)
- [ ] `[BODY]` Verify OSS read access from frontend (signed URLs or backend proxy) for shot previews

**Exit criteria:** full shot list fans out and returns a clip per shot in OSS, visibly parallel; killing one shot's API call routes it to Ken-Burns fallback without blocking the rest or consuming a retry; all statuses render live.

---

## Phase 4 — Continuity Agent + Human-in-the-Loop Review
*Goal: catch visual drift, auto-retry within a hard cap, escalate to a real human-review interrupt when exhausted.*

### KR
- [ ] `[BRAIN]` **Continuity Agent** (Qwen-VL): compare generated frame vs. reference photo + shared lighting/style string, return drift/consistency score
- [ ] `[BRAIN]` **Human-in-the-loop**: real LangGraph `interrupt()` carrying the **C4** payload; on resume apply `approve`/`retry-with-edit`/`accept-fallback`
- [ ] `[BODY]` Dashboard: continuity drift-score panel per shot

### RR
- [ ] `[BRAIN]` **Capped retry loop** (drift > threshold and retries < 2 → loop back to Video-Gen; hard cap at 2 — this is the *only* place retries are consumed); emit `drift_scored`/`interrupt_requested` events
- [ ] `[BODY]` **Human-review UI**: surface the interrupt (shot + drift score + candidate frames from OSS), offer the 3 actions, post resume payload per C4

**Scope-cut fallback (decide by start of phase, not mid-build):** if short on time, degrade to "flag only, no auto-retry/interrupt" — documented, not silently dropped.

**Exit criteria:** a drifting shot triggers up to 2 auto-regens; exhausted retries raise a real interrupt that pauses, surfaces in UI, and resumes correctly on all 3 human choices (or the documented flag-only fallback).

---

## Phase 5 — Voiceover/Captions + Assembly + Multi-Format Export
*Goal: a finished, captioned, voiced ad exported in 9:16 / 1:1 / 16:9 from the same shots.*

### KR
- [ ] `[BRAIN]` **Voiceover + Caption Agent** (Qwen TTS/CosyVoice) — parallel branch starting as soon as the script is final (doesn't wait on video-gen); VO audio + caption timing synced to script beats
- [ ] `[BODY]` **Format Export Node** (ffmpeg, deterministic): recompose the master cut into 9:16 / 1:1 / 16:9 using the reserved `text_overlay_zone`

### RR
- [ ] `[BODY]` **Assembly Agent** (ffmpeg, deterministic): stitch approved/fallback shots + VO audio + burned-in captions/CTA in `text_overlay_zone` + transitions/music timing
- [ ] `[BODY]` Upload final videos to OSS; surface for download/preview in the frontend

**Exit criteria:** one run produces a finished 15-30s ad with synced VO + burned captions/CTA, exported in all 3 aspect ratios, downloadable from OSS via the frontend.

---

## Phase 6 — Frontend Dashboard + Realtime Streaming
*Goal: one live dashboard streaming the whole run end to end.*

### KR
- [ ] `[BRAIN]` Ensure FastAPI emits the **complete C2 event stream** from `astream_events` for a full run; fix any gaps found once the dashboard consumes a real run
- [ ] `[BODY]` Help assemble panels into the unified dashboard (job progress + product-truths panel)

### RR
- [ ] `[BODY]` Assemble remaining panels into one coherent live dashboard (treatment, budget ledger, critic trace, continuity drift, per-shot status w/ justification, human-review surfacing, final video previews)
- [ ] `[BODY]` Wire real `astream_events` over WebSocket end-to-end (replace mock feeds); handle reconnect + late-join (render current state on connect)

**Exit criteria:** kicking off a real job from the frontend drives the entire dashboard live, no page refresh, ingest-to-export.

---

## Phase 7 — Deployment + Alibaba Cloud Proof
*Goal: whole system runs on Alibaba Cloud, with recorded proof.*

### KR
- [ ] `[BRAIN]` Confirm backend runs cleanly on ECS (env config, process mgmt, WS stays up across multi-minute runs); confirm DashScope + Wan calls work from inside Alibaba Cloud; confirm checkpoint tables reachable/writable in deployed DB
- [ ] `[BODY]` Capture **Proof of Deployment**: recorded clip + linked code file showing actual OSS upload, DB write, ECS execution (executed, not just referenced)

### RR
- [ ] `[BODY]` Deploy FastAPI backend to ECS; deploy/serve Next.js frontend; wire OSS + managed DB in the deployed environment
- [ ] `[JOINT]` Run one full job end-to-end against the deployed stack together and sign off

**Exit criteria:** full job runs end-to-end on deployed ECS + OSS + managed DB; Proof-of-Deployment recording + linked code file captured.

---

## Phase 8 — Polish, Demo Prep, Stretch
*Goal: a reliable, judge-ready submission; stretch only if genuinely ahead.*

### KR
- [ ] `[BRAIN]` **Demo safety net**: pre-warmed cache/replay path for a known-good job so demo day never depends on a live cold Wan/HappyHorse call (single most important demo-day insurance item)
- [ ] `[BODY]` Export the architecture diagram (from `TECHNICAL_DOCUMENTATION.md`'s Mermaid diagram) as an image for the submission page

### RR
- [ ] `[BODY]` Record the ~3-minute demo video (agent handoffs, truths/treatment panels, critic trace, budget dashboard, continuity/human-review, multi-aspect final video); upload publicly
- [ ] `[BODY]` Final documentation pass: README, feature description, confirm license visible in repo About section, track clearly identified

### Joint, only if ahead of schedule
- [ ] `[BRAIN]` A/B hook-variant export (same body, two hooks, two exports)
- [ ] `[BRAIN]` Dynamic mid-pipeline budget renegotiation
- [ ] ~~Cross-session brand-style memory~~ — explicitly cut, do not build

**Exit criteria:** Definition of Done (below) fully green; replay/cache path works independent of live API latency; demo video + architecture diagram finalized and uploaded.

---

## Phase 9 — Chat-Based Post-Generation Revision *(stretch, only start if Phases 0-7 are fully green)*
*Goal: targeted, cost-scoped fixes via chat instead of a full pipeline re-run.*

### KR
- [ ] `[BRAIN]` **Edit Router** (Qwen classification → `{scope, target_shot_ids[], entry_node, confidence, rationale}` per the routing table in `TECHNICAL_DOCUMENTATION.md` §5.16.1); below confidence threshold, ask a clarifying question instead of guessing
- [ ] `[BRAIN]` **Checkpoint-fork mechanism**: `get_state_history(thread_id)`, `update_state(checkpoint_id, patch)`, `graph.invoke(None, {"configurable": {"thread_id": new_branch_id}})`; confirm untouched shots are copied by reference (no re-render)
- [ ] `[BODY]` Chat UI (message thread on a completed job → Edit Router endpoint)

### RR
- [ ] `[BRAIN]` **Edit Interpreter** (vague ask → grounded patch: `treatment_patch`, `shot_patches[]`, `justification`; routed through the **same Justification Validator from Phase 2**); emit `edit_routed`/`edit_patch_ready`/`fork_created` events per **C5**
- [ ] `[BODY]` Preview/Confirm diff panel (reuses the Phase 4 `interrupt()` UI pattern: old-vs-new script/shot-list/treatment + estimated incremental cost; explicit confirm required before Video-Gen)
- [ ] `[BODY]` Version picker (list/tree UI over `job_versions`: branch_id, parent_branch_id, summary)

**Exit criteria:** all 5 canonical edit messages ("make the hook punchier," "shot 2 is too dark," "shorten it to 15s," "change the CTA text," "make it more energetic") route correctly, produce a validator-passing patch, show a diff/cost preview, and on confirm only re-render the shots actually affected (verified against OSS timestamps).

---

## Master Definition of Done
*(from `docs/PHASE_PLAN.md` §4 — every box green before submission)*

**Submission requirements (hard):**
- [x] Public GitHub repo with MIT license visible in the About section
- [ ] Proof of Alibaba Cloud deployment — recorded clip + linked code file (Phase 7)
- [ ] Architecture diagram exported as an image (Phase 8)
- [ ] ~3-minute demo video, uploaded publicly (Phase 8)
- [ ] Text description of features/functionality (Phase 8)
- [ ] Track identified: Track 2 — AI Showrunner
- [ ] (Optional) Blog post for Blog Post Prize eligibility

**Core functionality (must ship):**
- [ ] Seller submits 2-3 photos + one-line brief (+ optional intake) → finished ad video
- [ ] Product Truth Extractor returns 6+ specific facts consumed downstream
- [ ] Concept Agent + Critic Chain produce a scored, cross-pollinated winning script with a visible trace
- [ ] Treatment Agent produces a grounded director's treatment
- [ ] Shot-List Agent + Justification Validator + Budget Gate + Ledger all pass
- [ ] Video-Gen parallel fan-out + Ken-Burns fallback
- [ ] Continuity Agent drift scoring + capped retry + human-review interrupt (or documented flag-only degrade)
- [ ] VO + captions + Assembly + multi-aspect export (9:16/1:1/16:9)
- [ ] Live dashboard streams the full run from `astream_events`
- [ ] Whole pipeline runs on deployed Alibaba Cloud stack, not just locally

**Demo-day insurance:**
- [ ] Pre-warmed cache/replay path
- [ ] Demo video walks all the agent handoffs + panels

**Stretch (only if everything above is green and stable):**
- [ ] Phase 9: chat-based revision, demonstrated on the 5 canonical edit requests
- [ ] A/B hook-variant export
- [ ] Dynamic mid-pipeline budget renegotiation
- [ ] ~~Cross-session brand-style memory~~ — explicitly cut, do not build
