
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
- [ ] `[BRAIN]` **Critic Chain: Body-Checker** *(new — TECHNICAL_DOCUMENTATION.md §5.4.3)*: deterministic redundant-beat-pair pre-pass (lexical/embedding overlap on `beats[1:-1]`) + Qwen scoring call for `completion_score`/`promise_payoff_match`/`emotional_trigger_landed`; this is what backs the previously-undefined "Completion / structural fit" axis and gives the merge step an actual best-body signal
- [ ] `[BRAIN]` **Meta-Critic** (weighted composite: hook 25/pacing 20/completion 20 — now Body-Checker-backed/CTA 20/tone 15%, cross-pollinate hook+body+CTA using Body-Checker's per-variant score, re-derive contiguous beat timestamps for the merge, full reasoning trace); emit `truth_extracted`/`critic_score` events per C2
- [ ] `[BRAIN]` **Merge Coherence Validator** *(new — TECHNICAL_DOCUMENTATION.md §5.4.7, separate node from Meta-Critic)*: (a) re-run the Pacing-Checker's deterministic timing math against the merged script's re-derived beats — one repair-and-recheck attempt, then treat as failed; (b) a blind/cold independent Qwen-Plus coherence read (no merge rationale or source-variant info in its prompt) scoring voice/POV consistency, promise-payoff match, and register-shift at the two stitch points; on failure, **route by failure type** — voice/register failures to the Copy Editor, promise-payoff failures back to the Meta-Critic naming the flagged clash — then fall back to the single highest composite-scoring variant if the routed repair's re-check still fails; append every attempt to `merge_attempts[]`/`reasoning_trace`. **This is the fix for "the Meta-Critic grades its own merge" — the pass/fail call must live in this separate node, not inside the Meta-Critic's own call.**
- [ ] `[BRAIN]` **Copy Editor** *(new — TECHNICAL_DOCUMENTATION.md §5.4.8, required repair node, not optional)*: a Qwen-Plus call that fires when the Merge Coherence Validator flags a voice/register-consistency failure at the hook→body or body→CTA seam. Constrained, in-place polish only — may edit only the flagged transition text, must preserve every `grounding_truth_ids` claim and the single CTA verb, must stay within ~±10% of the original seam's word count so beat timing holds. Distinct node from both the Concept Agent (the writer) and the Meta-Critic (the merge-builder) — this is a copy-edit, not a rewrite. Output includes a before/after `original_seam_text`/`revised_seam_text` record for the trace. The patched merge is sent back through the Merge Coherence Validator for a full re-check (same retry slot as the existing swap path, not a new loop); a second failure falls back to the single highest composite-scoring variant exactly as the existing fallback does. Promise-payoff failures (missing content, not a voice mismatch) continue to use the existing Meta-Critic swap-to-second-best-piece path instead — the Copy Editor is never used for that failure type.
- [ ] `[BODY]` Minimal job-submission form (photos + one-line brief + optional intake) → ingest stub
- [ ] `[BODY]` Dashboard panel: critic reasoning trace + per-variant score table (now including the completion axis + a merge-attempts/validator-verdict view — including which repair path fired, Copy Editor seam polish or Meta-Critic swap, with before/after seam text when a copy-edit occurred — so a retry or fallback is visible, not hidden)

**Exit criteria:** real brief → 6+ product truths → 4 valid grounded script variants, each with a real completion/body score → one merged winning script whose merge has been independently re-validated (pacing re-check + blind coherence read passed, or a visible repair — Copy Editor seam polish or Meta-Critic swap, per which failure fired — followed by re-validation, or a fallback occurred) → human-readable scoring + merge-validation trace, visible in the dashboard.

---

## Phase 2 — Treatment Agent + Justification-Forced Shot-List + Budget Gate + Ledger
*Goal: camera-literate, budget-capped shot list where every choice is justified by script + product facts, not category.*

**Phase 2 research note (RR, see `docs/TECHNICAL_DOCUMENTATION.md` §5.6/§5.7 for full detail):** four Opus research passes (cinematography vocabulary, AI-video failure modes, grounded-generation technique, budget-allocation heuristics) landed three design revisions to the original single-call Shot-List Agent spec, now written into §5.6/§5.7 as the spec of record:
1. **Shot-List Agent is now two sequential Qwen calls** (Call A: justification only → Justification Validator → Call B: camera/composition fields conditioned on the validated justification), not one ordered-JSON call — Qwen/DashScope's `json_object` mode doesn't grammar-force key order the way OpenAI's `json_schema` mode does, so ordering alone wasn't a structural guarantee.
2. **C3 gets two additive enum values** (RR freezes C3 this phase): `camera_move += rack_focus`, `shot_type += product_in_hand` — both chosen because they're structurally hard to specify generically (a rack focus needs two named product referents; product_in_hand gives `demo`/`proof` beats a real composition instead of being forced into `lifestyle_context`).
3. **`allocated_budget` is real dollar cost** (`duration_sec × rate(resolution)`), allocated by a grounding-weighted formula (favors `hook`/`cta` beat roles, `macro_detail`/`hook_hero` shot types, and shots citing a `material`/`texture`/`construction_detail`/`imperfection` truth) rather than equal split, reusing the `_waterfill()` clamp-and-redistribute routine already in `backend/agents/meta_critic.py`.

**Interface handoff — Shot-List Agent (RR) ↔ Justification Validator (KR):** these are no longer independent sequential nodes; the validator is called *synchronously inside* the Shot-List Agent's own node, between Call A and Call B. Agreed contract (see §5.6 for the full spec): `validate_justifications(justifications: list[dict], winning_script, product_truths, treatment) -> list[ValidationResult]`, one `ValidationResult` per shot (`{shot_id, passed: bool, violation: Optional[str]}`) so RR's re-prompt can name the exact failure type. KR: please build to this signature (or flag here if a different shape is better) so RR isn't blocked — RR will develop/test the Shot-List Agent against a test-fixture `ValidationResult`/`Treatment` in the meantime, same pattern Phase 1 used for independent node testing.

**KR reply — built, three things to know before you build on it (`backend/agents/justification_validator.py`, tests in `backend/tests/test_justification_validator.py`):**
1. **Key name is `shot_id_or_beat_index`, not `shot_id`.** Made it generic on purpose since Treatment Agent's own beats validate through the same function (keyed by `beat_index`) — read `result["shot_id_or_beat_index"]` in your Shot-List Agent, not `result["shot_id"]`, or ping me if you'd rather I rename it back to `shot_id` and have Treatment Agent's caller-side code do its own translation instead.
2. **First-failure-wins, not exhaustive.** Checks run in a fixed order (quote → truth_id → treatment_ref → beat_function → stoplist) and `violation` reports only the *first* one that fails. Confirmed concretely: a justification with both a bad `truth_fact_id` and a stoplist-hit phrase comes back as `unknown_truth_id` only — the stoplist problem is invisible until that first issue is fixed and re-validated. If a shot has two simultaneous problems, your re-prompt loop will need a second round to surface the second one — one-fix-per-round, not both-at-once. Simpler contract, but flagging so it's not a surprise mid-debug.
3. **Verbatim-quote check is scoped to the whole `winning_script.text`, not to the specific beat/shot's claimed line** — and does **not** cross-check that `script_quote` actually belongs to the beat `treatment_ref` points at. Confirmed concretely: a justification with `treatment_ref=0` (the hook beat) but a `script_quote` copied from a totally different beat's line still passes, because the quote-exists-somewhere check and the treatment_ref-exists check run independently with no consistency check between them. Low real-world risk for short, non-repetitive ad scripts, but a shot could in principle cite the right treatment beat while quoting the wrong one and this validator won't catch it. If that's a real risk for Shot-List Agent's Call A output, we should either add a quote-belongs-to-treatment_ref's-own-beat check or agree it's out of scope for now.

**RR reply to KR's notes above (independently arrived at, before seeing KR's push — see the matching analysis in `backend/agents/shot_list_agent.py`'s `_build_call_a_reprompt` docstring):**
1. **`shot_id_or_beat_index` key — no change needed on your side.** Your generic naming is the right call for a function serving both Treatment Agent and Shot-List Agent; RR's `shot_list_agent.py` now reads either `shot_id` or `shot_id_or_beat_index` from a validator result (`_build_call_a_reprompt`), so swapping your real module in for RR's local stand-in needs zero further changes on either side.
2. **First-failure-wins** — RR's own local stand-in independently converged on the identical behavior. Confirms the two implementations read the spec the same way; no action needed.
3. **Quote-doesn't-verify-it's-from-the-cited-beat — shared known limitation, deferred jointly.** RR's own stand-in has the same class of gap (found independently via an adversarial test pass on `shot_list_agent.py`: a quote stitched across two beat lines can validate). Neither side is fixing this now — revisit once real Concept Agent → Treatment Agent → Shot-List Agent output is running end-to-end and we can see whether mismatched quote/`treatment_ref` pairs actually occur in practice, rather than guessing from hand-written fixtures.

### KR
- [x] `[BRAIN]` **Treatment Agent** (`director_persona`, `color_story`, `pacing_philosophy`, `beat_treatments[]` with `script_quote`/`truth_fact_id`/`why_not_generic`; "category" word disallowed in output) — `backend/agents/treatment_agent.py`, tests in `backend/tests/test_treatment_agent.py`. Not yet wired into `graph/build.py` (nothing downstream of `winning_script` is wired yet — same standalone-and-tested state `hook_checker.py` was in before its siblings existed).
- [x] `[BRAIN]` **Justification Validator** (deterministic: verbatim quote check, `truth_fact_id` exists, `treatment_ref` matches, stoplist reject; called synchronously from inside the Shot-List Agent between its two calls, per the interface handoff above — signature: `validate_justifications(...) -> list[ValidationResult]`) — `backend/agents/justification_validator.py`, tests in `backend/tests/test_justification_validator.py`. See the "KR reply" note above for three things to know before building Shot-List Agent on top of it.
- [ ] `[BODY]` Director's treatment panel in dashboard (persona/color story/pacing/per-beat justification)
- [ ] `[BODY]` DB migrations for ledger + treatment + shot-list-with-justification tables

### RR
- [x] `[BRAIN]` **Shot-List Agent** — two-call architecture (Call A: `justification` only per shot, numbered/ID'd source menus in-prompt; Call B: camera/composition fields conditioned on the validated justification), 3-7 shots, camera-literate schema per **C3** (now incl. `rack_focus`/`product_in_hand`); freeze C3 here; confirm no `product_category` field anywhere; anti-genericness rubric + swap-test instruction in Call B's prompt per §5.6 — built against a local stand-in for KR's Justification Validator (injectable, swappable), verified with an independent adversarial test pass (`backend/tests/test_shot_list_agent.py` + `test_shot_list_agent_edge_cases.py`, 68 tests); one confirmed bug found and fixed (lossy Call-A re-prompt was replacing the whole justification list instead of merging by `shot_id`, silently dropping already-valid shots)
- [x] `[BRAIN]` **Budget Gate** (deterministic cap check; if over cap, one **in-process deterministic reduce pass — no LLM re-invocation of the Shot-List Agent** — priority-ordered: downgrade resolution/retry-reserve on lowest-weight shots first, then cut (never merge) the lowest-weight shot with waterfill redistribution, uniform trim last resort, floor case flags a visible overage rather than hiding it; see §5.7 "Reduce is deterministic, not generative" for the reasoning); grounding-weighted allocation formula per §5.7; write allocations to Budget Ledger table; emit `budget_updated` events — `backend/agents/budget_gate.py`, tests in `backend/tests/test_budget_gate.py` + `test_budget_gate_edge_cases.py` (42 tests). Not yet wired into `graph/build.py` (same standalone posture as Shot-List Agent). Verified with an independent adversarial test pass; one confirmed bug found and fixed (two shots sharing a `shot_id` silently corrupted `ledger["spent"]` — `per_shot` collided on the duplicate key and under-reported true spend by a full shot; fixed by deriving `spent` from the returned shots directly rather than from `per_shot.values()`).
- [ ] `[BODY]` Live budget ledger panel (per-shot allocations, running total, cap line, over/under state) + per-shot justification tooltip

**Deferred follow-up (RR, revisit once the Concept Agent → Critic Chain → Treatment Agent → Shot-List Agent chain runs end-to-end, not testable meaningfully in isolation):** the Shot-List Agent currently does NOT re-prompt when Call A returns fewer than `MIN_SHOTS` (3) individually-*valid* shots — it logs a warning and proceeds degraded, unlike `concept_agent.py`'s explicit "too few valid variants -> re-prompt" trigger. Deliberately left as-is for now (see conversation decision) since it's hard to judge whether this actually happens often enough to matter without a real Concept Agent output feeding in, rather than hand-written test fixtures. Once an integration/e2e test suite exists across the real chain, check how often Call A under-delivers in practice and decide then whether to add the count-based re-prompt.

**Exit criteria:** valid 3-7 shot list conforming to frozen C3, budgets sum within cap (loop-back demonstrably works when seeded over budget), every justification passes the validator (re-prompt demonstrably works when seeded generic), ledger/treatment/shot rows visible in DB + dashboard.

---

## Phase 3 — Video-Gen Node + Graceful Degradation
*Goal: generate every shot in parallel with a hard-failure path that never blocks the pipeline.*

### KR
- [x] `[BRAIN]` **Video-Gen Node**: LangGraph parallel fan-out via `Send()`; structured prompt formula (Subject→Action→Camera→Lighting→Composition→Mood→Quality) + `negative_prompt` + reference product photo (image-to-video only, never text-to-video) — `backend/agents/video_gen_node.py`, tests in `backend/tests/test_video_gen_node.py`. Self-contained `Send()` fan-out (own private subgraph, no C1 change needed); budget-clamp policy per `agents/budget_gate.py`'s own documented RATE_720P/RATE_1080P comparison; not yet wired into `graph/build.py` (deliberate follow-up integration step, matching how Phase 2's agents were each merged standalone first — see module docstring).
- [ ] `[BODY]` Dashboard: per-shot generation status grid (queued/generating/done/fallback) with thumbnails

### RR
- [ ] `[BRAIN]` **Ken-Burns Fallback Node** — on hard API failure/timeout, route straight to fallback, **no retry consumed**
  **Interface note (from KR's Video-Gen Node, merged on p3kr):** on shot failure/timeout, Video-Gen Node sets shot status to `fallback_requested` and attaches a `failure_reason: {type, detail}` object; no retry count is touched on this path. This is KR's own assumed shape, not yet confirmed with RR — sync before building the fallback node's input handling, since the exact status string/failure_reason shape may need adjustment once both sides agree.
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
- [ ] Concept Agent + Critic Chain (incl. Body-Checker) produce a scored, cross-pollinated winning script that has passed the independent Merge Coherence Validator — voice/register failures actively repaired by the Copy Editor's constrained seam polish, promise-payoff failures repaired by the Meta-Critic's swap, or a visible fallback — with a visible trace
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
