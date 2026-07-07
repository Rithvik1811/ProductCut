# ProductCut — Technical Documentation

### Global AI Hackathon with Qwen Cloud — Track 2: AI Showrunner

> **Document status.** This is the authoritative technical specification for ProductCut. It supersedes Section 3 (System Architecture) and Section 4 (Agent Roster) of `PROJECT_PROPOSAL.md`. Where the proposal and this document disagree, this document wins. The problem statement and motivation framing carried forward from the proposal remain valid. **This revision** adds two architectural extensions arising from a second research pass: (1) **script-driven, not category-driven, direction** — a Product Truth Extractor, a Treatment Agent, a justification-forced Shot-List schema, and a lightweight seller creative-intake, all aimed at killing "mode collapse" toward generic same-feeling output; and (2) **post-generation chat-based revision** — an Edit Router, Edit Interpreter, and checkpoint-based scoped re-execution that let a seller request a targeted fix ("make the hook punchier," "shot 2 is too dark") without re-running the whole pipeline. Both extensions are additive to the existing DAG; no prior node is removed.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Why This Architecture](#2-why-this-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Agentic Architecture Diagram](#4-agentic-architecture-diagram)
5. [Agent-by-Agent Detail](#5-agent-by-agent-detail)
6. [Shared State Schema](#6-shared-state-schema)
7. [Database Design](#7-database-design)
8. [Error Handling & Graceful Degradation](#8-error-handling--graceful-degradation)
9. [Deployment on Alibaba Cloud](#9-deployment-on-alibaba-cloud)
10. [Judging Criteria Alignment](#10-judging-criteria-alignment)

---

## 1. Project Overview

### 1.1 The Problem

Small Etsy and Shopify sellers cannot afford professional product video ads. A single 15–30 second ad video from a freelance studio costs $300–$1,500 and takes days to turn around. Most sellers default to posting static photos instead, which convert worse than video across every major ad platform. The gap between "has product photos" and "has a scroll-stopping short-form video ad" is exactly the gap that keeps small sellers from competing for paid attention.

**ProductCut** closes that gap. A seller uploads 2–3 product photos and a one-line creative brief — for example, *"handmade ceramic mugs, cozy autumn vibe"* — and receives a finished 15–30 second product ad video, generated end-to-end by a coordinated crew of LLM agents running on Qwen Cloud.

This is deliberately **not** a general-purpose "type an idea, get any video" tool. The scope is intentionally narrow: **short-form product ad video for e-commerce sellers**. Narrow scope is what makes the agent pipeline's decisions — script selection, shot budgeting, visual consistency — meaningfully checkable, defensible, and demoable within a hackathon timeframe. A narrow domain also lets each agent encode real domain expertise (ad copywriting frameworks, camera grammar, pacing rules) instead of generic prompting.

### 1.2 Fit with Track 2 — AI Showrunner

Track 2 asks for an autonomous system that handles the full creative production chain — scriptwriting, storyboarding, video generation, and editing — while demonstrating narrative ability, multimodal orchestration, and quality maximization under a constrained budget. ProductCut maps directly onto every one of those requirements:

| Track 2 requirement | How ProductCut addresses it |
|---|---|
| Autonomously handle scriptwriting → storyboarding → video generation → editing | Product Truth Extractor → Concept Agent → Critic Chain → Treatment Agent → Shot-List Agent → Video-Gen → Continuity → Assembly → Format Export, orchestrated as a single LangGraph graph |
| Demonstrate narrative ability | The Concept Agent produces four structurally distinct scripts, each using a different copywriting framework, hook type, and emotional trigger; the Critic Chain scores and cross-pollinates them into one winning narrative |
| Multimodal orchestration | Text (brief → script) → image (product photos, reference frames) → video (Wan/HappyHorse) → speech (Qwen TTS) → final assembled cut |
| Maximize output quality under a limited budget | An explicit per-shot cost budget is enforced by a deterministic Budget Gate with a hard cap, streamed live to a cost dashboard, with graceful degradation (Ken-Burns fallback) when a shot cannot be generated |
| Agent negotiation / disagreement resolution the track wants judges to see | The Critic Chain runs five parallel specialist checkers whose scores are reconciled by a Meta-Critic that cross-pollinates the best hook, body, and CTA across variants, then an independent Merge Coherence Validator re-checks that merge before it is accepted — a visible, auditable negotiation with a genuine second opinion, not a black box and not a single agent grading its own work |

### 1.3 What the System Produces

For each job, ProductCut outputs a finished short-form product ad in **three aspect ratios** — 9:16 (TikTok / Reels / Shorts), 1:1 (feed), and 16:9 (YouTube) — plus a full transparency breakdown: the four candidate scripts, every critic score, the merge justification and its independent coherence/pacing re-check, the director's treatment, the per-shot justification trace, the per-shot budget ledger, and the continuity drift scores. The transparency breakdown is a first-class deliverable, not a debug artifact: it is what makes the system's autonomous creative decisions legible to a judge (and, in production, to a seller who wants to understand why the system made the choices it made).

### 1.4 Two Extensions: Script-Driven Direction and Chat-Based Revision

A prior research pass identified a failure mode called **"the Price of Format"**: when an LLM agent is given a rigid, templated prompt (e.g., "pick a camera move for this product category"), the template itself acts as a behavioral anchor that collapses entropy — output feels generic regardless of what the actual product or script says. This document's two extensions directly counter that:

- **Script-driven direction (Section 5.2, 5.5, 5.6).** The Shot-List Agent's camera/composition choices must be justified by direct reference to *this job's* specific script text and *this job's* specific product facts — never by a category lookup table. A **Product Truth Extractor** pulls concrete, non-generic facts from the actual photos before any script is written; a **Treatment Agent** writes a short director's-treatment-style justification connecting the winning script's specific beats to a visual approach; the Shot-List Agent's output schema then requires a per-shot `justification` object that must cite exact words from the script and a specific extracted fact, which is mechanically validated.
- **Chat-based revision (Section 5.16).** Once a full ad is generated, the seller can request changes in natural language. Rather than re-running the entire graph, an **Edit Router** classifies the request to the minimal affected stage(s), an **Edit Interpreter** turns a vague instruction ("more energetic") into a specific, grounded parameter delta, and LangGraph's checkpointer **forks execution from the affected node** — re-running only the downstream nodes that actually need to change, with a cheap preview/diff and seller confirmation gating any expensive re-render.

---

## 2. Why This Architecture

### 2.1 The Pipeline Is a Budget-Capped DAG, Not a Conversation

The core architectural decision is the choice of orchestration framework. ProductCut's pipeline is **not** a free-form multi-agent conversation. It is a directed acyclic graph (DAG) with five specific structural properties that the framework must model natively:

1. **A conditional retry loop.** The Continuity Agent can send a shot back to the Video-Gen Node for re-generation, but only up to a hard cap of 2 retries, after which it escalates to human review.
2. **A scoring / merge step with a visible reasoning trace, plus an independent re-check of the merge's own output.** The Critic Chain must produce an auditable record of how four candidate scripts were scored and merged into one, and the merge itself must be re-validated by a node other than the one that built it (Section 5.4.7) before it is accepted — this trace is a demo deliverable, not an implementation detail.
3. **Per-step budget / cost tracking with hard caps.** Every node that spends money (video generation especially) must debit a shared ledger, and a gate must be able to reject a plan that exceeds the cap.
4. **Parallel fan-out.** Multiple shots generate concurrently, then rejoin for assembly.
5. **Realtime progress streaming to a frontend.** The budget ledger, critic reasoning, and drift scores stream live to the dashboard while the graph runs.

To these five, the two extensions in this revision add a sixth and seventh property the framework must also model natively:

6. **Mechanically validated, quote-grounded structured output.** The Shot-List Agent's `justification` field is checked against the actual script text and truth-extraction record before the shot is accepted — a deterministic gate, not a vibe check.
7. **Resumable, forkable execution from an arbitrary interior node.** A chat-based edit must be able to re-enter the graph at Concept, Treatment, Shot-List, or Video-Gen — whichever stage the edit actually concerns — with modified state, and re-run only what's downstream of that node.

### 2.2 Framework Selection: LangGraph

**LangGraph was chosen** over CrewAI, Qwen-Agent, AutoGen/AG2, the OpenAI Agents SDK, LlamaIndex Workflows, and Temporal. It models all seven of the properties above as first-class primitives:

- **Conditional retry loop** → LangGraph **conditional edges**, with the retry cap living in **typed graph state** (`retry_count` per shot). The loop condition is a pure function of state, so it is inspectable and bounded by construction.
- **Fan-out / fan-in** → LangGraph's **`Send()`** primitive dispatches one Video-Gen task per shot in parallel, and the graph rejoins them at the Assembly node.
- **Budget ledger** → a plain field on shared state (`budget_ledger`) that every node updates; the Budget Gate is a deterministic conditional edge reading that field.
- **Human-in-the-loop** → LangGraph's **`interrupt()`** provides a genuine pause/resume, not a dead-end flag. The graph checkpoints, surfaces the flagged shot in the UI, and resumes from the checkpoint once the seller responds. The **same primitive** powers the pre-render "confirm this edit" gate in Section 5.16.
- **Realtime streaming** → LangGraph's **`astream_events`** emits per-node events that a FastAPI WebSocket endpoint forwards to the frontend.
- **Mechanical validation of structured output** → each reasoning node's output schema is a Pydantic model; a post-call deterministic validator function (not an LLM) checks the `justification` fields against the raw script/truth strings before the state update is accepted, re-prompting on failure (see Section 5.6).
- **Checkpoint-based forking** → LangGraph persists a checkpoint after every node via a `checkpointer` (Postgres-backed in production). `get_state_history(thread_id)` retrieves every past checkpoint of a job; `update_state(checkpoint_id, patch)` followed by `graph.invoke(None, config)` **forks a new execution branch from that checkpoint** with the patched state, re-running only nodes downstream of the fork point. This is the exact mechanism the chat-revision subsystem uses to avoid a full re-run.

Critically, **all model calls route through Qwen via DashScope's OpenAI-compatible endpoint**, so Qwen usage is deep and visible in every node's trace *regardless of the orchestration framework*. Choosing LangGraph for orchestration does not dilute the "native Qwen Cloud usage" story for judging — every reasoning, vision, and speech call is a Qwen Cloud call, and the LangGraph trace makes each one visible.

### 2.3 Why Not the Alternatives

- **LlamaIndex Workflows** was the **second choice**. It offers a similar event-driven fit (event-emitting steps map cleanly onto our streaming needs), but its human-in-the-loop story is less turnkey than LangGraph's `interrupt()` + checkpointer, and it has no equivalent to `update_state()`-driven forking, which the chat-revision subsystem depends on.
- **CrewAI, AutoGen/AG2, and Qwen-Agent** were rejected as *primary orchestrators* because they model **conversational, role-based delegation** — agents talking to agents — rather than a budget-capped DAG with parallel fan-out and a conditional, count-limited retry loop. Forcing our pipeline into a conversation abstraction would mean hand-rolling the graph, the budget gate, and the retry cap on top of a framework that fights that shape.
  - **Qwen-Agent specifically lacks a graph/edge primitive**, so building the retry loop and fan-out would mean writing the pipeline control flow by hand. It remains, however, an excellent reference for **tool-calling patterns** if we need them inside any single node — we treat it as a source of component-level patterns, not as the orchestrator.
- **Temporal** is a durable-execution engine that could model the DAG, retries, and even the fork-and-resume pattern, but it is operationally heavyweight for a hackathon and adds a workflow-worker deployment surface we do not need; LangGraph's checkpointer already gives us the durability (resume-after-crash) *and* the fork-from-node capability (chat revision) with far less infrastructure.

### 2.4 Countering Template Collapse — Script-Driven, Not Category-Driven, Direction

A prior research round on this project established that giving the Shot-List Agent a fixed per-product-category lookup table of camera moves ("mugs → macro + orbit; apparel → lifestyle wide") produces *"generic output with nouns swapped"* — the template itself is a behavioral anchor that collapses the model's entropy regardless of what the specific script says. Real commercial directors do not work this way: they write a **director's treatment**, a short document that argues, with reference to the *specific* script, why a specific visual approach fits *this* ad — reference imagery, color story, pacing philosophy, camera/lens choices, and an explicit rationale connecting each choice to the script's narrative beats, not to the product's category.

ProductCut encodes this as two new agents plus one schema change:

- **Product Truth Extractor** (Section 5.2) runs immediately after Ingest, before a single word of script is written. It is a Qwen-VL call that pulls specific, checkable facts from the actual uploaded photos — exact colors, materials, textures, distinguishing details, imperfections — into a `product_truths[]` list. Every fact carries a `truth_id`. This is the raw material that keeps everything downstream specific rather than generic.
- **Treatment Agent** (Section 5.5) runs after the Critic Chain has produced the winning script. It writes a ~250–400 word director's treatment: a persona (e.g. "kinetic, high-contrast, fast-cut director" vs. "quiet, tactile, slow-reveal director"), a color story, a pacing philosophy, and — critically — a one-sentence justification **per script beat**, each justification required to quote the script and cite a `truth_id`.
- **Justification-forced Shot-List schema** (Section 5.6): every shot object gains a `justification` field with `script_quote` and `truth_fact_id` sub-fields. A deterministic validator string-matches `script_quote` against the actual winning script text and `truth_fact_id` against a real entry in `product_truths[]`, rejecting (and forcing a re-prompt of) any shot whose justification cites nothing real or is generic enough to apply to any product. This mirrors research showing that quote-grounded, citation-forced generation measurably reduces hallucinated/generic output in structured LLM tasks.

A fourth, lighter-weight addition lets the **seller** inject their own specific creative direction beyond photos + one-line brief, modeled on how ad agencies run intake/discovery (mood words, reference ads liked/disliked, explicit "don't do this"): an optional, small **Intake** extension to Ingest (Section 5.1) capturing up to 3 mood words, one reference-ad link/description, one "never do this," and one freeform field — all optional, so the product never turns into a long form. This feeds the Product Truth Extractor (freeform text may name facts photos can't show), the Concept Agent (mood words bias framework/tone selection), the Treatment Agent (mood words bias persona selection), and the Critic Chain (the "never do this" becomes a hard rejection rule).

### 2.5 Post-Generation Revision — Scoped Regeneration, Not a Restart

Commercial tools that already ship "chat to edit" on generated media (e.g. Adobe Firefly's "Prompt to Edit," powered by Runway's Aleph model) apply a natural-language instruction as a **targeted patch to an existing clip** — analyzing the existing footage for context and editing in place — rather than regenerating from a blank slate. ProductCut adopts the same philosophy at the pipeline level, not just the single-clip level: a chat message is classified to the **minimal set of graph nodes that must re-run**, and LangGraph's checkpointer forks execution from that node with patched state, leaving every untouched upstream and sibling artifact byte-identical.

Three new components (Section 5.16) implement this:

- **Edit Router** — an LLM classification call that maps a chat message to `{scope, target_shot_ids[], entry_node}` (e.g. a tone complaint → `entry_node = concept`; "shot 2 is too dark" → `entry_node = shot_list`, `target_shot_ids = [shot_2]`; "shorten to 15s" → `entry_node = shot_list`, `scope = pacing`; "change the CTA text" → `entry_node = assembly`, no video-gen at all).
- **Edit Interpreter** — translates a vague instruction into a specific, grounded parameter delta *before* anything re-runs, so "make it more energetic" cannot regress back into generic output. It re-seeds the Treatment Agent's director persona toward a more kinetic profile and produces a concrete patch (e.g., shot durations 2.5s → 1.5s, add `camera_move: pan` on shots 2–3 only) — using the same script/truth-quote grounding discipline as Section 2.4, applied to the delta itself.
- **Preview/Confirm gate** — a LangGraph `interrupt()` fired *before* any Video-Gen re-render, showing the seller a cheap diff (old vs. new script/shot-list text plus an estimated incremental cost) and requiring explicit confirmation before the expensive call fires. This is the same `interrupt()` primitive already used for continuity human-review, applied to a second purpose.

---

## 3. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| **Frontend** | Next.js 14, React, Tailwind | Live dashboard consuming WebSocket events (budget ledger, critic trace, drift scores); post-generation view adds a chat panel and a diff/preview panel for edits |
| **Backend / Orchestration** | FastAPI (Python) + **LangGraph** | LangGraph is the chosen orchestration framework — see Section 2 for the full justification |
| **Orchestration durability** | LangGraph **Postgres checkpointer** (`langgraph-checkpoint-postgres`) | Persists a checkpoint after every node on the same managed Postgres instance as application data; this is what makes both crash-resume and chat-edit forking work without extra infrastructure |
| **LLM / Reasoning** | Qwen (Qwen-Max / Qwen-Plus) via DashScope OpenAI-compatible endpoint | Product Truth Extractor's text-side reasoning, Concept Agent, Critic Chain checkers, Meta-Critic, Merge Coherence Validator, Copy Editor, Treatment Agent, Shot-List Agent, Edit Router, Edit Interpreter |
| **Vision** | Qwen-VL via DashScope | Product Truth Extractor (photo → specific facts), Continuity Agent (product-identity drift + cross-shot style-consistency detection) |
| **Speech** | Qwen TTS / CosyVoice via DashScope | Voiceover synthesis + caption timing |
| **Video Generation** | Wan / HappyHorse (Tongyi Wanxiang / Alibaba video model family) | **Image-to-video only, never pure text-to-video** |
| **Video Assembly** | ffmpeg | Deterministic; not an LLM call |
| **Database** | Alibaba Cloud managed Postgres-compatible DB (PolarDB or RDS for PostgreSQL) | Job state, budget ledger, script/shot records, product truths, treatments, seller direction, chat/edit history, and the LangGraph checkpoint tables — all in one instance |
| **Storage** | Alibaba Cloud Object Storage (OSS) | Product photos, generated shot clips, final videos, cached demo assets |
| **Deployment** | Alibaba Cloud ECS (default) or Function Compute | Backend must run here for the "Proof of Deployment" requirement |
| **Realtime** | WebSocket from FastAPI, fed by LangGraph's `astream_events` | Live dashboard: budget ledger, critic reasoning trace, drift scores, edit-router decisions, preview diffs |

**Model-routing note.** Every LLM, vision, and speech call — across all nodes, including the two new agents and the chat-revision subsystem — goes through DashScope's OpenAI-compatible API surface. This keeps the client code uniform (one SDK, one auth path), and it means the Qwen Cloud dependency is exercised on essentially every intelligent step in the pipeline, satisfying the hackathon's requirement that the Alibaba Cloud service be *actually called*, not merely referenced.

**No new infrastructure required for either extension.** The Product Truth Extractor and Treatment Agent are additional Qwen/Qwen-VL calls on the existing graph; the chat-revision subsystem reuses the existing FastAPI + LangGraph + Postgres + OSS stack — the only addition is the `langgraph-checkpoint-postgres` package and three new tables (Section 7).

---

## 4. Agentic Architecture Diagram

```mermaid
flowchart TD
    U[Seller: photos + brief + optional intake] --> ING[Ingest Node - validate, upload to OSS, capture SellerDirection]
    ING --> PTE[Product Truth Extractor - Qwen-VL<br/>specific facts, colors, materials]
    PTE --> CA[Concept Agent - Qwen-Max<br/>4 scripts, forced distinct frameworks/hooks<br/>persona-seeded from truths + mood words]

    CA --> HC[Hook-Checker]
    CA --> PC[Pacing-Checker - deterministic timing math]
    CA --> BC[Body-Checker - completion / structural fit]
    CA --> CC[CTA-Checker]
    CA --> TC[Tone-Checker]
    HC --> MC[Meta-Critic - weighted aggregate + cross-pollinate merge]
    PC --> MC
    BC --> MC
    CC --> MC
    TC --> MC

    MC -->|merge candidate| CV{Merge Coherence Validator - independent re-check}
    CV -->|voice/register fail, attempts<1| CE[Copy Editor - constrained seam polish]
    CE --> CV
    CV -->|promise-payoff fail, attempts<1| MC
    CV -->|fails again either path| FBV[Fallback: highest single composite variant]
    CV -->|pass| WIN[winning_script finalized]
    FBV --> WIN

    WIN -->|winning script + reasoning trace| TA[Treatment Agent - Qwen<br/>director persona, color story,<br/>per-beat justification vs script+truths]
    TA --> SL[Shot-List Agent - Qwen<br/>camera-literate schema<br/>+ required per-shot justification]
    SL --> JV{Justification Validator}
    JV -->|quote/fact not found, re-prompt| SL
    JV -->|valid| BG{Budget Gate}
    BG -->|over cap, 1 retry| SL
    BG -->|ok| BL[(Budget Ledger)]
    BG --> VGFO[Fan-out per shot - Send()]

    VGFO --> VG[Video-Gen Node<br/>image-to-video, ref photo required]
    VG -->|API failure| FB[Fallback: Ken-Burns pan]
    VG --> CTY[Continuity Agent - Qwen-VL<br/>drift + style-consistency check]
    CTY -->|drift, retries<2| VG
    CTY -->|drift, retries exhausted| HR[interrupt: Human Review in UI]
    HR -->|resume| VG
    HR -->|accept fallback| FB
    CTY -->|pass| ASM
    FB --> ASM[Assembly Agent - ffmpeg]

    WIN --> VOX[Voiceover + Caption Agent - Qwen TTS<br/>synced to beat timestamps]
    VOX --> ASM

    ASM --> FMT[Format Export Node - ffmpeg<br/>9:16 / 1:1 / 16:9]
    FMT --> OUT[Final outputs to OSS]
    OUT --> FE[Frontend]

    FE -->|seller chat message| ER[Edit Router - Qwen<br/>classify scope + entry node]
    ER --> EI[Edit Interpreter - Qwen<br/>vague ask -> grounded delta]
    EI --> PV[interrupt: Preview/Confirm diff + cost]
    PV -->|confirm| FORK[Checkpointer fork -<br/>resume from entry node w/ patched state]
    PV -->|reject/refine| FE
    FORK -.->|re-enters graph at CA, TA, SL, or ASM| CA

    BL -.->|live ledger stream| FE
    MC -.->|reasoning trace stream| FE
    CV -.->|merge validation trace stream| FE
    CE -.->|seam before/after stream| FE
    CTY -.->|drift scores stream| FE
    ER -.->|routing decision stream| FE

    subgraph Alibaba Cloud
        BL
        OSS[(Object Storage)]
        DB[(Job/State DB + LangGraph checkpoints)]
    end
    OUT --> OSS
```

**Reading the diagram.** Solid arrows are graph edges (control/data flow). Dashed arrows are live streaming channels to the frontend via `astream_events` → WebSocket. The `Budget Gate`, `Justification Validator`, and **`Merge Coherence Validator`** (diamonds) and `interrupt: Human Review` / `interrupt: Preview/Confirm` (`HR`, `PV`) are the decision points that can loop, pause, or fork the graph. **The Merge Coherence Validator (`CV`) is a distinct node from the Meta-Critic (`MC`)** — it receives the Meta-Critic's merge candidate but is never the same call/context that produced it, which is what makes its pass/fail judgment an independent check rather than the merge-writer grading its own work. On failure, `CV` routes by **which** sub-check flagged: a voice/register seam failure loops to the **Copy Editor (`CE`)**, a distinct constrained-repair node (Section 5.4.8) that polishes only the flagged transition and hands the patched merge straight back to `CV` for re-validation; a promise-payoff failure loops to `MC` for one bounded retry (the existing swap-to-second-best-piece behavior), also returning to `CV`. Either path falls back to the single highest composite-scoring variant (`FBV`) on a second failure, and only the node downstream of that decision (`WIN`) is treated as the actual finalized script. The Voiceover branch (`WIN → VOX → ASM`) runs in parallel with the entire video-generation branch, because voiceover depends only on the finalized, validated script, not on the rendered shots. The **chat-revision loop** (`FE → ER → EI → PV → FORK`) is drawn re-entering at the Concept Agent, but `FORK`'s actual re-entry point is whatever `entry_node` the Edit Router chose — Concept, Treatment, Shot-List, or Assembly — never the whole graph from Ingest. Everything inside the `Alibaba Cloud` subgraph is a managed cloud resource (Budget Ledger table, Object Storage, Job/State DB, and now the LangGraph checkpoint tables that live in the same Postgres instance).

---

## 5. Agent-by-Agent Detail

The pipeline is a mix of **reasoning nodes** (Qwen / Qwen-VL / Qwen TTS calls) and **deterministic nodes** (pure code — validation, timing math, budget arithmetic, ffmpeg). Deterministic nodes are deliberately *not* LLM calls: they are the parts of the pipeline where correctness must be guaranteed, not sampled. Each node below is documented with its purpose, model/API, input/output contract, prompt-design strategy (where applicable), and failure/edge-case handling.

### 5.1 Ingest Node — *deterministic, with an optional structured intake*

**Purpose.** The pipeline's front door. Validates that the seller supplied 2–3 product photos and exactly one one-line creative brief, captures an **optional lightweight creative-intake**, then establishes all the durable state the rest of the graph relies on.

**Model / API.** None — this is pure code. No LLM call.

**Inputs.** Raw upload payload: 2–3 image files, a free-text brief string, and an optional `SellerDirection` object: up to 3 mood words, one reference-ad link or description ("I like the energy of X"), one explicit "never do this," and one freeform text field. All four intake fields are optional — the product never requires more than photos + one line to function; the intake exists purely to let a seller who *wants* to say more do so cheaply, mirroring how ad agencies run a lightweight discovery pass (mood words, liked/disliked references, explicit exclusions) without turning it into a long-form brief.

**Outputs.** Photos persisted to OSS (returning stable object URIs); a new `jobs` row in the database with status `ingested`; a `seller_direction` row if any intake field was filled; and an initialized LangGraph state object carrying `job_id`, `brief`, `product_photos[]` (the OSS URIs), and `seller_direction` (nullable).

**Validation / edge cases.** Rejects submissions with fewer than 2 or more than 3 photos, empty or missing briefs, unsupported image formats, or files exceeding a size ceiling. Validation failures are returned to the frontend as actionable errors *before* any spend occurs — the Ingest Node is the cheap gate that prevents malformed jobs from consuming any Qwen or video-gen budget. Because it writes the OSS URIs and the job record up front, every downstream node has a stable reference store for the source product photos (which the Product Truth Extractor, Video-Gen, and Continuity nodes depend on).

### 5.2 Product Truth Extractor — Qwen-VL via DashScope *(new)*

**Purpose.** Kill genericness at the source. Before a single word of script exists, this agent looks at the *actual* uploaded photos and extracts specific, checkable, non-generic facts — the raw material that keeps every downstream agent grounded in *this* product rather than falling back to category-level assumptions ("mugs are cozy and handmade-feeling").

**Model / API.** Qwen-VL via DashScope, given all 2–3 product photos plus the brief and any `seller_direction.freeform` text (which may name facts a photo can't show, e.g. "the wax finish smells like cedar").

**Output contract.** A `product_truths[]` array:
```json
{
  "truth_id": "t1",
  "fact": "hand-stitched leather cord wrap at the handle base, visibly irregular stitch spacing",
  "category": "material_detail",
  "source": "photo_2"
}
```
`category` is one of `color | material | texture | construction_detail | imperfection | scale_cue | brief_or_intake_fact`. The agent is instructed to produce **6–10 facts minimum**, each specific enough that it would be **false or absent for a different, similar product** (a generic fact like "the mug is ceramic" fails this bar; "a thin hairline glaze crack near the rim, left unrepaired as an intentional wabi-sabi detail" passes).

**Failure handling.** If the model returns fewer than 4 facts, or facts that are generic/reusable across arbitrary products (checked by a cheap heuristic: rejecting facts under ~6 words or matching a stoplist of generic adjectives), the node re-prompts once with the specific violation named. On persistent failure it proceeds with whatever facts passed the bar, flagged in the reasoning trace — the Concept and Treatment Agents downstream degrade gracefully to fewer, still-real anchors rather than blocking the job.

### 5.3 Concept Agent — Qwen-Max via DashScope

**Purpose.** Turn a one-line brief, product photos, and now `product_truths[]` plus optional `seller_direction` into **four structurally distinct** short-form ad scripts. Distinctness is the point: four near-duplicate scripts give the Critic Chain nothing to negotiate over.

**Model / API.** Qwen-Max via DashScope's OpenAI-compatible endpoint (the strongest reasoning model in the family, used here because script quality anchors the entire downstream pipeline).

**Input contract.** `brief` (string), `product_photos[]` (OSS URIs), `product_truths[]` (from Section 5.2), `seller_direction` (nullable — mood words, reference ad, never-do, freeform), and the target ad length (15s or 30s).

**Output contract.** A JSON array of four objects, each:
```json
{
  "variant_id": "v1",
  "text": "full script text",
  "framework": "PAS",
  "hook_type": "curiosity_gap",
  "emotional_trigger": "recognition",
  "grounding_truth_ids": ["t1", "t4"],
  "beats": [{ "t_start": 0.0, "t_end": 2.5, "line": "..." }],
  "target_length_sec": 15
}
```

**Prompt-design strategy — enforced diversity, not temperature roulette.** Variety is guaranteed by *explicit constraint in the prompt*, not by nudging temperature and hoping. Each of the four scripts is required to use:
- **A different copywriting framework** — Hook-Problem-Product-CTA, **PAS** (Problem-Agitate-Solution), **AIDA** (Attention-Interest-Desire-Action), or **BAB** (Before-After-Bridge).
- **A different hook type** — drawn from: pattern interrupt, bold claim, curiosity gap, direct address, contrarian / myth-busting, social proof, POV, before/after, price anchor, FOMO / urgency, how-to.
- **A distinct emotional trigger** — curiosity, recognition, FOMO, tribal identity, transformation / aspiration, or relief.

Each script must additionally contain: a **single named pain point** (not a vague benefit), a **hook line of ≤10 words**, **beat-level timestamps**, **exactly one CTA verb**, and — new — must cite at least **2 `truth_id`s** it is built around (`grounding_truth_ids`), so a script cannot be pure category-level boilerplate. If `seller_direction` is present, mood words bias framework/tone selection and the "never do this" is a hard constraint checked at the prompt level. The beat timing obeys an explicit pacing rule: a new visual beat every **2–3 seconds for the first 2–3 beats**, then **3–5 seconds** thereafter — so a 15s ad resolves to **5–7 shots** and a 30s ad to **8–12 shots**.

**Failure handling.** If the model returns fewer than four variants, malformed JSON, duplicated frameworks/hooks, or fewer than 2 `grounding_truth_ids` per variant, the node re-prompts once with the specific violation called out. Persistent malformation degrades to the best N valid variants (minimum 2) so the Critic Chain still has something to compare; a single valid variant short-circuits the critic negotiation and proceeds with that variant flagged as un-negotiated in the reasoning trace.

### 5.4 Critic Chain — 5 parallel specialist checkers + 1 aggregator + 1 independent post-merge validator + 1 constrained repair node

This is the pipeline's **agent-negotiation / disagreement-resolution** centerpiece — the component the Track 2 brief explicitly wants judges to *see*. Rather than one monolithic critic, five specialists score the four candidate scripts along orthogonal axes in parallel, and a Meta-Critic reconciles them into a cross-pollinated merge candidate. That candidate is then re-checked by a sixth component — the Merge Coherence Validator — that is architecturally separate from the Meta-Critic, so the one agent that builds the merge is never the only one that judges it. When the validator flags a voice/register seam problem specifically, a seventh component — the Copy Editor (Section 5.4.8) — performs a constrained polish of that seam, in the same repair role a professional copy editor plays on a stitched piece, before the merge goes back through the validator. The output is an auditable scoring trace, a merge justification, and an independent pass/fail record, never a black-box pick.

**Why a fifth checker and a sixth validator, not just "trust the Meta-Critic."** Two gaps in the original four-checker design turned out to matter in practice. First, Hook-Checker scores the opening line and CTA-Checker scores the close, but nothing scored the beats in between — the part of the script actually responsible for paying off the hook's promise, escalating (not repeating) the case for the product, and landing the script's declared `emotional_trigger`. A script can have a 5/5 hook and a 5/5 CTA while its middle beats say the same thing three ways or never deliver on what the hook promised, and the composite score would never reflect that. Second, the cross-pollination merge stitches pieces from three *independently written* scripts — each deliberately built in a different framework, hook type, and emotional trigger, per Section 5.3's enforced-diversity rule — and the only thing checking whether that stitch reads as one coherent voice was the Meta-Critic itself, immediately after building it. Real editorial practice does not let the editor who made a cut be the only person who signs off on it; a second, independent read — often specifically a fresh, cold read by someone uninvolved in the edit — is what catches voice seams and unpaid promises that the person who just assembled the piece is primed to overlook. Sections 5.4.3 and 5.4.7 below close these two gaps.

#### 5.4.1 Hook-Checker — Qwen

**Purpose.** Score each script's hook for **specificity and strength**, 1–5, with written justification. A strong hook names a pain, cites a number, or makes a contrarian claim; a weak hook is generic. Concretely: *"Check out this amazing mug"* scores low; *"Your coffee is cold in 12 minutes. Mine isn't."* scores high. **Input:** the four scripts' hook lines (in context). **Output:** `{variant_id: {hook_score, justification}}`. **Prompt strategy:** the rubric ships example-anchored (weak vs. strong exemplars) so scores are calibrated rather than arbitrary.

#### 5.4.2 Pacing-Checker — *deterministic code, not an LLM call*

**Purpose.** Validate timing math with guaranteed correctness. It confirms the beat timestamps **sum to the target length**, that **each beat is within the 2–3s / 3–5s pacing window** per the rule, and that each **voiceover line fits its beat duration** at a spoken rate of **~2.3 words/second**. Because timing correctness is arithmetic, not judgment, it is code — an LLM would be a strictly worse choice here. **Output:** `{variant_id: {pacing_score, violations[]}}`, where violations name the exact offending beat and metric.

#### 5.4.3 Body-Checker — Qwen, with a deterministic redundancy pre-pass *(new)*

**Purpose.** Score whether the script's **body** — the beats between the hook and the CTA — is actually doing its job, rather than being invisible to the Critic Chain the way it currently is. Concretely, it checks four things a creative director would read the middle of a script for: **(1) promise-payoff match** — does the body substantively develop the specific claim or pain the hook named, rather than just restating it in different words; **(2) non-redundancy** — does each beat add new information/proof, or does it repeat a beat that already made the same point; **(3) throughline** — is there one problem and one product promise carried consistently, or does a competing claim sneak in partway through; **(4) emotional-trigger fidelity** — does the body's content plausibly land the script's declared `emotional_trigger` (curiosity, recognition, FOMO, tribal identity, transformation/aspiration, relief), or is the trigger just a label with nothing in the beats actually earning it. Concretely: a **weak** body pays off "Your coffee is cold in 12 minutes. Mine isn't." with a beat that just re-asserts the claim ("Mine stays warmer, way longer") — no new proof, no escalation. A **strong** body earns it with a beat that supplies the specific mechanism behind the claim, tied to an actual product truth ("Mine isn't — because of the double-wall vacuum seal (t3), not a thicker wall").

**Model / API.** Qwen, scoring call (not generation).

**Input.** For each variant: `beats[1:-1]` (every beat except the hook, `beats[0]`, and the CTA, the last beat — the same hook/CTA convention already implicit in the Hook-Checker's and CTA-Checker's inputs, so no new schema field is required), plus `framework`, `emotional_trigger`, `grounding_truth_ids`, and `product_truths[]` for context on what a body beat's claims should be traceable to.

**Deterministic pre-pass (the part that must be guaranteed, not judged).** Before the LLM call, a pure-code step computes pairwise lexical/semantic overlap between every pair of body beat lines (cheap token-overlap ratio, or a single embedding-similarity pass) and flags any pair above a fixed threshold as a candidate redundant pair. This mirrors the Pacing-Checker's philosophy: *literal* repetition is mechanically detectable and should never be left to LLM sampling variance. The flagged pairs are passed into the LLM prompt as evidence the checker must weigh in on — it does not have to rediscover obvious repetition from scratch, and it is not allowed to silently overrule an unambiguous overlap flag without saying why.

**Output contract.** `{variant_id: {completion_score (1-5), redundant_beat_pairs: [[i, j]], promise_payoff_match: bool, emotional_trigger_landed: bool, justification}}`. This score is what fills the previously-undefined "Completion / structural fit" axis in the Meta-Critic's weighted composite below, and — just as importantly — it is what gives the cross-pollination merge an actual **"best body"** signal to select on, closing the gap where the merge picked a best body with nothing having scored bodies at all.

**Side benefit, not a substitute for real grounding checks.** Because this checker is given `product_truths[]` for context, it can flag a body beat whose "proof" doesn't trace to any cited truth — a partial, incidental check against ungrounded claims in the script's middle. This does **not** turn the Critic Chain into a fact-checker (the hook and CTA remain unverified against `product_truths[]`), so it should not be read as closing that broader gap — only as a modest side effect of giving one checker truth-context it didn't have before.

#### 5.4.4 CTA-Checker — Qwen

**Purpose.** Score **call-to-action clarity**. A good CTA is a single concrete verb plus a destination ("Tap to shop the autumn set"); a bad CTA is vague, missing, or competes with a second CTA. **Input:** each script's CTA line and surrounding closing beats. **Output:** `{variant_id: {cta_score, justification}}`. The checker specifically penalizes multiple competing CTAs, since split calls-to-action measurably depress conversion.

#### 5.4.5 Tone-Checker — Qwen

**Purpose.** Score **brand / tone fit against the seller's one-line brief and any `seller_direction`**. If the brief says *"cozy autumn vibe"* or the intake names mood words like *"quiet, tactile"*, a hard-sell, high-urgency script scores lower on tone even if its hook is strong. If `seller_direction.never_do` is set, the checker also hard-fails any script that violates it. **Input:** the brief, `seller_direction`, and each full script. **Output:** `{variant_id: {tone_score, justification, never_do_violation: bool}}`. This is the axis that keeps the merged result faithful to the seller's stated intent rather than optimizing purely for aggression.

#### 5.4.6 Meta-Critic — Qwen

**Purpose.** Aggregate and, critically, **cross-pollinate**. The Meta-Critic computes a **weighted composite** score per variant:

| Axis | Weight | Checker |
|---|---|---|
| Hook | 25% | Hook-Checker (5.4.1) |
| Pacing | 20% | Pacing-Checker (5.4.2) |
| Completion / structural fit | 20% | Body-Checker (5.4.3) |
| CTA | 20% | CTA-Checker (5.4.4) |
| Tone | 15% | Tone-Checker (5.4.5) |

Every axis in this table now has a checker backing it — the "Completion / structural fit" row previously had no defined source and was the least-anchored 20% of the composite; it is now the Body-Checker's `completion_score`.

Any variant with `never_do_violation = true` is excluded from consideration before weighting, regardless of composite score. Then — the **advanced feature** — instead of picking a single variant wholesale, it **merges the best-scoring hook + best-scoring body (per the Body-Checker's `completion_score`) + best-scoring CTA across all four variants into one winning script**, re-deriving contiguous beat timestamps for the merged sequence so it still targets the correct `target_length_sec` (borrowed beats arrive with three different original timelines, which do not line up on their own). This mirrors how professional ad teams A/B-test hooks independently of body copy: the strongest opening might live in variant 2 while the strongest close lives in variant 4, and a wholesale pick would throw away one of them.

**Output contract.** A **merge candidate**: the stitched script, which axis-winner came from which variant, and a merge rationale. **Critically, the Meta-Critic no longer has the final word on whether its own merge is coherent** — that self-grading was the weak link in the original design (the node that just built the merge is not a reliable judge of whether it reads as one voice). Instead, the merge candidate is handed to the **Merge Coherence Validator (5.4.7)**, a separate node, for an independent pass/fail check before anything downstream treats it as the winning script. This trace streams live to the frontend (dashed edge `MC -.-> FE` in the diagram) so the negotiation — including any retries or fallback the validator triggers — is visible in the demo, not hidden.

#### 5.4.7 Merge Coherence Validator — independent post-merge check *(new)*

**Purpose.** Close the second gap in the original design: nothing re-validated the merged script's timing math (the merge bypassed the Pacing-Checker entirely), and nothing coherence-checked the stitch *except* the agent that had just built it. This node runs after the Meta-Critic and is deliberately **not** the Meta-Critic — it receives the merge candidate but shares no call/context with the reasoning that produced it, which is what makes its judgment an actual second opinion rather than the same model re-confirming its own work in a different sentence.

**Two sub-checks, run in order:**

1. **Deterministic pacing re-check (cheap, runs first).** Re-runs the exact Pacing-Checker logic from 5.4.2 against the merged script's re-derived beat timestamps: do they sum to `target_length_sec`, does each beat fit its 2–3s/3–5s window, does each voiceover line fit its beat at ~2.3 words/second. This is arithmetic, not judgment, so it is code, exactly like 5.4.2. It is not redundant paperwork — beats borrowed from three variants with three different original timelines are exactly the kind of thing that plausibly fails re-assembled arithmetic even when every source variant individually passed.
2. **Independent LLM coherence read (only if the pacing re-check passes).** A **blind, cold read**: the model is given only the merged script's text and beats — **not** the merge rationale, **not** which piece came from which variant, **not** the Meta-Critic's own reasoning — and scores it against a checklist deliberately different in kind from the axis rubrics used to build the merge: **voice/POV consistency** across the stitch points, **promise-payoff match** (does the borrowed body actually pay off the borrowed hook's specific claim), and **register/transition smoothness** at the two seams (hook→body, body→CTA), flagging the beat index of any jarring shift. This mirrors the real editorial practice of a second, uninvolved reviewer reading a stitched piece cold — often specifically reading it as if hearing it for the first time — rather than the person who made the cut re-reading their own work.

**Model / API.** Qwen-Plus — a scoring/classification task, not open-ended generation, and deliberately a separate call from the Meta-Critic's Qwen-Max merge call (cheaper, and structurally independent — not just a different prompt in the same context).

**Output contract.** `{passed: bool, pacing_recheck: {passed, violations[]}, coherence_score (1-5), voice_consistency: bool, promise_payoff_match: bool, register_shift_flags: [beat_index], justification}`.

**Failure handling — bounded retry, then the existing fallback, never a self-grade.** This follows the same "deterministic check + capped retry, then deterministic fallback" shape as every other loop in the system (Budget Gate, Justification Validator, Continuity). The one retry slot below is now **routed by failure type**, not a single repair path:
- **Pacing re-check fails** → one deterministic repair attempt (proportionally re-time the merged beats to fit the pacing windows and target length, the same "reduce to fit" spirit as the Budget Gate's one loop-back) → re-check once. If it still fails, treat it as a merge failure and fall through to the retry/fallback path below.
- **Coherence read fails on voice/register** (a `register_shift_flags` entry or `voice_consistency = false`, with `promise_payoff_match = true`) → **one retry, routed to the Copy Editor (Section 5.4.8)**, which performs a constrained polish of only the flagged seam, then the patched merge returns to this same validator for a full re-check. This is a "how it's said" failure, and a copy-edit repairs it directly at the seam rather than re-rolling to a differently-voiced alternative piece.
- **Coherence read fails on promise-payoff** (`promise_payoff_match = false`, regardless of `register_shift_flags`) → **one retry**, routed back to the Meta-Critic with the specific flagged clash named (e.g., "body from v1 does not pay off the hook from v2's specific claim") so the retry swaps in the *second*-best-scoring piece for whichever axis was flagged. This is a "what is being said" failure — missing content or mechanism — which a copy-edit cannot manufacture without inventing new claims, so the swap remains the right repair here.
- **Second failure of either sub-check** (including a failed Copy Editor re-validation, or a failed swap re-validation) → falls back to the **single highest composite-scoring variant** (already fully valid — it individually passed every checker in its original, unmerged form, so no re-validation is needed) exactly as the original design's fallback did, except the decision to fall back is now made by the independent validator, not by the Meta-Critic (or the Copy Editor) marking its own homework.
- Every attempt (merge candidate, pacing re-check result, coherence read result, which repair path was taken, the repair's before/after where applicable, retry-or-fallback decision) is appended to the reasoning trace, so a demo run that hits a retry or a fallback is not a hidden failure — it is a visible, auditable instance of exactly the "agent disagreement resolution" the track wants judges to see, arguably a *better* demo moment than a merge that silently succeeds.

Only once the Merge Coherence Validator returns `passed: true` (on the first attempt, after a repaired retry, or via fallback) is the script written to `winning_script` and released downstream to the Treatment Agent and the Voiceover branch.

#### 5.4.8 Copy Editor — Qwen-Plus, constrained seam repair *(new)*

**Purpose.** A cross-pollinated merge stitches pieces that were each written for a *different* script — different framework, different hook type, different emotional trigger (Section 5.3's enforced-diversity rule guarantees this). When the Merge Coherence Validator's blind coherence read flags a **voice/register-consistency** problem at one of the two stitch points, the honest repair is not "try a different pre-written piece that also wasn't written to fit this hook" — it is what a professional copy editor actually does to a stitched piece: **polish the seam itself**, in place, without touching the substance. The Copy Editor is that node. It is deliberately **not** the Concept Agent: the Concept Agent's job is free generation of maximally distinct scripts from scratch, which is the wrong tool for a small, constrained, in-place edit — asking the writer to "revise" would reopen the door to the same open-ended generation the rest of the merge path is designed to avoid, and its output would carry none of the guarantees (grounding, non-redundancy, pacing) that got the original piece chosen in the first place. A copy editor, by professional convention, touches prose and transitions; they do not rewrite content or introduce new claims — which is exactly the scope this node is held to.

**Concretely:** a **weak** repair would be re-writing the whole borrowed body paragraph in the hook's voice (this is a rewrite, not a copy-edit, and it invalidates the Body-Checker's score on that piece). A **strong** repair takes a seam like a clipped, direct-address hook ("Your coffee is cold in 12 minutes. Mine isn't.") stitched to a body written in a warmer, third-person register ("Many people find their drinks losing warmth too quickly...") and smooths only the transition clause so the body's opening line picks up the hook's direct address ("Yours does too — here's why mine doesn't:") while every downstream claim, fact, and beat is left untouched.

**Input contract.** The merged script's full text and beats, the specific seam(s) flagged by `register_shift_flags` (beat index of the jarring shift), and the `grounding_truth_ids` + CTA verb that must survive the edit untouched.

**Output contract.** `{merged_script_patched, seams_edited: [beat_index], original_seam_text, revised_seam_text, justification}` — a **before/after** record of exactly what changed, so the repair is auditable rather than an opaque re-write.

**Constraints (enforced in the prompt, then checked deterministically before re-validation):**
- May modify **only** the transition text at the flagged seam(s) (the last line of one borrowed piece and/or the first line of the next) — not the full body, not the hook, not the CTA line itself.
- Must **preserve every factual claim and every cited `grounding_truth_ids`** already present in the merged script — no new claims, no dropped truths.
- Must **preserve the single CTA verb** — the Copy Editor never touches CTA content, only (if flagged) the body→CTA transition phrasing.
- Must stay within roughly **±10% of the original seam's word count**, so the re-derived beat timestamps from the merge (Section 5.4.6) still hold without a further pacing re-derivation.

**Model / API.** Qwen-Plus, one call — scoped/constrained generation, not open-ended rewriting (the same reasoning that puts the Merge Coherence Validator's coherence read on Qwen-Plus rather than Qwen-Max applies here: this is a bounded, checklist-shaped task, not the kind of open-ended creative generation Qwen-Max is reserved for in the Concept Agent).

**Routing — which failure goes to the Copy Editor vs. the existing swap.** The Merge Coherence Validator's failure handling (Section 5.4.7) now branches on **which** part of the coherence read failed, because the two failure types call for genuinely different repairs:
- **`register_shift_flags` non-empty / `voice_consistency = false`** (a voice/register seam problem) → **Copy Editor.** This is a "how it's said" problem at a specific, localized seam, which is precisely what a polish pass fixes — and swapping in a second-best piece doesn't reliably fix it, since that piece was written for yet another script's voice.
- **`promise_payoff_match = false`** (the borrowed body doesn't substantively pay off the borrowed hook's specific claim) → **existing swap-to-second-best-scoring-piece behavior, unchanged.** This is a "what is being said" problem — missing content or a missing mechanism — which a copy-edit cannot manufacture without inventing new claims, which is out of scope by design. A different body genuinely might pay off the hook where a polished version of the *same* body still would not, so the Meta-Critic swap (Section 5.4.7) remains the right tool here.
- If **both** conditions are flagged on the same merge candidate, the swap path takes precedence (fixing the content gap first), since a copy-edit on a seam whose underlying body is about to be replaced is wasted work.

**After the Copy Editor runs.** The patched merge is sent back through the **same Merge Coherence Validator** for one re-check (both sub-checks: pacing re-check, then the blind coherence read again) before it is accepted — the Copy Editor never marks its own work as sufficient, exactly the same "never a self-grade" principle that governs every other check in this pipeline. This re-check consumes the **same single retry slot** already defined in Section 5.4.7's failure handling — the Copy Editor path does not add a new loop or a new cap; it defines *what happens during* that existing retry when the flagged failure is a voice/register seam. If the re-validation still fails, that is the "second failure" already defined in 5.4.7, and the pipeline falls back to the single highest composite-scoring original variant exactly as before.

### 5.5 Treatment Agent — Qwen via DashScope *(new)*

**Purpose.** This is the agent-design answer to "how does a director translate a specific script into a specific shooting approach, instead of a category into a generic style guide." Modeled directly on a commercial director's treatment document, it produces a compact, script-anchored creative argument that the Shot-List Agent must build from — and every claim in it must be traceable to the actual script and actual product facts, not to genre convention.

**Model / API.** Qwen (Qwen-Plus) via DashScope.

**Input contract.** `winning_script` (with beats), `product_truths[]`, `seller_direction` (nullable).

**Output contract.**
```json
{
  "director_persona": "quiet, tactile, slow-reveal — favors stillness and macro detail over movement",
  "color_story": "warm neutrals, single accent of oxidized copper matching truth t3",
  "pacing_philosophy": "let the hook breathe for a full 3s before the first cut; no beat shorter than 2s",
  "beat_treatments": [
    {
      "beat_index": 0,
      "beat_function": "hook",
      "script_quote": "Your coffee is cold in 12 minutes. Mine isn't.",
      "truth_fact_id": "t3",
      "visual_approach": "static macro on the double-wall seam (t3) held through the line, no camera movement — the stillness IS the claim",
      "why_not_generic": "a push-in or orbit here would compete with the claim instead of proving it"
    }
  ]
}
```

**Prompt-design strategy — mapping beat function to shot grammar, not category to shot grammar.** The agent is explicitly instructed to reason from each beat's **narrative/emotional function** (hook = shock/tension, problem = discomfort, demo/proof = trust/clarity, CTA = urgency/clarity) to a camera treatment, and is **forbidden from naming the product category** as a justification anywhere in its output — a lightweight prompt-level guard (the word "category" and category names are disallowed in `why_not_generic`/`visual_approach`) that structurally blocks the laziest form of genericness. Every `beat_treatments[]` entry must include a verbatim `script_quote` and a real `truth_fact_id`.

**Failure handling.** If a beat treatment's `script_quote` cannot be found verbatim in `winning_script.text`, or `truth_fact_id` does not exist in `product_truths[]`, the node re-prompts once for that beat specifically (not the whole document). Persistent failure falls back to the single most literal, lowest-risk treatment for that beat (static framing, shared lighting only) rather than blocking the pipeline, flagged in the trace.

### 5.6 Shot-List Agent — Qwen

**Purpose.** Convert the winning script **and the Treatment Agent's document** into a concrete, **camera-literate** shot list of **3–7 shots**, each of which is a fully specified brief for the Video-Gen Node. This is where narrative becomes producible — and, per Section 2.4, where genericness is structurally blocked rather than merely discouraged.

**Model / API.** Qwen (Qwen-Plus is sufficient here; the task is structured decomposition rather than open-ended reasoning).

**Input contract.** `winning_script` (with beats), `treatment` (Section 5.5), `target_length_sec`. **Note: there is no `product_category` field anywhere in this contract or the shot schema** — this is deliberate. A category field is the seam through which a lookup-table shortcut would re-enter the system, so it is removed from the schema entirely rather than merely discouraged in the prompt.

**Output contract — per-shot schema.** Every field exists for a specific downstream reason:

| Field | Purpose / why it exists |
|---|---|
| `shot_id` | Stable identity for fan-out, retries, ledger and asset joins |
| `t_start`, `t_end` | Position in the timeline; drives Assembly ordering and voiceover sync |
| `beat_role` | `hook \| problem \| demo \| proof \| cta` — narrative function of the shot; also the key the Treatment Agent's beat-function-to-shot-grammar reasoning was keyed on |
| `description` | The actual **video-gen prompt text** for this shot |
| `shot_type` | Enum: `hook_hero \| macro_detail \| lifestyle_context \| hero_reframe \| cta_endcard \| product_in_hand` — a vocabulary of *compositional* options, not a category-keyed lookup; which option is chosen is decided per-job from the treatment, not fixed per product type. `product_in_hand` (v2, Phase 2 research) is the dedicated composition for `demo`/`proof` beats that need human-product interaction (a grip, a press, a pour) — without it those beats were being forced into the too-wide `lifestyle_context` or the human-free `macro_detail` |
| `camera_move` | Enum: `push_in \| orbit \| static \| pan \| tilt_up \| pull_back \| rack_focus`. **NEVER compound/stacked moves** — stacked camera moves visibly break current text-to-video models. `rack_focus` (v2, Phase 2 research) is a single optical event (so it doesn't violate the no-stacking rule) that shifts sharpness between two named product referents on different planes — it is structurally non-generic, since it cannot be specified without naming two real facts (e.g. label → double-wall seam). Reserve it for shots whose `justification`/reasoning references two distinct product referents, not one. Drift-risk ordering for the enum (Phase 2 research, evidence-backed): `static`/`push_in` are safest and may run the full 3–5s; `orbit` is highest-risk (the model must invent unseen geometry from one flat reference photo) — keep orbit **short/partial (a 15–30° arc, never a full rotation)** and never combine with a tight framing on fine text |
| `framing` | `fills_frame \| rule_of_thirds_left \| rule_of_thirds_right \| context_wide` |
| `lighting` | A **single shared style string reused across ALL shots** in the ad, e.g. *"soft key light, neutral background, clean commercial look"* — sourced from the treatment's `color_story`, not a stock phrase |
| `negative_prompt` | **Expanded (Phase 2 research), identity-first ordering** — earlier terms are weighted more heavily by Wan/Kling-family models, so geometry/label/texture-stability terms lead: *"warped label, distorted logo, morphing text, melted edges, deformed packaging, changing product shape, geometry warp, color shift, texture flicker, warbling surface, extra logos, extra text, watermark, subtitles, floating product, duplicated product, product leaving frame, background warping, flickering, jitter, deformed hands, fused fingers, low quality"*. Kept as one shared boilerplate string like `lighting`, tuned per-shot only for a shot's specific risk. Pair with a **positive** identity-preservation clause in `description` (see Prompt-design strategy below) — negative prompts alone did not fix the vanishing-product failure mode observed in our own de-risk test (`docs/DERISK_VIDEO_GEN_RESULT.md`); the positive instruction did |
| `reference_image_id` | Points to one of the seller's uploaded product photos; used for **image-to-video conditioning** |
| `text_overlay_zone` | `none \| left_third \| right_third \| lower_third` — **reserved negative space** for post-generation caption/CTA burn-in. AI-generated on-screen text/logos garble, so text is *never* generated directly — it is composited later into this reserved zone |
| `duration_sec` | Short by design (3–5s) — **drift compounds over longer single-shot durations**, so we keep shots brief |
| `allocated_budget` | This shot's slice of the job's cost cap; consumed by the Budget Gate |
| `voiceover_line` | The script line spoken over this shot; the Voiceover Agent syncs to it |
| `justification` | **New.** `{ script_quote: str, truth_fact_id: str, treatment_ref: str }` — `script_quote` must be a verbatim substring of `winning_script.text`; `truth_fact_id` must exist in `product_truths[]`; `treatment_ref` must match a `beat_treatments[].beat_index` from Section 5.5. This is the field that structurally forces every camera/composition decision to be script-conditioned rather than template-conditioned. |
| `status` | Lifecycle: `pending \| generating \| passed \| fallback \| review` |
| `retry_count` | Continuity-retry counter; the retry cap lives here in typed state |

**Generation architecture — two sequential Qwen calls, not one (Phase 2 research decision, revises the original single-call design).** The original spec asked for one structured-output call with `justification` ordered before the rest of the shot's fields, reasoning that field order alone would force citation-before-composition. Phase 2 research into Qwen/DashScope's structured-output modes found this guarantee doesn't hold: DashScope's plain `json_object` mode does not grammar-force key-emission order the way OpenAI's strict `json_schema` mode does, so on Qwen the ordering benefit would only be probabilistic, not structural. The agent is therefore split into two calls:
- **Call A — Justify.** Produces *only* `[{shot_id, beat_role, script_quote, truth_fact_id, treatment_ref}]` for every shot, nothing else. Sources are presented to the model as **numbered/ID'd menus, not prose** — script beats as a numbered list of exact quotable lines, `product_truths[]` as an explicit `id → fact` table, `treatment.beat_treatments[]` as an explicit `beat_index → visual_approach` table — with an explicit instruction that `script_quote` must be copied character-for-character from one listed line and `truth_fact_id`/`treatment_ref` must be one of the listed IDs. Low temperature (~0.2–0.3): this is structured extraction, not ideation.
- **Justification Validator (owned by KR, Section 5.6 handoff below) runs on Call A's output before Call B ever runs.**
- **Call B — Realize.** Given each shot's *already-validated* justification, produces the remaining camera/composition fields (`shot_type`, `camera_move`, `framing`, `description`, `negative_prompt`, `text_overlay_zone`, `duration_sec`, `voiceover_line`). Because the justification is now a validated fact in the prompt rather than a hoped-for token-order effect, the camera choice is provably conditioned on a real quote + real truth, not merely correlated with one.

This is the same "validate the small thing, then build on it" shape already used elsewhere in this codebase (Body-Checker's deterministic pre-pass before its LLM ruling; the Merge Coherence Validator's pacing re-check before its blind coherence read) — re-prompting on a Call A failure only re-runs the tiny justification object, not the full shot list, which converges faster and cheaper.

**Interface handoff to the Justification Validator (KR, deterministic, separate module).** The Shot-List Agent (RR) calls the Justification Validator as a synchronous step *inside* its own node, between Call A and Call B — not as an independent downstream graph node. Contract: `validate_justifications(justifications: list[dict], winning_script: WinningScript, product_truths: list[ProductTruth], treatment: Treatment) -> list[ValidationResult]`, one `ValidationResult` per shot (`{shot_id, passed: bool, violation: Optional[str]}`), checking:
1. `script_quote` is a verbatim (fuzzy-matched, case-insensitive) substring of the winning script text.
2. `truth_fact_id` exists in `product_truths[]`.
3. `treatment_ref` exists in the treatment's `beat_treatments[]`.
4. `script_quote` is not under 4 words, and the justification doesn't match a stoplist of category-generic phrases ("show the product clearly," "highlight quality").

Any shot failing 1–4 triggers a **single re-prompt on Call A, naming the exact violating shot, field, and failure type** (e.g. "Shot 3's quote is not verbatim — copy an exact span from these lines," "`t9` does not exist, choose from: t1, t2, t3," or "quote too short / matched a banned generic phrase — cite a specific ≥4-word span") so the retry is surgical rather than a generic "try again." On the second failure, that shot's justification falls back to the corresponding treatment beat's literal `visual_approach` verbatim (already quote-grounded by construction) and Call B proceeds from the fallback rather than blocking the job. Shots exceeding the pacing rule, using an out-of-enum camera move, or omitting a `reference_image_id` are repaired the same way as before (clamping shot count, snapping to nearest valid enum, defaulting the reference photo).

**Anti-genericness reasoning aids for Call B (Phase 2 research).** Beyond the justification gate, the prompt should give the model an explicit rubric for *why a specific camera_move follows from a specific fact*, so the choice isn't merely grounded but actually motivated:

| `camera_move` | Motivated only when the cited fact names… | Generic misuse to reject |
|---|---|---|
| `orbit` | a 3-D form worth circling (a bezel, a faceted surface, a sculpted silhouette) | "it's in this category, orbit it" |
| `push_in` | one arriving detail (a seam, an engraving, a texture) | pushing in on nothing in particular |
| `tilt_up` | a vertical geometry worth revealing | product isn't tall |
| `rack_focus` | two facts on different planes worth linking | only one plane of interest |
| `pull_back` | a scale/context reveal the product's size makes meaningful | reveals nothing new |
| `static` | a claim proven *by stillness* | default laziness |

Pair this with an explicit **swap test** instruction in the prompt: *"Before finalizing each shot, ask: if this product were replaced by a category competitor, would this exact shot still work? If yes, the shot is too generic — change it."* This is the natural companion to the Justification Validator's mechanical stoplist check — the rubric and swap test catch *plausible-sounding but category-generic* choices that would still pass the deterministic gate.

**Prompt-design strategy for Call B's `description`.** Keep shots **short (3–5s)**, static/push_in at full length and orbit/rack_focus short, to limit drift; **reuse one shared `lighting` string** across every shot; **reserve a `text_overlay_zone`** on any shot carrying a caption/CTA. Order the prompt text **Subject → Action/Motion → Camera → Lighting → Composition → Mood → Quality (80–120 words)**, with the product's real color/material/logo named **in the first 20–30 words** (front-loaded terms carry the most weight for Wan-family models) — spend the remaining budget on motion/camera, not re-describing the static scene the reference photo already fixes. Append a **positive identity-preservation clause** ("preserve product shape, keep label text, keep proportions") and, on any shot with human interaction or a scene-transition-adjacent move, an **anti-cut clause** ("product stays centered, never leaves frame, no scene cut") — this positive instruction, not the negative prompt, is what fixed the vanishing-product failure mode in our own de-risk test.

### 5.7 Budget Gate — *deterministic conditional edge*

**Purpose.** Enforce the hard cost cap **before** any money is spent on video generation. This is the "quality under a limited budget" guarantee made concrete.

**Model / API.** None — a deterministic conditional edge in the graph.

**Logic.** Sums `allocated_budget` across all shots and compares against the job's hard cost cap. **If over cap: one deterministic reduce pass, applied by the Budget Gate itself — NOT a loop-back into the Shot-List Agent's LLM calls** (revises the original "loops back to the Shot-List Agent" wording; see "Reduce is deterministic, not generative" below for the reasoning). **If within cap:** writes the approved per-shot ledger entries to the **Budget Ledger** table and releases the shots to the fan-out. Because both the cap and the running spend live as plain fields on graph state, the gate is a pure function of state and the live ledger streams to the frontend (dashed edge `BL -.-> FE`). Runs only **after** the Justification Validator (Section 5.6) has passed every shot, so budget is never spent evaluating a shot that would have been rejected on grounding grounds anyway.

**Reduce is deterministic, not generative (Phase 2 research decision).** The original spec's "loop back to the Shot-List Agent" wording was ambiguous about whether that meant re-invoking its LLM calls — which would mean new, unvalidated content re-entering the pipeline with no re-check before acceptance, since the original wording also said the second pass "accepts whatever fits." Two things settle this in favor of a deterministic repair, not a fresh generation:
- **Internal precedent.** Section 5.4.7 already describes the Pacing-Checker's own repair step as "the same 'reduce to fit' spirit as the Budget Gate's one loop-back" — and that repair is explicit deterministic math (proportional re-timing), not an LLM call. The Budget Gate was always meant to match that shape.
- **External practice.** Constraint-violation repair in LLM pipelines is generally better handled as a minimal, localized, deterministic edit over the existing (already-validated) output than as full regeneration — full regeneration re-opens content that already passed validation to no-longer-checked risk, costs more, and adds latency for a decision that doesn't need creative judgment.

Concretely: **cutting** a shot (removing an entry, redistributing its allocation) is mechanical — no content is generated, so nothing new needs validating, and the guarantee stays airtight (the reduced list is always a strict subset of already-validated content, never new material). **Merging** two shots into one coherent shot would be a genuine creative operation and is explicitly OUT of scope for now — the Budget Gate's reduce step only ever cuts, never merges. If content-preserving merging is wanted later, it would need an LLM call routed through the same repair-then-mandatory-re-validate pattern the Copy Editor already uses (Section 5.4.8) — never accepted on faith. Decide whether to build that path later, based on how often the deterministic cut-only reduce actually fires in real testing.

**`allocated_budget` semantics (Phase 2 research decision).** `allocated_budget` represents **real generation cost**, `duration_sec × rate(resolution_tier)`, not an abstract proxy — Wan pricing is flat per-second-by-resolution (camera move/shot complexity do not change cost), so a real-dollar ledger is both accurate and makes the "hard cost cap" and the live dashboard ledger literal rather than illustrative.

**Allocation formula — grounding-weighted, not equal-split.** Because shots are capped at 3–5s for drift reasons (Section 5.6), "give the important shot more" cannot mean more seconds — it has to cash out as **resolution tier and retry reserve** instead. For each shot `i`:
1. `base_i = duration_sec_i × rate_720p`.
2. `w_i = w_role[beat_role] × w_type[shot_type] × truth_bonus`, where `w_role` favors `hook`/`cta` (attention + conversion), `w_type` favors `macro_detail`/`hook_hero`, and `truth_bonus` (~1.1×) applies when the shot's `justification.truth_fact_id` cites a `product_truths[].category` of `material`, `texture`, `construction_detail`, or `imperfection` — i.e. the facts that make the product *specific*, not generic. Only the ratios between weights matter.
3. `alloc_i = (base_i × w_i) × (C / Σ(base × w))` — proportional allocation normalized to the job cap `C`.
4. Clamp each `alloc_i` to a feasible window (`min = 3s@720p`, `max = duration_sec_i@1080p`) and redistribute the remainder, reusing the existing clamp-and-redistribute `_waterfill()` routine already implemented in `backend/agents/meta_critic.py` for time-budget allocation — same algorithm family, second use in this codebase. A shot whose allocation reaches its 1080p cost is flagged to render at 1080p with one reserved retry; overflow redistributes to the next-highest-weight unclamped shots.

**What the deterministic reduce pass actually does, when over cap.** Priority-ordered, not uniform trimming (uniform trimming would starve the grounding-carrier shot exactly as hard as a throwaway transition, which defeats the point of the weighted allocation above), applied in-process by the Budget Gate against the already-validated shot list — no LLM call, per "Reduce is deterministic, not generative" above:
1. Downgrade resolution tier and drop the reserved retry on the lowest-weight shots first; recompute the total.
2. If still over cap, **cut** (never merge — see above) the single lowest-weight shot — never the `hook`, `cta`, or top-weighted `macro_detail`/`proof` shot — and redistribute the remaining shots' allocation via the same `_waterfill()` pass; recompute.
3. Uniform per-shot trimming is the last resort only.
4. **Floor case:** if the shot list is already at `MIN_SHOTS` (3) and every shot is already at its cheapest resolution with no retry reserve left to drop, there is nothing left to mechanically cut without breaking the 3-shot contract. Accept the over-cap total rather than pretend it fits, and flag it visibly in the reasoning trace and the `budget_updated` event (e.g. `over_cap: true`, `overage: $X`) rather than silently violating the cap — a visible overage beats a hidden one.

### 5.8 Video-Gen Node — *orchestration around the Wan/HappyHorse API*

**Purpose.** Generate each shot as actual video. This node is orchestration, not reasoning — it does not "think," it constructs a well-formed generation request and calls the model. Shots fan out and run **in parallel via LangGraph's `Send()`**.

**Model / API.** Wan / HappyHorse (Tongyi Wanxiang video family) — **image-to-video mode only**.

**Prompt-construction strategy.** For each shot it assembles a structured prompt following the formula **Subject → Action/Motion → Camera → Lighting → Composition → Mood → Quality (80–120 words)**, then appends the shot's `negative_prompt`. It **always passes the reference product photo for image-to-video conditioning — never pure text-to-video.** The reference image is the single biggest lever against product-identity drift (color, shape, and label changes); generating from text alone would let the model invent a product that is not the seller's.

**Input contract.** A single shot object (from the shot list) plus the OSS URI of its `reference_image_id` photo. **Output contract.** `generated_shots[shot_id] = {video_uri, drift_score (set later by Continuity), attempt}`, with the clip persisted to OSS.

**Failure handling.** On a **hard API failure or timeout** — an actual call failure, *not* a quality issue — the node routes **immediately to the Fallback (Ken-Burns) node** and, importantly, **does not consume a Continuity retry**. The retry budget is reserved for quality problems; infrastructure failures are handled by graceful degradation instead.

### 5.9 Fallback Node — "Ken-Burns pan" — *deterministic*

**Purpose.** Keep the pipeline moving when a shot cannot be generated, rather than letting one failed API call sink the entire demo.

**Model / API.** None — deterministic ffmpeg.

**Behavior.** On a video-gen hard failure, it produces a simple **pan/zoom (Ken-Burns) animation over the static product photo** for the shot's duration, marks the shot `status = fallback`, and forwards it to Assembly. This demonstrates **graceful degradation under constraint** — a slightly less dynamic shot in an otherwise complete ad — instead of a failed run. It is also reachable from the human-review interrupt, where a seller can explicitly *accept the fallback* for a persistently drifting shot.

### 5.10 Continuity Agent — Qwen-VL via DashScope

**Purpose.** Guard visual fidelity. For each generated shot it checks two things: **(a) product-identity drift** — does the generated product still match the source reference photo in color, shape, and label? — and **(b) cross-shot style consistency** — does the shot match the shared `lighting`/style string used across the ad?

**Model / API.** Qwen-VL (vision-language) via DashScope.

**Input contract.** The generated shot's frame(s), the source reference product photo, and the shared lighting/style string. **Output contract.** A **drift/similarity score** written to `generated_shots[shot_id].drift_score`, plus a short justification streamed live to the dashboard (dashed edge `CTY -.-> FE`).

**Control flow / failure handling.**
- **Drift within threshold** → shot passes to Assembly.
- **Drift over threshold and `retry_count < 2`** → re-queues the shot to the Video-Gen Node with `retry_count` incremented (optionally with a tightened prompt).
- **Drift over threshold and retries exhausted** → triggers a **LangGraph `interrupt()`**. This is a *genuine pause/resume*, not a dead-end flag: the graph checkpoints, and the flagged shot surfaces in the seller-facing UI with three options — **approve as-is**, **retry with an edited prompt**, or **accept the Ken-Burns fallback**. When the seller responds, the graph **resumes from the checkpoint**. This makes the human a bounded escape valve for the hardest shots without stalling everything else.

### 5.11 Voiceover + Caption Agent — Qwen TTS / CosyVoice via DashScope

**Purpose.** Produce the ad's audio and caption timing. It runs as a **parallel branch that starts as soon as the winning script is finalized** (edge `MC → VOX`), because voiceover depends only on the script, not on the rendered video — so it overlaps with the entire multi-minute video-generation branch and costs no extra wall-clock time.

**Model / API.** Qwen TTS / CosyVoice via DashScope.

**Input contract.** The finalized winning script with per-beat timestamps. **Output contract.** `voiceover = {audio_uri, caption_track_uri}` — a synthesized voiceover audio track and a caption timing track, both **aligned to the script's beat timestamps** so Assembly can lay them against the matching shots. **Failure handling.** If synthesis fails for a line, the node retries that line; on persistent failure the ad can assemble with captions only (silent-with-captions is a valid short-form format), recorded in the trace.

### 5.12 Assembly Agent — ffmpeg, *deterministic (not an LLM call)*

**Purpose.** Cut the master video. **Model / API:** none — deterministic ffmpeg. It **stitches all approved and fallback shots in script order**, overlays the **voiceover audio track**, **burns captions and the CTA text into each shot's reserved `text_overlay_zone`**, and aligns basic transitions and the music-timing cue to the script's pacing. **Output:** a single **master cut** written to OSS (`master_cut_uri`). Because it consumes only the shot list's ordering/timing metadata and the pre-generated clips, it is fully deterministic and adds no model cost. Burning text into the reserved zones — rather than ever generating on-screen text — is what keeps captions and CTAs crisp instead of garbled. This is also the node a **CTA-text-only chat edit** (Section 5.16) re-enters directly, since changing overlay text never requires touching a video clip.

### 5.13 Format Export Node — ffmpeg, *deterministic*

**Purpose.** Recompose the single master cut into **three aspect ratios** — **9:16** (TikTok/Reels/Shorts), **1:1** (feed), and **16:9** (YouTube) — by **recropping and repositioning the reserved text-overlay zones per format**. Because it reuses the already-generated shots, it incurs **no additional LLM or video-gen cost** — the reserved negative space from the shot schema is exactly what makes per-format recropping safe (text never gets cropped off, because its zone is known). **Output:** `exports = {aspect_9x16, aspect_1x1, aspect_16x9}`.

### 5.14 Output

**Purpose.** Finalize the job. Pushes the **final videos in all three formats to OSS**, marks the job **complete in the DB**, and emits the final event to the frontend, which then displays the results **plus the full cost and reasoning breakdown** (the four scripts, critic scores, merge justification and its independent validator verdict, the director's treatment, per-shot justifications, budget ledger, and drift scores). This closing transparency payload is what turns an autonomous black box into a demoable, auditable showrunner — and it is also the state the chat-revision subsystem (Section 5.16) treats as its starting checkpoint.

### 5.15 SellerDirection & Truth/Treatment Recap *(schema note, not a node)*

Sections 5.1, 5.2, and 5.5 together form the "script-driven direction" extension described in Section 2.4. There is no additional node here — this subsection exists purely to make the data lineage explicit: `seller_direction` (5.1) → informs → `product_truths[]` (5.2) and `Concept Agent` (5.3) → informs → `Treatment Agent` (5.5) → informs → `Shot-List Agent` (5.6), with the **Justification Validator** as the single mechanical checkpoint that enforces the whole chain actually stayed grounded end to end.

### 5.16 Chat-Based Revision Subsystem *(new — post-generation)*

**Purpose.** Once `Output` (5.14) has produced a finished ad, the seller can converse in natural language to request a change ("make the hook punchier," "the second shot feels too dark," "shorten it to 15s," "change the CTA text") and get a **targeted fix**, not a full re-run. This subsystem is a separate, small sub-graph that attaches to the completed job's checkpoint rather than a modification of the main pipeline's edges.

#### 5.16.1 Edit Router — Qwen via DashScope

**Purpose.** Classify a chat message to the minimal graph re-entry point. **Input:** the chat message, the current job's full state snapshot (script, treatment, shot list, generated assets). **Output contract:**
```json
{
  "scope": "shot_visual | copy_tone | pacing_length | cta_text | global",
  "target_shot_ids": ["shot_3"],
  "entry_node": "shot_list",
  "confidence": 0.92,
  "rationale": "message names 'the second shot' and 'too dark' -> lighting/exposure change scoped to one shot"
}
```
**Routing table** (the concrete mapping from research into this codebase):

| Chat message pattern | `scope` | `entry_node` | Re-runs |
|---|---|---|---|
| "make the hook punchier" / tone complaint | `copy_tone` | `concept` | Concept + Critic Chain + Treatment + Shot-List (only hook shots) + Video-Gen (only hook shots) |
| "shot 2 is too dark" / single-shot visual complaint | `shot_visual` | `shot_list` | Shot-List (one shot) + Video-Gen (one shot) + Continuity (one shot) |
| "shorten it to 15s" | `pacing_length` | `shot_list` | Shot-List (re-time/drop shots) + Assembly + Export — **no Video-Gen call** if trimming existing clips suffices |
| "change the CTA text" | `cta_text` | `assembly` | Assembly + Export only — **zero LLM reasoning calls beyond the router, zero video-gen** |
| "more energetic overall" | `global` | `concept` (via Treatment re-seed) | Treatment (new persona) + Shot-List (all shots) + Video-Gen (all shots) — the one path that is intentionally expensive, because the ask is genuinely global |

**Failure handling.** Below a confidence threshold (e.g. 0.6), the router does not guess — it asks a clarifying question in the chat UI instead of silently picking a scope.

#### 5.16.2 Edit Interpreter — Qwen via DashScope

**Purpose.** Prevent chat-based revision from reintroducing genericness. Vague asks are translated into a **specific, grounded parameter delta** before anything re-runs — the same discipline as Section 2.4, applied to an edit rather than a first pass.

**Input contract.** The chat message, the Edit Router's output, and the current `treatment` / `shot_list` / `winning_script` slices relevant to `target_shot_ids`. **Output contract.** A patch object, e.g. for "more energetic":
```json
{
  "treatment_patch": { "director_persona": "kinetic, high-contrast, fast-cut — favors movement and quick reveals" },
  "shot_patches": [
    { "shot_id": "shot_2", "duration_sec": 1.5, "camera_move": "pan" },
    { "shot_id": "shot_3", "duration_sec": 1.5, "camera_move": "push_in" }
  ],
  "justification": "persona re-seeded per chat request; durations tightened from 2.5s to fit a faster-cut pacing philosophy consistent with the new persona"
}
```
Every patch field still passes through the **same Justification Validator** from Section 5.6 before it is accepted — an edit patch is not exempt from grounding requirements.

#### 5.16.3 Preview/Confirm Gate — LangGraph `interrupt()`

**Purpose.** Video-gen calls cost real money; no re-render fires without the seller seeing what will change and approving it first. **Behavior.** Before the forked branch reaches Video-Gen, the graph interrupts and the frontend renders a **diff view**: old vs. new script text (if changed), old vs. new shot cards (camera move, duration, justification) for every `target_shot_ids` entry, and an **estimated incremental cost** (number of shots that will actually re-render × per-shot cost from the Budget Ledger). The seller either **confirms** (graph resumes, forked branch proceeds to Video-Gen) or **rejects/refines** (returns to chat, no spend occurs). This reuses the identical `interrupt()` primitive as the Continuity human-review gate (Section 5.10) — same mechanism, second purpose.

#### 5.16.4 Scoped Re-execution — LangGraph checkpoint fork

**Purpose.** Actually perform the "re-run only what's downstream" guarantee. **Mechanism.** Every completed job is a LangGraph `thread_id` with a full checkpoint history (one checkpoint per node, per Section 2.2). On confirm, the backend calls `get_state_history(thread_id)` to find the checkpoint corresponding to `entry_node`, calls `update_state(checkpoint_id, patch)` with the Edit Interpreter's patch, then `graph.invoke(None, {"configurable": {"thread_id": new_branch_id}})` — which **creates a new branch** starting from that checkpoint with the patched state, re-running only nodes downstream of `entry_node`. Untouched shots' `generated_shots[shot_id]` entries are copied byte-identical into the new branch's state (their OSS URIs are simply reused, no re-render), so Assembly only ever re-stitches what actually changed. Every branch is retained (not overwritten) so the seller can revert to any prior version from a version list in the UI.

**Failure handling.** If a fork's re-run fails (e.g. a re-generated shot drifts and exhausts retries), it surfaces the same Continuity `interrupt()` as a first-pass run would — the chat-revision path shares every downstream failure-handling mechanism already built for the main pipeline; it does not need its own.

---

## 6. Shared State Schema

LangGraph passes a single typed state object through every node; each node reads the fields it needs and returns updates that LangGraph merges back in. This shared state is where the retry cap, the budget ledger, the human-review queue, and now the truth/treatment/justification/edit lineage physically live — which is what makes the bounded loops, pause/resume behavior, and scoped re-execution inspectable rather than hidden in control flow.

```text
job_id
brief
product_photos[]
seller_direction: {                 // nullable — all fields optional
    mood_words[], reference_ad{ url_or_text, why }, never_do, freeform
}

product_truths[]: {                 // NEW — Section 5.2
    truth_id, fact, category, source
}

script_variants[]: {
    variant_id, text, framework, hook_type, emotional_trigger,
    grounding_truth_ids[],           // NEW
    beats: [{ t_start, t_end, line }],
    target_length_sec
}

critic_scores{ variant_id: {
    hook, pacing,
    completion,                     // NEW — Body-Checker's completion_score (5.4.3), fills the previously-undefined axis
    completion_detail: { redundant_beat_pairs[], promise_payoff_match, emotional_trigger_landed },   // NEW
    cta, tone, composite, justification, never_do_violation
} }

merge_attempts[]: {                 // NEW — Section 5.4.6/5.4.7/5.4.8, one entry per merge attempt (usually 1, up to 2 + fallback)
    attempt_number, hook_source_variant, body_source_variant, cta_source_variant,
    merged_script,
    pacing_recheck: { passed, violations[] },
    coherence_check: { passed, coherence_score, voice_consistency, promise_payoff_match, register_shift_flags[], justification },
    copy_edit: { seams_edited[], original_seam_text, revised_seam_text, justification },  // NEW — Section 5.4.8, populated only when the Copy Editor ran on this attempt
    outcome: "accepted | copy_edited_then_accepted | retried | fell_back_to_variant"
}

winning_script                      // only set once the Merge Coherence Validator returns passed:true, or the fallback variant is selected
reasoning_trace                     // now includes merge_attempts[] alongside critic_scores, so a retry/fallback is part of the visible trace, not hidden

treatment: {                        // NEW — Section 5.5
    director_persona, color_story, pacing_philosophy,
    beat_treatments[]: { beat_index, beat_function, script_quote, truth_fact_id, visual_approach, why_not_generic }
}

shot_list[]: {                      // full schema from Section 5.6
    shot_id, t_start, t_end, beat_role, description,
    shot_type, camera_move, framing, lighting, negative_prompt,
    reference_image_id, text_overlay_zone, duration_sec,
    allocated_budget, voiceover_line, status, retry_count,
    justification: { script_quote, truth_fact_id, treatment_ref }   // NEW
}

budget_ledger: { cap, spent, per_shot{} }

generated_shots{ shot_id: { video_uri, drift_score, attempt } }

voiceover: { audio_uri, caption_track_uri }

master_cut_uri

exports: { aspect_9x16, aspect_1x1, aspect_16x9 }

human_review_queue[]

// --- Chat-revision fields (NEW — Section 5.16) ---
chat_thread[]: { role, message, ts }
edit_requests[]: {
    edit_id, message, router_output{ scope, target_shot_ids, entry_node, confidence, rationale },
    interpreter_patch{ treatment_patch, shot_patches[], justification },
    status: "pending_preview | confirmed | rejected | applied | failed",
    fork_branch_id, estimated_cost, actual_cost
}
version_history[]: { branch_id, parent_branch_id, created_at, summary }
```

**Notes on key fields.**
- `product_truths[]`, `treatment`, and `shot_list[].justification` together are the mechanical backbone of "script-driven, not category-driven" direction (Section 2.4) — every one of them is validated against the others by the Justification Validator before the Budget Gate runs.
- `critic_scores`, `merge_attempts[]`, and `reasoning_trace` together form the streamed negotiation trace — they are populated by the Critic Chain, the Meta-Critic, the Merge Coherence Validator, and the Copy Editor, and read by the frontend live. `merge_attempts[]` is what makes the "no self-grading" fix inspectable: a reader of the trace can see the Meta-Critic's candidate, the independent validator's pacing/coherence verdict on it, which repair path was taken when it failed (a Copy Editor seam polish with its before/after text, or a Meta-Critic swap), and whether a retry or fallback followed — separate entries from separate nodes, not one node's unopposed say-so.
- `shot_list[].retry_count` is where the Continuity retry cap is enforced; the conditional edge back to Video-Gen reads this field, so the loop is bounded by state, not by ad-hoc counters.
- `budget_ledger` is updated by the Shot-List Agent (allocations), the Budget Gate (approval), and the Video-Gen Node (actual spend), and streamed to the dashboard throughout.
- `human_review_queue` holds shots parked by a Continuity `interrupt()` awaiting seller resolution; the graph resumes from its checkpoint when an entry is resolved.
- `edit_requests[]` and `version_history[]` are populated only by the chat-revision subsystem (Section 5.16); `fork_branch_id` is the LangGraph `thread_id` of the branch created by a confirmed edit, and `version_history[]` is what the "old vs. new" version picker in the UI reads.

---

## 7. Database Design

ProductCut persists **structured relational state** in a managed Postgres-compatible database and **binary assets** in OSS. The split is deliberate: the job → shots → assets relationships are genuinely relational with clear foreign keys, so they belong in a relational DB; photos and video files are large opaque blobs that belong in object storage, never in the DB.

### 7.1 Recommended Engine

**Alibaba Cloud's managed Postgres-compatible database — PolarDB or RDS for PostgreSQL.** The data has clear foreign-key relationships (`jobs` → `shot_lists` → `generated_assets`), transactional budget-ledger writes, and structured JSON columns for the script/shot/treatment/truth payloads — all of which a managed relational Postgres handles natively. OSS is used **purely for binary assets** (product photos, per-shot clips, master cuts, exports), referenced from the DB by URI — **not** for structured state. The same instance also hosts the **LangGraph checkpoint tables** (`langgraph-checkpoint-postgres`'s own schema), so no second database is introduced for the chat-revision subsystem.

### 7.2 Tables

| Table | Key columns | Why it must persist |
|---|---|---|
| `jobs` | `job_id` (PK), `seller_id`, `brief`, `status`, `created_at`, `product_photo_refs` | The root record for every submission; drives status/resume and links to all child rows |
| `seller_direction` | `job_id` (FK), `mood_words[]`, `reference_ad_url_or_text`, `reference_ad_why`, `never_do`, `freeform` | Persists the optional intake so it can be re-read by any later chat edit, not just the first pass |
| `product_truths` | `job_id` (FK), `truth_id`, `fact`, `category`, `source_photo` | The grounding facts every downstream justification must cite; needed for audit and for the Justification Validator to re-check on edit forks |
| `budget_ledger` | `job_id` (FK), `shot_id` (FK), `allocated`, `spent`, `cap` | Per-shot budget accounting; source of the live cost dashboard, the hard-cap enforcement audit trail, and the chat-edit "estimated incremental cost" preview |
| `script_variants` | `job_id` (FK), `variant_id`, full script JSON, critic scores | Preserves all four candidate scripts and every critic score for the transparency breakdown and negotiation trace |
| `treatments` | `job_id` (FK), `version` (int), full treatment JSON | One row per treatment version — the original plus any chat-edit re-seed, so the UI can show "persona was X, now Y" |
| `shot_lists` | `job_id` (FK), `shot_id`, `version` (int), full shot schema incl. `justification`, `status`, `retry_count` | The producible shot plan; `status`/`retry_count` persistence is what lets a crashed run resume mid-generation; `version` lets a chat-edited shot coexist with its prior version for the diff view |
| `generated_assets` | `job_id` (FK), `shot_id` (FK), `version` (int), `video_uri`, `drift_score`, `attempt_number` | Maps each generated (or fallback) clip in OSS back to its shot and version, with its drift score and attempt number; untouched-shot rows are simply reused across chat-edit versions, not duplicated |
| `human_review_events` | `job_id` (FK), `shot_id` (FK), `flagged_at`, `resolution` | Records every Continuity interrupt and how the seller resolved it — the human-in-the-loop audit log |
| `chat_edit_requests` | `job_id` (FK), `edit_id`, `message`, `router_output` (JSON), `interpreter_patch` (JSON), `status`, `fork_branch_id`, `estimated_cost`, `actual_cost`, `created_at` | The full audit trail of every chat-based edit request, its routing decision, its grounded patch, and its eventual cost — needed both for the preview/diff UI and for judging transparency |
| `job_versions` | `job_id` (FK), `branch_id` (= LangGraph `thread_id`), `parent_branch_id`, `created_at`, `summary` | The version tree the seller navigates ("v1 original", "v2: punchier hook", "v3: 15s cut"); `parent_branch_id` lets the UI render it as a tree, not just a flat list |

The foreign-key chain `jobs.job_id → shot_lists → generated_assets` (and `→ budget_ledger`, `→ human_review_events`, `→ chat_edit_requests`, `→ job_versions`) is exactly why a relational engine is the right choice: these joins are constant, and referential integrity keeps orphaned assets and ledger entries from accumulating — including across chat-edit branches, where a naive design could otherwise silently duplicate or orphan asset rows.

---

## 8. Error Handling & Graceful Degradation

ProductCut treats robustness as a demo feature, not an afterthought: every failure mode has a **bounded, defined** outcome, and none of them can hang the pipeline in an unbounded loop.

- **Video-gen hard failure (API error / timeout).** Routes **immediately to the Ken-Burns fallback**, and **does not consume a Continuity retry.** Infrastructure failures and quality failures are handled by different mechanisms, so an outage doesn't burn the quality-retry budget.
- **Continuity drift.** Capped at **2 retries** back to Video-Gen. On exhaustion, escalates to a **human-review `interrupt()`** — a **real pause/resume via LangGraph's checkpointer**, not a dead-end flag. The seller resolves it (approve / edit-and-retry / accept fallback) and the graph resumes from the checkpoint.
- **Budget overrun at the shot-list stage.** **One** loop back to the Shot-List Agent with an explicit "reduce" instruction, then a **hard accept** of whatever fits — **never an infinite loop.**
- **Merge coherence/pacing failure (new).** The Meta-Critic's cross-pollinated merge is re-checked by the **independent** Merge Coherence Validator (Section 5.4.7), never by the Meta-Critic itself. A failed deterministic pacing re-check gets **one** proportional re-timing repair, then one re-check. A failed independent coherence read is repaired by **whichever node fits the failure**, not a single undifferentiated retry: a voice/register seam problem (`register_shift_flags` / `voice_consistency = false`) goes to the **Copy Editor** (Section 5.4.8) for a constrained, in-place polish of only the flagged transition — preserving every cited fact, truth ID, and the CTA verb — before returning to the validator for re-check; a promise-payoff problem (`promise_payoff_match = false`) goes back to the Meta-Critic naming the specific flagged clash, so the retry swaps the flagged piece for its second-best-scoring alternative, not a blind re-roll. Either repair path consumes the same single bounded retry — this is not a second loop layered on top, it is what happens *inside* the one retry slot the validator already allows. A second failure of either sub-check (including a failed Copy Editor re-validation) **falls back to the single highest composite-scoring variant** — already individually valid, since it passed every checker unmerged — the same "bounded retry, then safe deterministic fallback" shape as every other loop in the system, except here the fallback decision is made by a node that never wrote the thing it is judging.
- **Justification/grounding failure (new).** A shot whose `justification` fails the deterministic validator (Section 5.6) gets **one** targeted re-prompt naming the exact violation; on a second failure it falls back to the treatment's literal `visual_approach` text for that beat (already quote-grounded) rather than blocking the job — the same "bounded retry, then safe deterministic fallback" shape as every other loop in the system.
- **Chat-edit routing ambiguity (new).** If the Edit Router's confidence is below threshold, it does not guess a scope — it asks a clarifying question in chat. This prevents a misrouted edit from silently re-rendering the wrong shot and spending budget on the wrong fix.
- **Chat-edit re-render failure (new).** A forked branch's Video-Gen or Continuity failure is handled by the **same** fallback/retry/interrupt machinery as a first-pass run (Sections 5.9, 5.10) — no separate error-handling path was built for edits, which keeps the failure surface to one set of mechanisms.
- **Checkpoint-backed durability.** LangGraph's **checkpointer persists state at every node**, so a crashed or flaky run **resumes instead of restarting from scratch.** This same mechanism doubles as a **demo-day safety net** (a **pre-warmed cached run** can be resumed/replayed if live generation is flaky during the presentation) and as the **substrate for chat-edit forking** — one persistence mechanism serves crash-recovery, demo replay, and scoped revision.

The throughline: every loop in the system (retry, budget, review, justification, merge validation, edit-routing) has a hard cap or a human off-ramp, and every intermediate state is checkpointed — so the worst case is a slightly degraded ad delivered on time, never a hung or lost run, and a chat edit that goes wrong never costs more than the one shot it touched.

---

## 9. Deployment on Alibaba Cloud

The entire backend runs on Alibaba Cloud, satisfying the hackathon's **Proof of Deployment** requirement. This section documents where each component runs and why.

### 9.1 Backend Compute — ECS (recommended) vs. Function Compute

The FastAPI + LangGraph application is deployed on **Alibaba Cloud ECS as the default choice.** The tradeoff:

- **ECS (recommended for this project).** A **persistent process** is the right fit because ProductCut relies on **long-lived WebSocket connections** (streaming the budget ledger, critic trace, and drift scores throughout a multi-minute run) and on **LangGraph checkpointing state** in memory / on disk between requests. A pipeline run takes minutes and streams continuously; a persistent server handles that naturally. The chat-revision subsystem's checkpoint forking (Section 5.16.4) has the same requirement — forking a branch is a stateful Postgres read/write followed by re-entering the graph, which is far more natural on a persistent process than on a cold FaaS invocation.
- **Function Compute (serverless).** Cheaper at idle, but **WebSocket and long-running graph execution are awkward on FaaS** — function timeouts and the stateless execution model fight both the multi-minute pipeline and the persistent stream. Function Compute is appropriate only for **stateless helper endpoints** (e.g., a thumbnail generator or a webhook receiver), not the orchestrator itself.

**Recommendation: ECS for the orchestrator**, specifically because of the long-lived WebSocket streaming, the multi-minute checkpointed pipeline execution, and now the stateful branch-forking the chat-revision subsystem depends on.

### 9.2 Qwen Cloud / DashScope — the model plane

**Every** LLM (Concept, Critic Chain, Meta-Critic, Treatment, Shot-List, Edit Router, Edit Interpreter), **vision** (Product Truth Extractor and Continuity via Qwen-VL), and **speech** (Voiceover via Qwen TTS/CosyVoice) call goes through **Qwen Cloud / DashScope's OpenAI-compatible API endpoint.** For the hackathon this must be **visibly, actually called in the codebase** — not merely referenced in docs. The submission includes a **recorded clip plus a linked code file demonstrating an actual Alibaba Cloud service call**, which is an explicit submission requirement. Because every intelligent node — including both new agents and the chat-revision subsystem — is a DashScope call, this requirement is satisfied pervasively rather than in a single token integration point.

### 9.3 Storage — OSS

Alibaba Cloud **Object Storage (OSS)** holds all binary assets: the seller's **product photos**, each **per-shot generated video clip**, the assembled **master cut**, and the **three aspect-ratio exports** (9:16 / 1:1 / 16:9). It also caches demo assets for the pre-warmed safety-net run. A chat edit that only touches copy or CTA text (Section 5.16.1) never writes a new OSS object at all — untouched clips are referenced, not re-uploaded. Structured state never lives in OSS — only blobs, referenced by URI from the DB.

### 9.4 Database — managed Postgres

The **managed Postgres-compatible DB (PolarDB or RDS for PostgreSQL)** holds all structured job and agent state from Section 7: jobs, seller direction, product truths, budget ledger, script variants, treatments, shot lists, generated-asset records, human-review events, chat-edit requests, job versions, and the LangGraph checkpoint tables. One instance, one connection pool, no second data store introduced for either extension.

### 9.5 Realtime — WebSocket over `astream_events`

A **FastAPI WebSocket endpoint** is fed by LangGraph's **`astream_events`**, pushing **budget-ledger updates, the critic reasoning trace, continuity drift scores, and now edit-router decisions and preview diffs** to the frontend **live during a run.** This is what makes the autonomous pipeline legible in the demo — judges watch the negotiation, the spend, and (in an extended demo) a live chat edit happen in real time.

### 9.6 Secrets & Configuration

All credentials — the **DashScope API key**, **video-gen (Wan/HappyHorse) API credentials**, the **DB connection string**, and **OSS access keys** — are injected via **Alibaba Cloud's secret/config management**, **never hardcoded** in the repository. This keeps the public GitHub repo (a submission requirement) free of live credentials while the deployed ECS instance receives them at runtime. No new secrets are introduced by either extension.

---

## 10. Judging Criteria Alignment

The hackathon weights four criteria. Below, each of ProductCut's features maps explicitly to the criterion it serves, so the demo walkthrough can point at concrete evidence for every point.

| Criterion | Weight | ProductCut features that satisfy it |
|---|---|---|
| **Technical Depth & Engineering** | **30%** | LangGraph DAG with a **bounded conditional retry loop** (Continuity → Video-Gen, capped at 2); **parallel fan-out via `Send()`**; **deterministic Budget Gate** with a hard cap and single reduce-loop; **checkpointer-backed resume** after crash; genuine **`interrupt()` pause/resume** human-in-the-loop; a **deterministic Justification Validator** that mechanically checks LLM output against source text before it's accepted; an **independent Merge Coherence Validator** that re-runs the Pacing-Checker's timing math and a blind coherence read against the Meta-Critic's own merge output before accepting it, with a capped retry **routed by failure type** to either the **Copy Editor** (constrained seam repair) or the Meta-Critic (piece swap) and a deterministic fallback; **checkpoint-fork-based scoped re-execution** for chat edits (re-running only downstream nodes, not the whole graph); deterministic ffmpeg Assembly and multi-format Export; clean split of relational state (Postgres) vs. blobs (OSS) |
| **Innovation & AI Creativity** | **30%** | The **Critic Chain** — five parallel specialist checkers (including a **Body-Checker** scoring promise-payoff, redundancy, and emotional-trigger fidelity) reconciled by a **Meta-Critic that cross-pollinates** the best hook + body + CTA across variants, then handed to an **independent post-merge validator** rather than self-graded, whose voice/register failures are repaired by a dedicated **Copy Editor** — a constrained, professionally-scoped seam-polish node distinct from both the writer (Concept Agent) and the merge-builder (Meta-Critic) — rather than a blind re-roll; **constraint-enforced script diversity**; a **Product Truth Extractor** that grounds every downstream agent in specific, checkable product facts instead of category assumptions; a **Treatment Agent** that reasons from narrative-beat function to camera grammar (not product category to camera grammar), modeled on a real director's treatment; a **justification-forced shot schema** that structurally blocks template reuse; a **chat-based revision subsystem** (Edit Router + Edit Interpreter + Preview/Confirm) that performs surgical, cost-aware re-generation instead of a blind full re-run |
| **Problem Value & Impact** | **25%** | A **named niche** (Etsy/Shopify sellers) with a real cost-savings story ($300–$1,500 studio ads → automated); output in **three ready-to-post aspect ratios**; **graceful degradation** (Ken-Burns fallback) so a seller always gets a finished ad; a **live cost dashboard** that makes spend transparent and bounded; a **post-generation chat revision** feature that mirrors how a real seller would actually work with a freelance editor ("make the hook punchier") without paying for a full re-shoot |
| **Presentation & Documentation** | **15%** | This complete technical document; the **mermaid architecture diagram**; the **live-streaming dashboard** (budget ledger, critic reasoning trace, drift scores, edit-router decisions) that makes the autonomous decisions watchable in real time; the closing **transparency breakdown** (four scripts, all critic scores, merge justification, director's treatment, per-shot justifications, ledger, drift scores) surfaced to the frontend; a **diff-style preview** before any chat-driven re-render |

**Cross-cutting note on Qwen Cloud usage.** Because every reasoning, vision, and speech decision routes through Qwen via DashScope — including the two new agents and the chat-revision subsystem — the "native Qwen Cloud usage" story is not confined to one integration point — it is exercised on essentially every intelligent node in the graph, and the LangGraph trace makes each call visible for judging.
