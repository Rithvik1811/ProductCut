# Merchant Marquee

> **Three photos in. One honest ad out.**

An autonomous multi-agent pipeline that turns 2–3 product photos and a one-line creative brief into a finished, narrated, 15–30 second short-form video ad — script, voiceover, shots, and all. Built on Qwen Cloud, LangGraph, and Alibaba Cloud infrastructure.

[![Deploy](https://github.com/Rithvik1811/merchant-marquee/actions/workflows/deploy.yml/badge.svg)](https://github.com/Rithvik1811/merchant-marquee/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)

---

## What It Does

Small Etsy and Shopify sellers can't afford professional video ads ($300–$1,500 per clip, days of turnaround). Merchant Marquee closes that gap end-to-end:

1. Seller uploads 2–3 product photos and a one-line brief
2. A coordinated crew of 14 AI agents runs the full production chain — extracting product truths from photos, writing four structurally distinct ad scripts, negotiating a winner, directing the visual approach, generating video shots in parallel, checking every frame for drift, and assembling a final cut with synced voiceover and captions
3. A finished 15–30 second ad is delivered in three aspect ratios (9:16, 1:1, 16:9) with a full transparency breakdown of every creative decision

---

## Architecture

```mermaid
flowchart TD
    U[Seller: photos + brief + optional intake] --> ING[Ingest Node]
    ING --> PTE[Product Truth Extractor\nQwen-VL — specific facts, colors, materials]
    PTE --> CA[Concept Agent\nQwen-Max — 4 scripts, forced distinct frameworks]

    CA --> HC[Hook-Checker]
    CA --> PC[Pacing-Checker — deterministic timing math]
    CA --> BC[Body-Checker]
    CA --> CC[CTA-Checker]
    CA --> TC[Tone-Checker]
    HC --> MC[Meta-Critic\nweighted aggregate + cross-pollinate merge]
    PC --> MC
    BC --> MC
    CC --> MC
    TC --> MC

    MC -->|merge candidate| CV{Merge Coherence Validator\nindependent re-check}
    CV -->|voice/register fail| CE[Copy Editor\nconstrained seam polish]
    CE --> CV
    CV -->|promise-payoff fail| MC
    CV -->|pass| WIN[winning_script finalized]

    WIN --> TA[Treatment Agent\ndirector persona, color story, per-beat justification]
    TA --> SL[Shot-List Agent\ncamera-literate schema + justification]
    SL --> JV{Justification Validator\ndeterministic quote + truth check}
    JV -->|invalid| SL
    JV -->|valid| BG{Budget Gate}
    BG -->|over cap| SL
    BG -->|ok| VGFO[Fan-out per shot via Send]

    VGFO --> VG[Video-Gen Node\nWan i2v — reference photo required]
    VG -->|hard failure| FB[Ken-Burns Fallback\nffmpeg pan/zoom on source photo]
    VG --> CTY[Continuity Agent\nQwen-VL — drift + identity check]
    CTY -->|drift, retries < 2| VG
    CTY -->|retries exhausted| HR[interrupt: Human Review]
    CTY -->|pass| ASM

    WIN --> VOX[Voiceover + Caption Agent\nQwen TTS — parallel branch]
    VOX --> ASM[Assembly Agent\nffmpeg — stitch + captions + CTA burn-in]
    FB --> ASM
    ASM --> FMT[Format Export\nffmpeg — 9:16 / 1:1 / 16:9]
    FMT --> OSS[(Alibaba OSS)]
    OSS --> FE[Frontend]

    BL[(Budget Ledger)] -.->|live stream| FE
    MC -.->|reasoning trace| FE
    CTY -.->|drift scores| FE

    subgraph Alibaba Cloud
        OSS
        DB[(RDS PostgreSQL\nJobs + LangGraph checkpoints)]
        BL
    end
```

**Solid arrows** are graph edges (control and data flow). **Dashed arrows** are live streaming channels to the frontend via `astream_events` → WebSocket. Diamonds are decision nodes that can loop, pause, or branch. The `interrupt: Human Review` is a genuine LangGraph `interrupt()` — the graph checkpoints, surfaces the shot in the UI, and resumes from the checkpoint on seller response.

---

## Pipeline Walk

| Stage | Node(s) | What happens |
|---|---|---|
| **1 — Ingest** | `Ingest Node` | Validates photos (2–3), stores to OSS, captures optional seller direction (mood words, never-do, freeform) |
| **2 — Product Truths** | `Product Truth Extractor` (Qwen-VL) | Extracts 6–10 specific, non-generic facts from the actual uploaded photos — exact colors, materials, textures, scale cues. Every fact gets a `truth_id` that flows through all downstream nodes |
| **3 — Scripts** | `Concept Agent` (Qwen-Max) | Generates 4 structurally distinct 15–30s ad scripts, each forced to use a different copywriting framework (PAS, AIDA, BAB, Hook-Problem-Solution), hook type, and emotional trigger. Each script must cite ≥ 2 product truth IDs |
| **4 — Critic Chain** | Hook / Pacing / Body / CTA / Tone checkers → `Meta-Critic` | Five specialist checkers score all four scripts in parallel across orthogonal axes (hook strength, timing math, body completion, CTA clarity, tone fit). Meta-Critic cross-pollinates: picks the best-scoring hook, body, and CTA from across all variants and merges them into one script |
| **5 — Merge Validation** | `Merge Coherence Validator` + `Copy Editor` | An independent Qwen-Plus node cold-reads the merged script for voice consistency and promise-payoff match — it never shares context with the Meta-Critic that built it. Voice seam failures route to the Copy Editor (constrained polish only); promise-payoff failures route back to the Meta-Critic for a swap |
| **6 — Treatment** | `Treatment Agent` (Qwen-Plus) | Writes a director's treatment: persona, color story, pacing philosophy, and per-beat justifications each required to cite a verbatim script quote and a product truth ID |
| **7 — Shot List** | `Shot-List Agent` (two Qwen calls) + `Justification Validator` | Call A produces only justifications (script quote + truth ID + treatment ref); a deterministic validator checks them before Call B ever runs. Call B produces camera/composition fields conditioned on the validated justifications. 3–7 shots, no product category field anywhere in the schema |
| **8 — Budget Gate** | `Budget Gate` (deterministic) | Grounding-weighted allocation (hook/CTA shots and truth-grounded shots get more budget). If over cap: priority-ordered deterministic reduce — downgrade resolution, cut lowest-weight shots, redistribute via waterfill. Never re-invokes the Shot-List Agent's LLM calls |
| **9 — Video Gen** | `Video-Gen Node` (Wan 2.6 i2v) + `Ken-Burns Fallback` | Shots fan out in parallel via LangGraph `Send()`. Every shot is image-to-video with the seller's reference photo — never pure text-to-video. Hard API failures degrade immediately to a Ken-Burns pan/zoom on the source photo |
| **10 — Continuity** | `Continuity Agent` (Qwen-VL) | Checks each clip for product-identity drift (early-frame categorical check) and cross-shot style consistency. Up to 2 auto-retries per shot; exhausted retries raise a real `interrupt()` for seller approval |
| **11 — Voiceover** | `Voiceover + Caption Agent` (Qwen TTS) | Runs as a parallel branch from script finalization — doesn't wait on video gen. Produces a synced audio track and caption timing file aligned to beat timestamps |
| **12 — Assembly** | `Assembly Agent` (ffmpeg) | Stitches approved shots in script order, overlays the voiceover, burns captions and CTA text into each shot's reserved `text_overlay_zone`. Deterministic — no LLM call |
| **13 — Export** | `Format Export Node` (ffmpeg) | Recomposes the master cut into 9:16, 1:1, and 16:9 using the pre-reserved text overlay zones. Zero additional video-gen cost |

---

## Agent Roster

| Agent | Model | Role |
|---|---|---|
| Product Truth Extractor | Qwen-VL | Photo → specific facts with truth IDs |
| Concept Agent | Qwen-Max | 4 distinct ad scripts, framework/hook/trigger forced diverse |
| Hook-Checker | Qwen | Score hook specificity and strength, 1–5 |
| Pacing-Checker | Deterministic | Validate timing math: beat sums, word rate, windows |
| Body-Checker | Qwen + deterministic pre-pass | Promise-payoff match, non-redundancy, throughline, trigger fidelity |
| CTA-Checker | Qwen | Single-verb CTA clarity + bridge/transition smoothness |
| Tone-Checker | Qwen | Brand fit + hard `never_do` rejection |
| Meta-Critic | Qwen-Max | Weighted composite + cross-pollinate best hook/body/CTA |
| Merge Coherence Validator | Qwen-Plus | Independent cold read: pacing recheck + voice/POV consistency |
| Copy Editor | Qwen-Plus | Constrained seam polish at flagged stitch points only |
| Treatment Agent | Qwen-Plus | Director's treatment: persona, color story, per-beat justification |
| Shot-List Agent | Qwen-Plus (2 calls) | Call A: justifications only → validated → Call B: camera/composition |
| Continuity Agent | Qwen-VL | Early-frame identity check + drift score per shot |
| Voiceover + Caption Agent | Qwen TTS | Synced audio track + caption timing |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 16, React 19, Tailwind CSS 4 |
| **Backend** | FastAPI (Python 3.10), async WebSocket streaming |
| **Orchestration** | LangGraph — conditional edges, `Send()` fan-out, `interrupt()`, Postgres checkpointer |
| **LLM / Reasoning** | Qwen-Max, Qwen-Plus (DashScope OpenAI-compatible endpoint) |
| **Vision** | Qwen-VL (product truth extraction, continuity drift scoring) |
| **Speech** | Qwen TTS / CosyVoice (DashScope Singapore endpoint) |
| **Video Generation** | Wan 2.6 image-to-video (DashScope US endpoint) |
| **Video Assembly** | ffmpeg (deterministic — no LLM calls) |
| **Database** | Alibaba Cloud RDS PostgreSQL (jobs, budget ledger, LangGraph checkpoints) |
| **Storage** | Alibaba Cloud OSS (photos, shot clips, final videos) |
| **Deployment** | Alibaba Cloud ECS — Docker Compose, GitHub Actions auto-deploy on push to `master` |
| **Realtime** | LangGraph `astream_events` → FastAPI WebSocket → browser |

---

## Key Design Decisions

**Why LangGraph, not CrewAI or AutoGen.** The pipeline is a budget-capped DAG with parallel fan-out, a count-limited retry loop, and resumable checkpointed execution — none of which map cleanly to conversational role-based delegation. LangGraph's `Send()`, conditional edges, `interrupt()`, and Postgres checkpointer are all first-class primitives, not workarounds.

**Script-driven direction, not category-driven.** Every camera/composition choice in the shot list must be justified by a verbatim quote from the actual winning script and a real product truth ID. The `product_category` field does not exist anywhere in the schema — it was removed as the seam through which template-based genericness re-enters the pipeline.

**The Meta-Critic never grades its own merge.** The Merge Coherence Validator is a structurally separate node that receives the Meta-Critic's output but shares no call context with the reasoning that produced it — a cold read by a different model call, not the merge-writer re-confirming its own work.

**Budget allocation is real dollars, not an abstract proxy.** `allocated_budget = duration_sec × rate(resolution_tier)` — Wan pricing is flat per second per resolution. Every per-shot allocation, the running total, and the hard cap are real figures visible on the live dashboard.

**Graceful degradation, never a blocking failure.** Hard video-gen API failures degrade immediately to a Ken-Burns ffmpeg fallback. Identity drift goes to 2 auto-retries then a human `interrupt()`, never an infinite loop. The Budget Gate's reduce pass only cuts (never merges) shots from the already-validated list — no new content re-enters the pipeline at the gate.

---

## Deployment

The system runs on Alibaba Cloud ECS behind Docker Compose, with GitHub Actions auto-deploying on every push to `master`.

```
ECS instance (43.112.113.40)
├── productcut-backend-1   FastAPI + LangGraph  :8000
└── productcut-frontend-1  Next.js              :3000
```

Every push to `master` → GitHub Actions SSHes into ECS → `git pull` → `docker compose -f docker-compose.prod.yml up --build -d`. Build time is fast because Docker layer caching preserves the Python pip and npm install layers between deploys.

Environment secrets (API keys, DB URL, OSS credentials) live in `backend/.env` on the ECS instance and are injected at runtime via `env_file` — never baked into the image.

---

## Live Demo

**[merchant-marquee.com](http://43.112.113.40:3000)** — upload any product photos and try it live.

---

## License

[MIT](LICENSE)
