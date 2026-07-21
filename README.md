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
2. A coordinated crew of AI agents runs the full production chain — extracting product truths from photos, web-researching specs, writing four structurally distinct ad scripts, picking the strongest one, directing the visual approach, generating video shots in parallel, checking every frame for drift, and assembling a final cut with synced voiceover and captions
3. A finished 15–30 second ad is delivered in three aspect ratios (9:16, 1:1, 16:9) with a full transparency breakdown of every creative decision

---

## Architecture

```mermaid
flowchart TD
    IN([Seller · Photos + Brief]) --> BR[Brand Research\nTavily · optional]
    BR --> PTE[Product Truth Extractor\nQwen-VL]
    PTE --> PRN[Product Research\nTavily + Qwen · spec products]
    PRN --> CA[Concept Agent\nQwen-Max · 4 scripts]

    CA --> HC[Hook-Checker\nQwen]
    CA --> PAC[Pacing-Checker\ndeterministic]
    CA --> BC[Body-Checker\nQwen]
    CA --> CC[CTA-Checker\nQwen]
    CA --> TC[Tone-Checker\nQwen]
    HC & PAC & BC & CC & TC --> MC[Meta-Critic\nQwen-Max · picks best variant]

    MC --> VDA[Visual Direction Agent\nQwen-Max]
    MC --> VDE[Voice Direction Agent\nQwen]

    VDA --> TA[Treatment Agent\nQwen-Max]
    TA --> SLA["Shot-List Agent\nQwen-Max · call A → call B"]
    SLA --> BG[Budget Gate\ndeterministic · grounding-weighted]

    BG -->|Send per shot| VG[Video-Gen Node\nWan 2.7 i2v]
    VG --> KB[Ken-Burns Fallback\nffmpeg · pan/zoom]
    KB --> CTY[Continuity Agent\nQwen-VL · identity + drift]
    CTY --> CTG{Continuity Gate}
    CTG -->|retry| VG
    CTG -->|pass| ASM[Assembly Agent\nffmpeg]

    VDE --> VOX[Voiceover + Caption Agent\nQwen TTS]
    VOX --> ASM

    ASM --> FMT[Format Export\nffmpeg · 9:16 / 1:1 / 16:9]
    FMT --> OSS[(Alibaba OSS)]
    OSS --> OUT([merchantmarquee.com])

    BG -.->|budget stream| OUT
    MC -.->|reasoning stream| OUT
    CTY -.->|drift stream| OUT

    subgraph CLOUD[Alibaba Cloud]
        OSS
        DB[(RDS PostgreSQL\njobs · checkpoints)]
    end

    classDef llm fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef det fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef io fill:#fef9c3,stroke:#ca8a04,color:#713f12

    class PTE,PRN,CA,HC,BC,CC,TC,MC,VDA,TA,SLA,CTY,VOX,VG,VDE llm
    class BR,PAC,BG,KB,CTG,ASM,FMT det
    class IN,OUT io
```

**Solid arrows** are graph edges. **Dashed arrows** are live streaming channels via `astream_events` → WebSocket. Diamonds are decision nodes that can loop or branch. The Continuity Gate is a genuine LangGraph conditional edge — retry loops back to Video-Gen; exhausted retries raise a real `interrupt()` for seller approval.

---

## Pipeline Walk

| Stage | Node(s) | What happens |
|---|---|---|
| **1 — Ingest** | `Ingest Node` | Validates photos (2–3), stores to OSS, captures optional seller direction (mood words, never-do constraints, freeform notes) |
| **2 — Brand Research** | `Brand Research` (Tavily) | **Optional.** Fetches and summarises brand identity from the seller's URL. No-op if no URL is provided — the node is always in the graph but skips immediately |
| **3 — Product Truths** | `Product Truth Extractor` (Qwen-VL) | Extracts 6–10 specific facts from uploaded photos — colors, materials, textures, scale cues, form factor. Every fact gets a `truth_id` that flows through all downstream nodes |
| **4 — Product Research** | `Product Research Node` (Tavily + Qwen) | For spec-driven products only: autonomously web-searches for public specs/features and distills up to 10 checkable facts for copy and VO. No-op for appearance-driven products |
| **5 — Scripts** | `Concept Agent` (Qwen-Max) | Generates 4 structurally distinct ad scripts, each using a different copywriting framework (PAS, AIDA, BAB, Hook-Problem-Solution), with every claim grounded to a truth ID |
| **6 — Critic Chain** | Hook / Pacing / Body / CTA / Tone checkers → `Meta-Critic` | Five specialist checkers score all four scripts in parallel. Meta-Critic picks the single highest composite-scoring variant (hook 25%, pacing 20%, completion 20%, CTA 20%, tone 15%) and writes it to `winning_script` |
| **7 — Visual Direction** | `Visual Direction Agent` (Qwen-Max) | Maps each script beat to a shot type, camera move, human-presence judgment, and framing notes — open-world free-form descriptions, not closed enums |
| **8 — Treatment** | `Treatment Agent` (Qwen-Max) | Director's treatment: persona, color story, pacing philosophy, per-beat visual justifications citing verbatim script quotes and truth IDs |
| **9 — Shot List** | `Shot-List Agent` (Qwen-Max, 2 calls) | Call A: beat-to-truth justifications only → deterministically validated. Call B: camera/composition fields conditioned on validated justifications |
| **10 — Budget Gate** | `Budget Gate` (deterministic) | Grounding-weighted allocation. If over cap: priority-ordered reduce — downgrade resolution, cut lowest-weight shots, redistribute via waterfill. No LLM re-invocation |
| **11 — Video Gen** | `Video-Gen Node` (Wan 2.7 i2v) + `Ken-Burns Fallback` | Shots fan out in parallel via LangGraph `Send()`. Every shot is image-to-video with the seller's reference photo. Hard failures degrade to Ken-Burns pan/zoom |
| **12 — Continuity** | `Continuity Agent` (Qwen-VL) + `Continuity Gate` | Early-frame identity check + drift score per shot. Up to 2 auto-retries; exhausted retries raise a real `interrupt()` for seller approval |
| **13 — Voiceover** | `Voice Direction Agent` + `Voiceover + Caption Agent` (Qwen TTS) | Parallel branch from script finalization. Voice Direction rewrites beats for natural spoken delivery with per-beat emotion and pacing. TTS synthesises audio; captions are aligned to beat timestamps |
| **14 — Assembly** | `Assembly Agent` (ffmpeg) | Fan-in join of the video and voiceover branches (deferred until both settle). Stitches approved shots, overlays voiceover, burns captions and CTA into each shot's reserved zone |
| **15 — Export** | `Format Export Node` (ffmpeg) | Recomposes master cut into 9:16, 1:1, and 16:9. Zero additional video-gen cost |

---

## Agent Roster

| Agent | Model | Role |
|---|---|---|
| Brand Research | Tavily | Fetches and summarises brand identity from seller URL — skips if no URL provided |
| Product Truth Extractor | Qwen-VL | Photo → specific facts with truth IDs |
| Product Research Node | Tavily + Qwen | Web-sourced specs/features for spec-driven products (copy/VO only) |
| Concept Agent | Qwen-Max | 4 distinct ad scripts, framework/hook/trigger forced diverse |
| Hook-Checker | Qwen | Score hook specificity and strength, 1–5 |
| Pacing-Checker | Deterministic | Validate timing math: beat sums, word rate, windows |
| Body-Checker | Qwen + deterministic pre-pass | Promise-payoff match, non-redundancy, throughline, trigger fidelity |
| CTA-Checker | Qwen | Single-verb CTA clarity + bridge/transition smoothness |
| Tone-Checker | Qwen | Brand fit + hard `never_do` rejection |
| Meta-Critic | Qwen-Max | Weighted composite score across 5 axes; picks single best-scoring variant |
| Visual Direction Agent | Qwen-Max | Maps beats to shot type, camera move, human presence — open-world free-form |
| Treatment Agent | Qwen-Max | Director's treatment: persona, color story, per-beat justification |
| Shot-List Agent | Qwen-Max (2 calls) | Call A: justifications only → validated → Call B: camera/composition |
| Budget Gate | Deterministic | Grounding-weighted budget allocation and shot pruning |
| Video-Gen Node | Wan 2.7 i2v | Parallel image-to-video generation per shot |
| Ken-Burns Fallback | ffmpeg | Pan/zoom fallback on hard video-gen failures |
| Continuity Agent | Qwen-VL | Early-frame identity check + drift score per shot |
| Continuity Gate | Deterministic | Route: auto-retry → human interrupt → assembly |
| Voice Direction Agent | Qwen | Rewrites beats for spoken delivery; assigns per-beat emotion and pacing |
| Voiceover + Caption Agent | Qwen TTS | Synced audio track + caption timing aligned to beat timestamps |
| Assembly Agent | ffmpeg | Fan-in stitch: shots + voiceover + captions |
| Format Export Node | ffmpeg | Recompose master cut into 9:16, 1:1, 16:9 |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 16, React 19, Tailwind CSS 4 |
| **Backend** | FastAPI (Python 3.10), async WebSocket streaming |
| **Orchestration** | LangGraph — conditional edges, `Send()` fan-out, `interrupt()`, Postgres checkpointer |
| **LLM / Reasoning** | Qwen3.7-Max (DashScope OpenAI-compatible endpoint) |
| **Vision** | Qwen-VL (product truth extraction, continuity drift scoring) |
| **Speech** | Qwen TTS / CosyVoice (DashScope Singapore endpoint) |
| **Video Generation** | Wan 2.7 image-to-video (DashScope US + Singapore endpoints) |
| **Video Assembly** | ffmpeg (deterministic — no LLM calls) |
| **Database** | Alibaba Cloud RDS PostgreSQL (jobs, budget ledger, LangGraph checkpoints) |
| **Storage** | Alibaba Cloud OSS (photos, shot clips, final videos) |
| **Deployment** | Alibaba Cloud ECS — Docker Compose + Nginx, GitHub Actions auto-deploy on push to `master` |
| **Realtime** | LangGraph `astream_events` → FastAPI WebSocket → browser |

---

## Deployment

Runs on Alibaba Cloud ECS behind Nginx + Docker Compose, continuously deployed via GitHub Actions on every push to `master`.

```
ECS instance (43.112.113.40) — merchantmarquee.com
├── Nginx (reverse proxy, SSL termination)
├── merchant-marquee-backend-1   FastAPI + LangGraph  :8000
└── merchant-marquee-frontend-1  Next.js              :3000
```

Every push to `master` → GitHub Actions SSHes into ECS → `git pull` → `docker compose up --build -d`. Environment secrets (API keys, DB URL, OSS credentials) live in `backend/.env` on the ECS instance — never baked into the image.

---

## Live Demo

**[https://merchantmarquee.com](https://merchantmarquee.com)** — upload any product photos and try it live.

---

## License

[MIT](LICENSE)
