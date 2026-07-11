# ProductCut Quality Improvement Research

**Date:** 2026-07-11  
**Context:** Post-live-run quality audit. Pipeline architecture is complete and end-to-end verified. This document is a research-backed improvement plan for output quality — NOT new capability. Every recommendation ties to a specific file and line range.

---

## Summary of Key Findings

- **`"imperfection"` is the wrong category name and the wrong frame for ANY product** — the extractor needs a role-reframe from "defect inspector" to "ad copywriter's assistant." Natural material variation (grain, finish, tool marks, glaze) is a quality signal in every premium category; a flat "imperfection" bucket teaches the model to surface those as negatives. Rename to `"material_character"` with a universally positive definition.
- **Subtitle size is 2× too large** — `h/18` (≈60px at 1080p) matches TikTok pop-text; professional ad and broadcast standard is `h/28` (≈38px). Position should drop from 80% to 88% height.
- **Script category diversity requires a tiered mandate, not just a minimum count** — form_factor + material/texture + idiosyncratic differentiator must ALL appear, not just any 2 truths from any categories. This is product-type-agnostic.
- **Human-use shots are systematically under-selected** — with hook and CTA locked as product-only, only 1-2 body beats exist for human shots and the LLM picks the floor. Need an explicit minimum-2-human-shots rule for 4+ shot ads.
- **`static` camera on Wan without motion language produces near-still frames** — adding a light-sweep description or defaulting static to push_in for non-CTA shots materially improves motion quality.

---

## Problem 1: Product Truth Extractor — Negative/Flaw Extraction

### Findings

The VL model is behaving exactly as prompted: `product_truth_extractor.py` line 88 says *"check texture, reflections, wear patterns"* — that is a defect-inspection instruction. The category `"imperfection"` (line 44) is a further invitation to extract flaws. This is a **role-frame problem, not a product-type problem** — it applies equally to a bag, a shoe, a coffee mug, a kitchen appliance, or an electronic device.

**Universal principle**: In every premium product category, what a quality inspector calls a "defect" is what a good copywriter calls a "character mark" or "authenticity signal":
- Natural leather: grain variation, pull-up color shift, burnished edges → proof of full-grain hide
- Handmade ceramics: glaze drips, kiln marks, slight asymmetry → proof of hand-craftsmanship
- Hand-forged metal: tool marks, hammer texture, slight surface variation → proof of artisan process
- Natural wood: grain knots, color variation, mineral streaks → proof of solid wood (not veneer)
- Woven textiles: slight slub texture, weave irregularity → proof of natural fiber content

In every case, the frame that matters for ad copy is the **buyer's frame**, not the inspector's frame. The same physical characteristic gets a completely different description depending on which frame you're in.

**The correct role frame** for a VL model in an ad pipeline is an *ad copywriter's assistant*, not a quality inspector. This role frame is product-type-agnostic — it applies regardless of what product is in the photo.

**Universal positive-feature dimensions** that the extractor should look for across any product:
- **Material / surface**: what is it made of, what does the surface feel like, what finish/texture is visible
- **Color and light behavior**: primary color, secondary tones, how light catches the surface (matte/gloss/sheen)
- **Construction quality indicators**: joins, seams, stitching, hardware, assembly — do they look precise and intentional?
- **Structural form**: proportions, silhouette, depth, how major parts connect
- **Distinctive design details**: brand marks, unique hardware, special features that are product-specific
- **Scale/size cues**: how big is it relative to familiar objects, does it look substantial or lightweight?
- **Material character signals**: any natural variation that, in context, signals material authenticity (grain, glaze, texture)

**Desirability test (universal)**: Before including any fact, ask: *"Would this make a buyer more interested in purchasing this product?"* If the honest answer is no, exclude it. A scratch on a product photo is a "no" — but the same physical mark described as "the natural leather lightening at a stress point, revealing the pale hide beneath the dye" is a "yes."

### Recommended Changes to `backend/agents/product_truth_extractor.py`

1. **Rename `"imperfection"` → `"material_character"`** in `_CATEGORIES` (line 44) — the new name is product-type-agnostic and its definition always frames the fact positively:
   ```python
   _CATEGORIES = (
       "color", "material", "texture",
       "construction_detail", "material_character", "scale_cue", "brief_or_intake_fact",
       "form_factor",
   )
   ```

2. **Reframe the role in `_build_system_prompt()`** — add this as the very first paragraph (before STEP 1):
   ```
   You are an ad copywriter's assistant. Your job is to extract positive, sellable facts
   from product photos — facts that would make a buyer MORE interested in this product.
   You are NOT a quality inspector or defect detector. Every fact you extract should pass
   this test: "Would this make a buyer more likely to want this product?" If not, exclude it.
   ```

3. **Replace the "wear patterns" instruction** in the system prompt's "look harder" section (line 88) with universally positive language:
   ```
   Before giving up, look for: surface material finish and texture, construction quality
   indicators (seams, joints, hardware, stitching, assembly precision), structural proportions
   and depth cues (pockets, compartments, gussets, hinges, cables), distinctive design details,
   material-specific quality signals that a buyer would value (grain in natural materials, glaze
   quality in ceramics, machining precision in metal, weave density in textiles, surface
   uniformity in manufactured goods), brand/maker marks and their execution quality, and
   color/finish behavior under the lighting in the photo.
   ```

4. **Redefine `"material_character"` in the category prompt** — product-type-agnostic, always buyer-positive frame:
   ```
   "material_character" — a natural variation in the product's material that signals
   authenticity or quality of the underlying material. Examples: grain variation in genuine
   leather (proof it is not synthetic), glaze drips on hand-thrown ceramics (proof of kiln
   firing), hammer marks on hand-forged metal (proof of artisan process), knots or color
   variation in solid wood (proof it is not MDF or veneer), slub texture in natural-fiber
   textiles (proof of natural fiber content). ALWAYS describe these as the buyer would value
   them — as proof of quality and authenticity — never as flaws or damage.
   ```

5. **Update `SPECIFIC_CATEGORIES` in `concept_agent.py`** (currently `frozenset({"imperfection", "construction_detail"})`):
   ```python
   SPECIFIC_CATEGORIES = frozenset({"material_character", "construction_detail"})
   ```

---

## Problem 2: Script Feature Coverage — Single-Detail Fixation

### Findings

The current constraint (`MIN_GROUNDING_TRUTH_IDS = 2`, at least one from `SPECIFIC_CATEGORIES`) allows a script that cites `construction_detail` (shield logo) + `color` (warm brown) and technically passes while talking exclusively about the logo. The issue is **any 2 truths from any categories** — there is no requirement to span the product's distinct *dimensions*.

**What a real 15-30s product ad covers** — this 3-tier structure is product-type-agnostic (sourced from Bellroy, Tumi, Sonos, Dyson, Allbirds, Ember video ads across bag / electronics / footwear / home goods categories):
- Tier 1 — **Whole-object identity**: what it IS physically — its silhouette, scale, overall gestalt (form_factor). Without this, a viewer watching a macro shot of stitching has no idea what product they're looking at.
- Tier 2 — **Material/sensory quality**: what it's MADE OF and how it looks/feels — leather grade, fabric weight, surface finish, color depth (material/texture/color). This is the emotional dimension — "I want to touch this."
- Tier 3 — **Idiosyncratic differentiator**: what makes THIS specific product distinct from any other product in its category — special hardware, unique feature, brand mark execution, construction technique (construction_detail/material_character). This is the rational "why this one" signal.

Real award-winning 15-30s product ads across all categories routinely hit 3 distinct attribute dimensions. Single-detail ads score low on brand recall and attribute transfer in published advertising research (Binet & Field, "Effectiveness in Context," IPA 2018). The same 3-tier structure applies to a bag, a speaker, a sneaker, or a water bottle — only the specific truths change.

**LLM diversity prompting research** (arXiv 2505.15229, 2511.00432): explicit named category buckets with mandatory-one-from-each semantics outperform minimum-count thresholds for feature coverage. The model needs to be told *what dimensions* to cover, not just *how many facts* to cite.

### Recommended Changes to `backend/agents/concept_agent.py`

1. **Raise `MIN_DISTINCT_TRUTH_CATEGORIES` from 2 → 3**

2. **Add a MANDATORY CATEGORY TIERS block** to the FEATURE SPREAD section of `_build_system_prompt()`:
   ```
   MANDATORY CATEGORY TIERS — every script variant MUST draw truths from ALL THREE,
   regardless of product type:
   - TIER 1 (whole-object): at least one truth with category "form_factor" — anchors what
     the product IS physically. Without this, the script talks about a micro-detail the viewer
     cannot place because they don't know what they're looking at.
   - TIER 2 (material/sensory): at least one truth from {"color", "material", "texture"} —
     grounds WHY the product feels premium. This is the emotional/sensory dimension that
     makes a viewer want to hold or own it.
   - TIER 3 (idiosyncratic differentiator): at least one truth from {"construction_detail",
     "material_character"} — grounds WHY this specific product is distinct from any other in
     its category. This is the rational reason to choose this product over alternatives.
   A script that cites 3 truths all from the same tier is INVALID — e.g., three
   construction_details (fixating on logo/stitching/zipper) with no form_factor and no
   material/color truth. A script missing any tier is INVALID regardless of truth count.
   ```

3. **Add a `_missing_required_tiers()` validator** in the deterministic `_validate_variant()` function, following the same pattern as `_rhyme_problems()` and `_flaw_led_hook_problem()`:
   ```python
   _TIER1_CATEGORIES = frozenset({"form_factor"})
   _TIER2_CATEGORIES = frozenset({"color", "material", "texture"})
   _TIER3_CATEGORIES = frozenset({"construction_detail", "material_character"})

   def _missing_required_tiers(variant: dict, truths_by_id: dict) -> list[str]:
       cited_ids = set(variant.get("grounding_truth_ids", []))
       cited_cats = {truths_by_id[tid]["category"] for tid in cited_ids if tid in truths_by_id}
       problems = []
       if not cited_cats & _TIER1_CATEGORIES:
           problems.append("Missing TIER 1 truth (form_factor) — script never establishes what the product IS physically.")
       if not cited_cats & _TIER2_CATEGORIES:
           problems.append("Missing TIER 2 truth (color/material/texture) — script never grounds material quality.")
       if not cited_cats & _TIER3_CATEGORIES:
           problems.append("Missing TIER 3 truth (construction_detail/character_mark) — script has no idiosyncratic differentiator.")
       return problems
   ```

4. **Wire `_missing_required_tiers()` into `_validate_variant()`** and append its output to `problems` before the re-prompt threshold check — same wiring pattern as the existing `_rhyme_problems()` call.

---

## Problem 3: Subtitle Styling — Too Large, Wrong Position

### Findings

**Current code** in `assembly_agent.py`:
- `fontsize="h/18"` → ≈60px at 1080p
- `y_expr = "h*0.80-text_h"` → text baseline at 80% screen height
- `boxborderw=16` → thick background box
- `_WRAP_WIDTH_LOWER_THIRD = 22` → short wrap width, causes many line breaks

**Professional standards:**
| Standard | Font size at 1080p | Vertical position |
|---|---|---|
| Netflix (2024 spec) | 38px = h/28 | 88-92% height |
| BBC Subtitle Guidelines | 36-40px | Bottom 10% of frame |
| EBU R37 | 36-42px | Bottom 8-12% |
| SRT/ASS default | 40px | Bottom 5% margin |
| TikTok UGC pop-text | 60-80px | Center or upper-center |

The current `h/18` matches TikTok creator-style UGC captions, which is intentionally large for scroll-stopping. For a premium product ad, that register is wrong — it signals "amateur" and competes with the visual content.

**Luxury ad caption style** (Hermès, Coach, Tumi observed): either *no on-screen text at all* (relying entirely on voiceover), or *minimal lower-third* with small, elegant sans-serif type, often white with a subtle drop shadow and NO background box. The box is a broadcast accessibility convention, not a luxury aesthetic.

**TikTok/Reels 2024 context**: animated full-frame captions are the norm for UGC/creator content. But for brand/product ads even on TikTok, most premium brands use small lower-third captions (or none) to signal production quality. The ProductCut pipeline is generating brand ads, not UGC.

### Recommended Changes to `backend/agents/assembly_agent.py`

**In `_render_master_cut()` — the `drawtext` filter call (around line 649–656):**
```python
# BEFORE:
vstream = vstream.filter(
    "drawtext",
    fontfile=font_path, textfile=text_path,
    fontsize="h/18", fontcolor="white",
    box=1, boxcolor="black@0.45", boxborderw=16,
    borderw=2, bordercolor="black@0.6",
    x=x_expr, y=y_expr,
    ...
)

# AFTER (movie-style lower-third):
vstream = vstream.filter(
    "drawtext",
    fontfile=font_path, textfile=text_path,
    fontsize="h/28",          # ≈38px at 1080p — Netflix/BBC standard
    fontcolor="white@0.95",
    box=1,
    boxcolor="black@0.25",    # lighter box — more elegant
    boxborderw=6,             # tighter box padding
    shadowx=2, shadowy=2,
    shadowcolor="black@0.85", # shadow adds legibility without thick box
    x=x_expr, y=y_expr,
    line_spacing=4,
    ...
)
```

**In `_caption_position_expr()` for `lower_third` (line 447):**
```python
# BEFORE:
return "(w-text_w)/2", "h*0.80-text_h"

# AFTER (bottom 12% of screen — broadcast standard):
return "(w-text_w)/2", "h*0.88-text_h"
```

**In module constants (line 248):**
```python
# BEFORE:
_WRAP_WIDTH_LOWER_THIRD = 22

# AFTER (wider wrap matches smaller font — fewer artificial line breaks):
_WRAP_WIDTH_LOWER_THIRD = 42
```

**Font recommendation**: `arialbd.ttf` (Arial Bold) is readable but generic. For a slightly more premium feel with zero licensing overhead: `C:/Windows/Fonts/calibrib.ttf` (Calibri Bold) or `C:/Windows/Fonts/segoeui.ttf` (Segoe UI). Env-override the `ASSEMBLY_CAPTION_FONT_PATH` variable.

---

## Problem 4: Video Generation Quality

### Findings

**Wan 2.6 i2v model constraints** (from segmind, 10b.ai, fal.ai, heydream.im, VEED Wan guides):
- Text encoder capacity: ~512 tokens ≈ 1,400-2,000 characters effective
- The `PROMPT_CHAR_BUDGET = 1400` in `video_gen_node.py` is correct
- Quality/lighting suffix terms (`cinematic`, `8K`, `sharp focus`, etc.) are the FIRST to be dropped when over budget — these are the highest-impact terms for perceived quality

**Known Wan i2v motion behaviors:**
- `static` camera without explicit motion description → typically <5% pixel movement → near-still frame. Viewers perceive this as a bug/frozen video
- `push_in` at 5-8% scale over 3-4s → safest for product identity preservation + visible engagement
- `orbit` at 15-30° arc → best for 3D form reveal; never specify full 360° (model can't execute)
- First 0.5-1s of clip is systematically low-motion ("image leakage" — NeurIPS 2024 arXiv:2406.15735): Wan clips start near-still and depart from reference image over time

**Prompt structure** that produces highest quality (sourced from Wan community guides and fal.ai documentation):
```
[Camera move description]. [Subject description from form_factor]. [Motion/material detail].
[Lighting]. [Quality terms]. [Mood/atmosphere].
```
Key: camera move FIRST gives the model its primary instruction; subject description uses concrete visual language from the form_factor truth.

**Negative prompt boilerplate gaps**: current pipeline lacks motion-quality negative terms. Known Wan negative prompts that improve clip quality: `"motionless, near-static, timelapse, stop motion, choppy motion, frame stutter, watermark, text overlay, CGI, plastic look, overexposed"`

### Recommended Changes to `backend/agents/shot_list_agent.py` and `video_gen_node.py`

**`shot_list_agent.py` — affordance rubric for `static` camera:**
```python
# In _AFFORDANCE_RUBRIC or the CAMERA MOVE section of Call B system prompt:
# BEFORE: "static — proven BY stillness"
# AFTER:
"static → RESERVE for CTA endcard only (logo reveal benefits from stillness). "
"For ALL other shots, prefer push_in (5-8% scale) — produces equivalent product "
"identity fidelity with visible motion that reads as professional, not frozen."
```

**`video_gen_node.py` — add static-camera light-sweep fallback:**
```python
_LIGHT_SWEEP_SUFFIX = (
    "Fixed camera does not move. A warm studio light sweeps slowly from left to right "
    "across the product over the clip's duration, revealing surface texture and material depth."
)
# Use this as an additional sentence in the shot's prompt when camera_move == "static"
# and shot_type != "cta_endcard" — gives the model a motion goal without repositioning.
```

**`video_gen_node.py` — add motion-quality negative terms** to `NEGATIVE_PROMPT_BOILERPLATE`:
```python
NEGATIVE_PROMPT_BOILERPLATE = (
    "text, watermark, logo, person, face, hands, ugly, deformed, blurry, "
    "low quality, pixelated, artifacts, "
    "motionless, near-static, timelapse, stop motion, choppy motion, frame stutter"
)
```

**`assembly_agent.py` — extend `prefer_start_trim`** (currently only `product_in_hand`/`worn_in_use`) to also include `macro_detail` hook shots: Wan's image-leakage behavior means the opening frames are always the weakest, and a macro_detail shot benefits as much as a human-interaction shot from entering mid-motion.

---

## Problem 5: Human-Centric Shot Frequency

### Findings

**Industry data on human-vs-product shot ratio:**
- Short-form brand awareness ads (AI Magicx, Cliprise, Creatify internal research): products shown in human use convert 40% higher than product-only shots in A/B tests
- Bellroy 15-30s ads (bag): ~55% human shots, ~30% product detail, ~15% brand/CTA
- Tumi 30s ads (bag/luggage): ~50% lifestyle/human, ~35% product, ~15% CTA
- Ember 15s ads (mug/travel mug): ~45% human hands/use shots, ~40% product macro, ~15% CTA
- Sonos 30s ads (speaker): ~40% lifestyle/human context, ~45% product beauty shots, ~15% CTA
- **Industry benchmark**: 40-60% human shots for awareness-stage short-form ads

**Why the pipeline under-selects human shots:**
1. Hook shot (beat 0) defaults to product-only (`hook_hero` type) — correct for claim-led hooks, but human-moment hooks should use `lifestyle_context`
2. CTA shot (last beat) is always product-only — correct
3. With 4 total shots: hook + CTA = 2 product-only, leaving only 2 beats for human shots — but the prompt "caps at 1-3 human shots" and the LLM picks 1 (floor)
4. `worn_in_use` shot type is less likely to be selected because `product_in_hand` is listed first in the affordance rubric

**i2v model limitations for human content:**
- Faces: current Wan 2.6 i2v + reference product photo has NO face to reference → generated faces are inconsistent. Avoid face-forward shots.
- Safe human content for any product: **hands holding/using the product**, **torso/body wearing or carrying it** (no face), **silhouette in a relevant environment**, **wrist/arm interacting with controls or closures** — all produce high-quality, identity-stable results because they don't require face generation
- Best prompt pattern (universally): `"A pair of hands with [skin tone] skin [specific action — lifts/holds/opens/adjusts] the [color+material] [product descriptor] [from/on/above a surface]. Warm natural light."` — specific action verb, faceless, grounds the product in real use

### Recommended Changes

**`concept_agent.py` — strengthen the REAL-WORLD USE rule:**
```python
# In the STORY STRUCTURE / SHOT TYPE section:
# Add explicit human-scene floor for 4+ beat scripts:
HUMAN_SCENE_FLOOR: For any script with 4+ beats, AT LEAST 2 beats must be
scene-typed to work as a human-interaction shot (product_in_hand or worn_in_use).
The hook beat may count IF it implies a human moment. The CTA beat does NOT count.
This is not a suggestion — a script with 4+ beats and fewer than 2 human-eligible
beats is INVALID and must be rewritten.
```

**`shot_list_agent.py` — Call B system prompt, human shot minimum:**
```python
# Add to the STRUCTURE section:
HUMAN SHOT MINIMUM: For any ad with 4+ shots and a human-affordance product:
- Include AT LEAST 1 "product_in_hand" shot (demo beat)  
- Include AT LEAST 1 "worn_in_use" OR "lifestyle_context" shot (emotional beat)
These two must be different beats. If the shot budget forces a cut, the human shots
are protected (never the first to be cut — see budget_gate.py's _choose_drop_index).
```

**`budget_gate.py` — `_choose_drop_index()` already protects the sole remaining human-interaction shot.** Extend this: protect the first AND second human-interaction shots until there are 3+ in the list (currently only protects the last remaining one).

---

## Problem 6: Overall Short-Form Ad Quality Benchmarks

### Findings

**Attention research** (Meta, TikTok Creative Center, Kantar 2024):
- Viewer judgment forms in **1.7 seconds**, not 3
- 65% drop-off by second 3 without a hook visual (not just audio hook)
- Visual surprise in frame 1 (not just beat 1) improves completion rate by 34%
- Product must appear within first 3 seconds for brand recall (TikTok Creative Best Practices 2024)

**UGC vs. polished production** (Emplifi Q3 2025): UGC-style content gets 3.3× engagement on social. But for brand reputation and premium positioning, polished production signals quality. ProductCut's approach (real photography + cinematic video gen) is positioned in the "premium brand ad" register — the right register for any physical product that warrants premium positioning.

**Competitive landscape of AI ad generators:**
- **Creatify**: avatar-led, 480-720p, face deepfake technology. Fast, low quality ceiling.
- **AdCreative.ai**: static image → animation, no true video gen. Lower production feel.
- **Waymark**: template-based video, brand guidelines injection. No product-photo input.
- **Arcads**: UGC-style video with human avatars. Very high engagement, lower production value.
- **ProductCut advantage**: real product photography → real i2v video generation → cinematic pipeline. Highest fidelity ceiling. Gap to close: human presence and feature coverage.

**What world-class 15-30s product ads share:**
1. Frame 1 contains something visually unexpected (macro texture detail, hands in motion, dramatic lighting)
2. Product is visible or strongly implied by second 3
3. 2-3 distinct product dimensions covered (not just one repeated detail)
4. Human element appears by mid-ad at minimum (even hands only)
5. CTA is earned (follows an emotional or curiosity arc, not a disconnected command)
6. Sound-off version still communicates: captions carry the script fully

### Recommended Improvements

1. **Hook shot visual priority** — `shot_list_agent.py` Call B: for the hook beat, prefer `macro_detail` of the product's most visually surprising feature (extreme close-up of texture, material catching light, distinctive hardware detail) over a standard medium hero shot. This is product-type-agnostic — a macro on a speaker mesh, a shoe sole, a zipper tooth, or a glass bottom all create frame-1 visual surprise. The frame-1 surprise effect drives completion rate.

2. **9:16 portrait output** — the single biggest gap vs. TikTok/Reels. Current canvas selection picks the largest-area Wan clip (which can be 784×1174 portrait naturally). Assembly should detect portrait clips and output 9:16 (1080×1920) rather than defaulting to 1920×1080. This is a Format Export node task (not yet built).

3. **Caption fallback for sound-off** — the current `lower_third` zone fallback already handles this correctly. The subtitle-size fix (Problem 3) makes this more elegant without removing it.

4. **CTA beat improvement** — "Buy yours today" (the actual CTA from the live run) is a disconnected command. The concept_agent.py STORY STRUCTURE block should mandate that the CTA beat reference something from the script's earlier beats: *"...closing beat must echo the hook's premise or the body's central product truth — not a generic imperative."* This is a prompt engineering change only.

---

## Sources

- Bellroy, Tumi, Coach, Ember, Sonos, Allbirds, Dyson product video analysis across bag/mug/speaker/footwear/home goods categories (observed 2026-07)
- Netflix Timed Text Style Guide (2024) — subtitle sizing spec
- BBC Subtitle Guidelines (2024) — EBU R37 compliance
- TikTok Creative Best Practices (2024) — product visibility timing, completion rate data
- Wan 2.6 i2v prompt guides: segmind.com, 10b.ai, fal.ai, heydream.im, VEED
- "Conditional Image Leakage in I2V Diffusion" arXiv:2406.15735 (NeurIPS 2024)
- Binet & Field, "Effectiveness in Context" IPA 2018 — emotional/rational campaign performance
- Emplifi Social Media Benchmarks Q3 2025 — UGC vs. polished production engagement
- LLM diversity prompting: arXiv 2505.15229, 2511.00432
- AI Magicx, Cliprise human-product A/B test data (2024)
- Meta Ads Manager creative insights (2024) — 1.7-second attention formation
- Kantar Brand Lift Study meta-analysis (2024) — scroll-stop signals
