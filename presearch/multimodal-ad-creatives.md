# Multimodal Ad Creatives — Image Generation & Visual Evaluation

## Overview

Expand AdCraft from text-only ad copy to full multimodal ad creatives. Generate brand-consistent images for every FB/IG ad using Gemini's native image generation (Nano Banana), evaluate visual quality and copy-image synergy via multimodal LLM-as-judge, and use a cost-optimized quality escalation ladder for image generation. This completes the v2 scope that was cut from the original build.

The pipeline extension is: approved ad copy → quality gate → visual prompt synthesis → flash-image generation → multimodal evaluation → escalate or store. Same SDK, same eval model, same SQLite persistence. No new API keys required.

## Summary

Add image generation to AdCraft using Gemini's native image generation (Nano Banana) via the existing google-genai SDK. Cost-optimized via quality escalation ladder: only ads exceeding a dynamic text quality threshold (starts at 7.0, ratchets up with running average) enter image gen. Flash-image generates first (free tier, 500 RPD). If visual eval fails, escalate to pro-image-preview (one retry). This means most images cost $0 and only quality failures touch the expensive model. Visual evaluation adds 3 new dimensions (brand consistency, composition quality, text-image synergy) scored by Gemini 2.5 Pro multimodal. Images stored as local files with paths in SQLite. Pillow added for image handling. Dashboard extended with image gallery. Imagen 4 available as last-resort fallback via same SDK.

## Features

### v2 Features (Image Generation)

15. **Visual Prompt Engineering** — Gemini 2.5 Flash synthesizes image generation prompts from approved ad copy + brand style guide. Extracts the emotional hook, key visual elements, and brand constraints. Outputs a structured `VisualBrief` with prompt text, negative prompts, aspect ratio, and style references. (`src/generate/visual_prompt.py`, `src/models/creative.py`)

16. **Quality Escalation Image Generation** — Cost-optimized image gen via escalation ladder. Step 1: Only ads exceeding a dynamic text quality threshold enter image gen (threshold starts at 7.0, ratchets up based on running weighted average of passing ads — same quality ratchet pattern as v3 text pipeline). Step 2: Generate with `gemini-2.5-flash-image` (free tier, fast). Step 3: Evaluate with Gemini 2.5 Pro multimodal. Step 4: If visual eval fails threshold, retry once with `gemini-3-pro-image-preview` (premium). Step 5: If still fails, force-fail and log. Most ads never touch the premium model. (`src/generate/image_engine.py`, `src/iterate/visual_healing.py`)

17. **Visual Evaluation Dimensions** — Extend the evaluation framework with 3 new visual dimensions scored independently: **Brand Consistency** (color palette, typography style, visual tone match), **Composition Quality** (layout, focal point, visual hierarchy), **Text-Image Synergy** (does the image reinforce the copy's message and emotional hook?). Multimodal evaluation via Gemini 2.5 Pro with image + copy + rubric in a single call. (`src/evaluate/visual_rubrics.py`, `src/evaluate/engine.py`)

18. **Composed Ad Evaluation** — Final evaluation stage that scores the complete ad unit (copy + image together) for overall coherence and publishability. Separate from individual text and visual scoring. Uses Gemini 2.5 Pro multimodal with the full ad mockup context. (`src/evaluate/composed.py`)

19. **Image Gallery Dashboard Tab** — New Streamlit tab showing generated images alongside their copy, visual prompts, and evaluation scores. Filter by visual score, brand consistency score, image model used. Click to view full evaluation details. (`src/dashboard/app.py` — new tab)

20. **Visual Cost Tracking** — Extend performance-per-token to include image generation costs. Track cost per creative unit (copy generation + image generation + all evaluations). Compare cost-efficiency across image models (flash-image vs. pro-image-preview vs. imagen-4). (`src/analytics/cost.py`, `src/db/queries.py`)

### Cut
- **Flux/Replicate integration** — Adds a new dependency and billing surface for marginal quality gain. Gemini native covers our needs. Revisit only if Gemini image quality proves insufficient.
- **4K resolution** — Overkill for FB/IG ads. 1K is standard, 2K for high-quality finals.
- **Image editing/inpainting** — Multi-turn image editing is supported by the API but adds pipeline complexity. Generate fresh rather than edit.

## Technical Research

### APIs & Services

**Gemini Native Image Gen (Nano Banana)**
- Model IDs: `gemini-2.5-flash-image` (fast, cheap), `gemini-3-pro-image-preview` (quality), `gemini-3.1-flash-image-preview` (newest)
- SDK: `google-genai` (already installed)
- API call:
  ```python
  response = client.models.generate_content(
      model="gemini-2.5-flash-image",
      contents=prompt,
      config=types.GenerateContentConfig(
          response_modalities=["TEXT", "IMAGE"],
          image_config=types.ImageConfig(
              aspect_ratio="1:1",
              image_size="1K"
          )
      )
  )
  # Extract image from response
  for part in response.candidates[0].content.parts:
      if part.inline_data:
          image_bytes = part.inline_data.data
  ```
- Aspect ratios: 1:1, 3:4, 4:3, 9:16, 16:9 (plus 1:4, 1:8, 2:3, 4:1, 4:5, 5:4, 8:1, 21:9)
- Resolution: 512, 1K (default), 2K, 4K
- Reference images: up to 14 (10 objects or 4 characters for style consistency)
- All output includes SynthID watermark (non-removable)
- Auth: same `GEMINI_API_KEY` already in use

**Imagen 4 (Fallback)**
- Model IDs: `imagen-4.0-generate-001` (standard), `imagen-4.0-ultra-generate-001` (ultra), `imagen-4.0-fast-generate-001` (fast)
- SDK: `google-genai` (same SDK, different method)
- API call:
  ```python
  response = client.models.generate_images(
      model="imagen-4.0-generate-001",
      prompt=prompt,
      config=types.GenerateImagesConfig(
          number_of_images=1,
          image_size="1K",
          aspect_ratio="1:1",
          person_generation="allow_adult"
      )
  )
  image_bytes = response.generated_images[0].image.image_bytes
  ```
- 1-4 images per call
- Person generation control: `dont_allow`, `allow_adult`, `allow_all`
- Note: Imagen 3 is shut down. Imagen 4 is current.

**Multimodal Visual Evaluation (Gemini 2.5 Pro)**
- Already our text evaluation model
- For visual eval, pass image + copy + rubric as content parts:
  ```python
  from PIL import Image

  img = Image.open("path/to/generated.png")
  response = client.models.generate_content(
      model="gemini-2.5-pro",
      contents=[
          rubric_prompt,
          img,
          f"Ad copy:\nHeadline: {ad.headline}\nPrimary text: {ad.primary_text}\nCTA: {ad.cta_button}"
      ],
      config=types.GenerateContentConfig(
          response_mime_type="application/json",
          response_schema=VisualEvaluationResult.model_json_schema()
      )
  )
  ```

### Architecture

**Pipeline Extension (Quality Escalation Ladder):**
```
Brief → Generate Copy → Evaluate Copy → [copy score >= dynamic threshold?]
  → Yes → Generate Visual Prompt → Flash-Image generates → Evaluate Visual
    → [visual passes?]
      → Yes → Composed Eval → Store in library
      → No → Pro-Image regenerates (1 retry) → Re-evaluate
        → [passes now?]
          → Yes → Composed Eval → Store
          → No → Force-fail, log reason
  → No (score 7.0-threshold) → Store as text-only ad (still publishable)
  → No (score < 7.0) → Iterate copy (existing flow)
```

**Dynamic threshold**: starts at 7.0 (same as text publishable gate). After each batch, recalculates as max(7.0, running_weighted_avg - 0.5). This means as average quality rises, the bar for image gen rises too. By batch 3+, only top-performing copy gets images.

**Escalation rationale**: flash-image is free tier (500 RPD). Pro-image costs ~$0.04/image. By trying cheap first, we get most images for $0 and only spend on the ~20-30% that fail initial visual eval. This is performance-per-token applied to image gen itself.

**Key decisions:**
- Sequential, not parallel: copy must be approved before image prompt generation. The image should reinforce the approved copy, not diverge from it.
- Quality escalation: flash-image first (free), pro-image only on visual eval failure. Most images cost $0.
- Dynamic threshold gates image gen: starts at 7.0, ratchets up with quality. Demonstrates the v3 quality ratchet applied to image gen entry.
- Visual iteration is simpler than copy iteration: regenerate the image with a modified prompt rather than component-level fixes.

### Patterns
- **Image storage**: Local filesystem at `data/images/{ad_id}_{variant}.png`. Path stored in SQLite, not bytes.
- **Image handling**: Pillow for loading/saving. PIL.Image objects passed directly to Gemini API (SDK handles serialization).
- **Visual prompts**: Structured VisualBrief Pydantic model, not raw strings. Includes negative prompts to prevent off-brand imagery.
- **Cost tracking**: Image gen costs tracked manually (flat rate per model per resolution) since LiteLLM may not support image model cost lookups. Record in same `cost_usd` pattern as text gen.
- **Aspect ratio selection**: Determined by brief's `placement` field — feed posts get 1:1, stories get 9:16, banners get 16:9.

### Schema Changes

**ALTER ads table** (new columns):
```sql
ALTER TABLE ads ADD COLUMN image_path TEXT;
ALTER TABLE ads ADD COLUMN visual_prompt TEXT;
ALTER TABLE ads ADD COLUMN image_model TEXT;
ALTER TABLE ads ADD COLUMN image_cost_usd REAL;
```

**New evaluation dimensions** (stored in existing `evaluations` table):
- `brand_consistency` — visual brand adherence
- `composition_quality` — layout and visual hierarchy
- `text_image_synergy` — copy-image coherence

No new tables needed. The existing `evaluations` table handles visual dimensions naturally — just new dimension values.

### Shared Interfaces

New file: `src/models/creative.py`:
- `VisualBrief` — prompt, negative_prompt, aspect_ratio, resolution, style_refs[], placement (used by features: 15, 16)
- `ImageResult` — image_bytes, file_path, model_id, cost_usd, generation_config (used by features: 16, 19, 20)
- `VisualEvaluationResult` — brand_consistency_score, composition_score, synergy_score, rationales, overall_visual_score (used by features: 17, 18, 19)

Modified: `src/models/ad.py`:
- `AdCopy` gains: `image_path`, `visual_prompt`, `image_model`, `image_cost_usd` (used by features: 16, 19, 20)

Modified: `src/models/evaluation.py`:
- `EvaluationResult` used as-is — visual dimensions are just new `DimensionScore` entries

### Dependencies

**New:**
- `Pillow>=10.0` — image loading, saving, format conversion. Required for passing images to Gemini API and saving generated output.

**Existing (no changes):**
- `google-genai>=1.0` — already supports image generation models
- `anthropic>=0.40` — unchanged
- All other deps unchanged

### Gotchas

1. **Gemini image safety filters are stricter than text.** Educational content about "test anxiety" or "stressed students" may trigger image safety blocks even when text generation succeeds. Mitigation: visual prompts must avoid depicting distress. Focus on positive outcomes (confident students, celebration moments).
2. **SynthID watermarks are permanent.** All Gemini-generated images include invisible watermarks. Not removable. Fine for ad creatives — Meta doesn't reject watermarked images.
3. **Rate limits for image gen may differ from text.** Gemini image models may have separate RPM/RPD limits from text models. Monitor and adjust RateLimiter accordingly. Expect ~10 RPM for image gen on free tier.
4. **Image size on disk.** 1K PNG is ~200-500KB. At 50 ads × 3 variants = 150 images ≈ 50-75MB. Manageable for local storage.
5. **Gemini native image gen returns TEXT + IMAGE.** The response contains both modalities. Must iterate `response.candidates[0].content.parts` and check for `inline_data` to extract the image bytes.
6. **Imagen 4 vs Gemini native are different APIs.** Gemini native uses `generate_content()` with `response_modalities=['TEXT', 'IMAGE']`. Imagen 4 uses `generate_images()`. Different methods, different response shapes.
7. **Pillow dependency.** The google-genai SDK accepts PIL.Image objects directly. Pillow must be installed for this to work.

### Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Image safety filters block valid ad prompts | Med | High | Conservative visual prompts focused on positive outcomes. Retry with softened language. Log all blocks. |
| Generated images look "AI-generated" / uncanny | Med | Med | Use style reference images for consistency. Prefer lifestyle/abstract over photorealistic people. |
| Visual evaluation scores don't correlate with human judgment | High | Med | Calibrate visual rubrics against manually-rated reference images before scaling. Same calibration-first approach as text eval. |
| Image gen rate limits block batch pipeline | Med | Med | Separate rate limiter for image gen. Process images after all copy is approved to batch efficiently. |
| Cost overrun from image generation at scale | Low | Low | Escalation ladder keeps most generation on free tier. Pro-image only fires for ~20-30% of ads. Budget ~$1-2 per 50-ad batch. |

### Cost Estimate

**Development complexity:**
| Feature | Size | Notes |
|---------|------|-------|
| 15. Visual Prompt Engineering | M | Prompt design is the hard part — must capture brand constraints |
| 16. Image Generation Engine | M | Two model integrations (Gemini native + Imagen 4 fallback) |
| 17. Visual Evaluation Dimensions | M | New rubrics, calibration against reference images |
| 18. Composed Ad Evaluation | S | Single multimodal eval call, scoring logic |
| 19. Image Gallery Dashboard | M | New Streamlit tab with image display, comparison view |
| 20. Visual Cost Tracking | S | Extend existing cost tracking with image costs |

**Operational costs per pipeline run (50 ads, free tier):**
| Component | Est. Cost |
|-----------|-----------|
| Visual prompt gen (Gemini 2.5 Flash) | $0 (free tier) |
| Flash-image gen (~35-40 qualifying ads) | $0 (free tier, 500 RPD) |
| Pro-image retry (~8-10 flash failures) | ~$0.30-0.40 |
| Visual evaluation (Gemini 2.5 Pro) | ~$0.50-1.00 |
| Composed evaluation (Gemini 2.5 Pro) | ~$0.25-0.50 |
| **Total image pipeline** | **~$1-2** |
| **Combined with existing text pipeline** | **~$1-7** |

### Deployment

- **Storage directory**: `data/images/` — created by bootstrap, gitignored
- **SDK update**: `uv sync` to pull latest google-genai with image model support
- **New dependency**: `uv add Pillow`
- **No new env vars**: same `GEMINI_API_KEY` covers image models
- **Brand reference images**: stored in `data/reference_images/` — Varsity Tutors brand assets (logo, color swatches, example creatives) for style conditioning

## Environment

- `GEMINI_API_KEY` — same key, covers text and image models (required, already exists)
- No new environment variables needed

## Decisions

- **Quality escalation ladder**: flash-image first (free tier), pro-image only on visual eval failure — performance-per-token applied to image gen itself
- **Dynamic image gen threshold**: starts at 7.0, ratchets up with running average — demonstrates v3 quality ratchet in image pipeline
- **Fallback**: Imagen 4 via same SDK — different API method but no new deps
- **Skip Flux/Replicate**: adds dependency + billing surface for marginal gain. Revisit if Gemini quality insufficient
- **Visual evaluation**: Gemini 2.5 Pro multimodal — already our evaluator, handles image+text natively
- **Image storage**: local files + path in SQLite — simple, no cloud deps
- **Sequential pipeline**: copy approved before image gen — prevents wasted image gen on bad copy
- **Resolution**: 1K default, 2K for finals — FB/IG doesn't benefit from 4K
- **3 visual eval dimensions**: brand consistency, composition, synergy — mirrors the 5 text dimensions approach

## Constraints

- Must use existing google-genai SDK (no new API providers)
- Must integrate with existing SQLite schema (ALTER, don't redesign)
- Must fit existing pipeline state machine pattern (new stages, same architecture)
- Must maintain existing text-only pipeline as working path (image gen is additive)
- Brand reference images must be manually curated (no automated brand asset extraction)
- All generated images include SynthID watermarks (non-negotiable, Google policy)

## Reference

- [Gemini Native Image Generation docs](https://ai.google.dev/gemini-api/docs/image-generation)
- [Imagen 4 via Gemini API](https://ai.google.dev/gemini-api/docs/imagen)
- [Google GenAI Python SDK](https://github.com/googleapis/python-genai)
- [Gemini brand consistency patterns](https://cloud.google.com/transform/closing-the-creative-gap-how-gemini-supports-brand-consistency)
- [Nano Banana Pro for ad creatives](https://blog.google/innovation-and-ai/products/nano-banana-pro/)
