"""Visual evaluation rubrics and scoring constants for AdCraft.

Defines four visual evaluation dimensions, their rubric text, CoT
multimodal prompt templates, few-shot examples, weights, and JSON
schemas for structured output.

This module is pure text -- no PIL dependency. The engine assembles
the content list (prompt + image + ad copy) for multimodal calls.
"""

from __future__ import annotations

from src.theme import THEME

# -------------------------------------------------------------------
# Dimension weights -- must sum to 1.0
# -------------------------------------------------------------------

VISUAL_DIMENSION_WEIGHTS: dict[str, float] = {
    "brand_consistency": 0.30,
    "composition_quality": 0.225,
    "text_image_synergy": 0.225,
    "instructional_clarity": 0.25,
}

VISUAL_DIMENSIONS: list[str] = list(VISUAL_DIMENSION_WEIGHTS.keys())

# -------------------------------------------------------------------
# Rubric text per visual dimension
# -------------------------------------------------------------------

VISUAL_RUBRICS: dict[str, str] = {
    "brand_consistency": (
        "BRAND CONSISTENCY "
        f"-- Does the image visually align with the {THEME.brand_name} brand?\n"
        "\n"
        f"{THEME.brand_name} brand visual identity: cyan ({THEME.primary_color}), dark navy\n"
        f"({THEME.secondary_color}), accent purple ({THEME.accent_color}), white text,\n"
        "clean sans-serif typography.\n"
        f"Visual tone is {', '.join(THEME.visual_tone)}.\n"
        f"Imagery features {', '.join(THEME.people_descriptors)}, and\n"
        f"{', '.join(THEME.setting_descriptors)}.\n"
        "\n"
        "Criteria:\n"
        "- Color palette adherence (cyan, dark navy, purple -- not neon,\n"
        "  not warm/earthy)\n"
        "- Typography style (clean, modern sans-serif -- not decorative,\n"
        "  not handwritten)\n"
        f"- Visual tone ({', '.join(THEME.visual_tone[:2])} -- not clinical, not\n"
        "  chaotic, not fear-based)\n"
        f"- Brand recognizability (could this be identified as a {THEME.brand_name}\n"
        "  ad?)\n"
        "\n"
        "1-3 (Poor): Generic stock photo with clashing color palette.\n"
        "  No brand signals whatsoever. Warm earthy tones, dark moody tones,\n"
        "  or overly corporate sterile look. Could belong to any company.\n"
        f"  The image actively damages brand perception if used in a {THEME.brand_name} ad.\n"
        "\n"
        "4-6 (Adequate): Neutral palette that doesn't clash but isn't\n"
        f"  distinctly {THEME.brand_name}. Generic educational imagery (chalkboard, books)\n"
        "  without brand-specific visual language. Inoffensive but\n"
        f"  forgettable. A viewer wouldn't associate this with {THEME.brand_name}\n"
        "  specifically.\n"
        "\n"
        "7-8 (Strong): Clear brand alignment -- cyan/dark navy palette present,\n"
        "  clean layout, modern tone. Image features confident students or\n"
        "  supportive learning moments. Professional and polished.\n"
        f"  Recognizably educational and on-brand, though not uniquely {THEME.brand_name}.\n"
        "\n"
        f"9-10 (Exceptional): Distinctly {THEME.brand_name}. Cyan/dark navy/purple palette\n"
        "  dominates, typography is clean and modern, visual tone is\n"
        f"  perfectly {', '.join(THEME.visual_tone[:2])}. Image could only belong\n"
        f"  to {THEME.brand_name}. Builds brand equity with every impression.\n"
        "\n"
        "First describe what you observe in the image (colors, subjects,\n"
        "mood, typography if present), then assess against each criterion\n"
        "above, then score.\n"
        "\n"
        "Do NOT inflate scores. A generic educational image with acceptable\n"
        "colors is a 5-6, not a 7-8. Only images with deliberate,\n"
        "recognizable brand alignment deserve 7+."
    ),
    "composition_quality": (
        "COMPOSITION QUALITY "
        "-- Is the image well-composed for an ad context?\n"
        "\n"
        "This image will appear in a Facebook/Instagram ad viewed at\n"
        "mobile-scroll speed (sub-2-second glance). It must communicate\n"
        "instantly.\n"
        "\n"
        "Criteria:\n"
        "- Layout balance (visual weight distributed intentionally,\n"
        "  not lopsided)\n"
        "- Focal point clarity (one clear element draws the eye first)\n"
        "- Visual hierarchy (viewer's eye follows an intentional path)\n"
        "- Negative space usage (breathing room, not cluttered)\n"
        "- Image quality/sharpness (professional-grade, not\n"
        "  blurry/pixelated/artifacts)\n"
        "\n"
        "1-3 (Poor): Cluttered with no focal point. Multiple competing\n"
        "  elements fight for attention. Poor image quality -- blurry,\n"
        "  pixelated, visible compression artifacts. At mobile-scroll\n"
        "  speed, the viewer registers nothing. Layout feels random\n"
        "  or amateurish.\n"
        "\n"
        "4-6 (Adequate): Identifiable subject but composition doesn't\n"
        "  guide the eye. Acceptable image quality but nothing draws\n"
        "  attention at scroll speed. Layout is functional but not\n"
        "  strategic. The viewer sees 'an image' but isn't compelled\n"
        "  to stop scrolling.\n"
        "\n"
        "7-8 (Strong): Clear focal point, good visual hierarchy. Eye\n"
        "  moves naturally from the key element to supporting details.\n"
        "  Professional image quality. Effective use of negative space.\n"
        "  At scroll speed, the main message registers instantly.\n"
        "\n"
        "9-10 (Exceptional): Ad-ready composition that stops the scroll.\n"
        "  Perfect balance of focal point, hierarchy, and negative space.\n"
        "  Every element is intentionally placed. Professional quality\n"
        "  that would pass creative review at a top agency.\n"
        "\n"
        "First describe what you observe in the image (layout, focal\n"
        "point, spacing, quality), then assess against each criterion\n"
        "above, then score.\n"
        "\n"
        "Do NOT inflate scores. An acceptable but unremarkable composition\n"
        "is a 5-6, not a 7-8. Only genuinely scroll-stopping compositions\n"
        "with clear intentional hierarchy deserve 7+."
    ),
    "text_image_synergy": (
        "TEXT-IMAGE SYNERGY "
        "-- Does the image reinforce the ad copy's message?\n"
        "\n"
        "The image should amplify the copy's message and emotional hook\n"
        "-- not merely illustrate it literally, and certainly not\n"
        "contradict it. The strongest pairings create a unified message\n"
        "that's more powerful than either element alone.\n"
        "\n"
        "Strong synergy: A headline about 'unlocking potential' paired\n"
        "with a confident student mid-achievement -- the image amplifies\n"
        "the emotional hook.\n"
        "Weak synergy: A headline about 'unlocking potential' paired with\n"
        "a literal door/key image -- the image illustrates literally\n"
        "without adding meaning.\n"
        "Anti-synergy: A headline about 'personalized learning' paired\n"
        "with a crowded classroom image -- the image contradicts the copy.\n"
        "\n"
        "Criteria:\n"
        "- Message reinforcement (image supports the copy's core claim)\n"
        "- Emotional alignment (image evokes the same feeling as the copy)\n"
        "- Complementary rather than redundant (image adds meaning,\n"
        "  doesn't just repeat)\n"
        "- Unified narrative (copy + image together tell one coherent\n"
        "  story)\n"
        "\n"
        "1-3 (Poor): Image contradicts or ignores the copy entirely.\n"
        "  Emotional mismatch -- copy is encouraging but image is sterile\n"
        "  or anxiety-inducing. No meaningful connection between what the\n"
        "  viewer reads and what they see. The two elements feel like they\n"
        "  belong to different ads.\n"
        "\n"
        "4-6 (Adequate): Image is thematically related but generic. A\n"
        "  tutoring ad shows a student studying -- technically relevant\n"
        "  but adds no meaning beyond what the copy already says. No\n"
        "  emotional amplification. The image could accompany any\n"
        "  similar ad.\n"
        "\n"
        "7-8 (Strong): Image clearly reinforces the copy's specific\n"
        "  message. Emotional tone aligns -- if the copy is about\n"
        "  confidence, the image shows confidence. The combination is\n"
        "  stronger than either element alone. Viewer gets a coherent\n"
        "  experience.\n"
        "\n"
        "9-10 (Exceptional): Image and copy form an inseparable unit.\n"
        "  The image amplifies the emotional hook in a way that makes\n"
        "  the copy land harder. Remove either element and the ad loses\n"
        "  significant impact. This is creative synergy at a professional\n"
        "  level.\n"
        "\n"
        "First describe what you observe in the image, then read the ad\n"
        "copy provided, then assess how well the two work together against\n"
        "each criterion above, then score.\n"
        "\n"
        "Do NOT inflate scores. A thematically related but generic pairing\n"
        "is a 5-6, not a 7-8. Only pairings where the image genuinely\n"
        "amplifies the copy's specific message deserve 7+."
    ),
    "instructional_clarity": (
        "INSTRUCTIONAL CLARITY "
        "-- Does the image provide cognitive support for learning concepts?\n"
        "\n"
        "Evaluate whether the image reinforces the educational context of\n"
        "the ad. The strongest educational images serve as pedagogical aids\n"
        "themselves, helping the viewer understand or feel the learning\n"
        "experience being offered.\n"
        "\n"
        "Criteria:\n"
        "- Educational relevance (image connects to the learning domain)\n"
        "- Cognitive support (image aids understanding, not just decoration)\n"
        "- Learning moment depiction (captures the process or result of\n"
        "  learning)\n"
        "- Pedagogical authenticity (image reflects real learning, not\n"
        "  staged/generic education stock)\n"
        "\n"
        "1-3 (Poor): Image contradicts or ignores the educational context.\n"
        "  Generic stock photo with no connection to learning or tutoring.\n"
        "  The image could accompany any product ad -- nothing signals\n"
        "  education, learning, or academic growth.\n"
        "\n"
        "4-6 (Adequate): Generic educational imagery -- books, graduation\n"
        "  caps, classrooms -- that signals 'education' without adding\n"
        "  cognitive value. The image says 'this is about school' but\n"
        "  doesn't help the viewer understand the tutoring experience.\n"
        "\n"
        "7-8 (Strong): Image reinforces subject understanding or the\n"
        "  tutoring relationship. Shows learning-in-progress moments,\n"
        "  visible progress indicators, or tutor-student interaction\n"
        "  that demonstrates the pedagogical approach. The viewer can\n"
        "  see what the learning experience looks like.\n"
        "\n"
        "9-10 (Exceptional): Image serves as a pedagogical aid itself.\n"
        "  Visual metaphor for concept mastery, the tutoring relationship,\n"
        "  or the transformation from confusion to understanding. The\n"
        "  image makes the viewer feel the 'Aha!' moment or envision\n"
        "  themselves in a productive learning experience.\n"
        "\n"
        "First describe what you observe in the image, then assess how\n"
        "well it supports the educational message against each criterion\n"
        "above, then score.\n"
        "\n"
        "Do NOT inflate scores. A generic educational image (books on a\n"
        "desk, graduation cap) is a 4-6, not a 7-8. Only images that\n"
        "actively reinforce the learning experience deserve 7+."
    ),
}

# -------------------------------------------------------------------
# Few-shot examples per visual dimension (text descriptions)
# -------------------------------------------------------------------

VISUAL_FEW_SHOT_EXAMPLES: dict[str, str] = {
    "brand_consistency": (
        "Example 1 (Score: 8):\n"
        "Image description: A confident high school student sitting at a\n"
        "clean desk with a laptop, dark navy wall behind them with subtle\n"
        "cyan accent stripe. Cool, modern lighting from the left. Student\n"
        "is smiling while reviewing notes. Clean sans-serif watermark in\n"
        "bottom corner. Color palette is predominantly dark navy, white, and\n"
        "cyan. Modern, tech-forward feel.\n"
        f"Rationale: Strong brand alignment -- the cyan/dark navy palette is\n"
        f"immediately recognizable as {THEME.brand_name}. Clean composition\n"
        "with modern lighting matches the tech-forward, approachable brand\n"
        f"tone. The confident student embodies {THEME.brand_name}'s innovative positioning.\n"
        "Only misses a 9 because the cyan accent is subtle and the overall\n"
        "feel, while polished, doesn't push brand distinction to its peak.\n"
        "Score: 8\n"
        "\n"
        "Example 2 (Score: 3):\n"
        "Image description: A generic stock photo of a crowded classroom\n"
        "with fluorescent lighting. Students look disengaged, slumped in\n"
        "chairs. Walls are beige with a green chalkboard. Neon yellow text\n"
        "overlay says 'TUTORING AVAILABLE.' Color palette is beige, green,\n"
        "yellow -- no cyan or dark navy. The mood is institutional and\n"
        "impersonal.\n"
        "Rationale: Zero brand alignment. The color palette clashes\n"
        f"entirely with {THEME.brand_name}'s cyan/dark navy/white identity. Fluorescent\n"
        f"lighting and disengaged students convey the opposite of {THEME.brand_name}'s\n"
        "tech-forward, approachable tone. The neon text overlay is off-brand\n"
        f"typography. This image would damage brand perception if used in a\n"
        f"{THEME.brand_name} ad.\n"
        "Score: 3"
    ),
    "composition_quality": (
        "Example 1 (Score: 9):\n"
        "Image description: A single student centered in frame, shot from\n"
        "slightly above, with a laptop open in front of them. The student's\n"
        "face is the clear focal point, well-lit against a softly blurred\n"
        "background. Generous negative space on the right side where ad\n"
        "text would overlay. Visual hierarchy: face > laptop screen >\n"
        "background details. Sharp focus, no artifacts, professional\n"
        "lighting with subtle depth of field.\n"
        "Rationale: Excellent ad-ready composition. The student's face is\n"
        "an unambiguous focal point that draws the eye in under a second.\n"
        "Intentional negative space on the right provides perfect text\n"
        "placement area. Visual hierarchy is clean -- the viewer's eye\n"
        "moves naturally from the student to the laptop to the blurred\n"
        "background. Professional image quality throughout. This stops\n"
        "the scroll.\n"
        "Score: 9\n"
        "\n"
        "Example 2 (Score: 2):\n"
        "Image description: A busy image with five students, two tutors,\n"
        "stacks of books, a whiteboard covered in equations, motivational\n"
        "posters on the wall, and a plant in the corner. Every area of the\n"
        "frame is filled with detail. Multiple students are looking in\n"
        "different directions. The image is slightly soft/out-of-focus\n"
        "overall. Lighting is flat.\n"
        "Rationale: No focal point -- five competing subjects with no\n"
        "visual hierarchy. The frame is packed with detail that the eye\n"
        "can't parse at scroll speed. Flat lighting means nothing pops.\n"
        "The slight softness suggests either poor image quality or missed\n"
        "focus. At mobile scroll speed, this registers as visual noise,\n"
        "not communication.\n"
        "Score: 2"
    ),
    "text_image_synergy": (
        "Example 1 (Score: 9):\n"
        "Ad copy: 'Your child's potential isn't limited by a test score.\n"
        "Our expert tutors help students break through plateaus and reach\n"
        "scores they didn't think possible.'\n"
        "Image description: A student pumping their fist in celebration\n"
        "while looking at a laptop screen, with a tutor beside them\n"
        "sharing in the moment. The student's expression is genuine\n"
        "surprise and joy -- a breakthrough moment captured.\n"
        "Rationale: Perfect synergy. The copy talks about 'breaking\n"
        "through plateaus' and reaching unexpected scores -- the image\n"
        "captures exactly that breakthrough moment. The emotional hook\n"
        "(possibility, exceeding expectations) is amplified by the\n"
        "student's genuine surprise and joy. The tutor's presence\n"
        "reinforces 'our expert tutors.' Neither element is redundant --\n"
        "the copy explains the promise while the image shows it realized.\n"
        "Together they form a complete narrative.\n"
        "Score: 9\n"
        "\n"
        "Example 2 (Score: 2):\n"
        "Ad copy: 'Personalized 1-on-1 tutoring tailored to your child's\n"
        "unique learning style. Every session adapts to their pace.'\n"
        "Image description: An aerial shot of a large lecture hall filled\n"
        "with hundreds of students taking an exam. Rows of identical desks\n"
        "stretch into the distance. No individual faces are\n"
        "distinguishable.\n"
        "Rationale: Direct contradiction. The copy emphasizes\n"
        "personalization, 1-on-1 attention, and individual adaptation.\n"
        "The image shows the exact opposite -- mass, impersonal,\n"
        "standardized testing. The emotional tone clashes completely: the\n"
        "copy promises intimacy while the image conveys anonymity. This\n"
        "pairing would undermine the ad's message and confuse the viewer.\n"
        "Score: 2"
    ),
    "instructional_clarity": (
        "Example 1 (Score: 8):\n"
        "Image description: A tutor and student sitting together at a\n"
        "clean desk, working through a math problem on a whiteboard. The\n"
        "student is pointing at a specific step while the tutor nods\n"
        "encouragingly. A progress chart on the wall shows score\n"
        "improvement over time. The lighting is warm and focused on\n"
        "the working area.\n"
        "Rationale: Strong instructional clarity. The image shows a real\n"
        "learning moment -- the student is actively engaged with the\n"
        "material, not just posed. The visible progress chart reinforces\n"
        "the concept of measurable improvement. The tutor-student dynamic\n"
        "is authentic and demonstrates personalized instruction. A viewer\n"
        "can envision themselves in this learning scenario.\n"
        "Score: 8\n"
        "\n"
        "Example 2 (Score: 3):\n"
        "Image description: A stock photo of a graduation cap sitting on\n"
        "top of a stack of textbooks against a plain white background.\n"
        "No people, no learning context, no interaction. The image is\n"
        "clean but generic.\n"
        "Rationale: Zero instructional clarity. A graduation cap on books\n"
        "is the most generic 'education' visual possible -- it signals\n"
        "the category but provides no cognitive support. There's no\n"
        "learning moment, no tutoring relationship, no pedagogical\n"
        "context. This image could accompany any education ad from any\n"
        "company and adds nothing to the viewer's understanding of the\n"
        "tutoring experience.\n"
        "Score: 3"
    ),
}

# -------------------------------------------------------------------
# Prompt templates (multimodal -- image passed separately by engine)
# -------------------------------------------------------------------

VISUAL_SINGLE_DIMENSION_PROMPT = (
    "You are an expert visual advertising evaluator assessing an ad\n"
    f"creative for {THEME.brand_name}, a leading online learning platform.\n"
    "\n"
    "You are evaluating the visual dimension: {dimension_name}\n"
    "\n"
    "RUBRIC:\n"
    "{rubric_text}\n"
    "\n"
    "FEW-SHOT EXAMPLES:\n"
    "{examples}\n"
    "\n"
    "INSTRUCTIONS:\n"
    "The ad image and copy text follow this prompt. Evaluate the image\n"
    "in context of the copy.\n"
    "\n"
    "1. First, describe what you observe in the image -- colors,\n"
    "   subjects, composition, mood, and any text/typography visible.\n"
    "2. Then assess the image against each criterion in the rubric\n"
    "   above, referencing specific visual elements.\n"
    "3. Then provide your score (1-10) based strictly on the rubric.\n"
    "4. Be rigorous. A score of 7+ means the image is genuinely strong\n"
    "   on this dimension. A score of 5-6 means adequate but\n"
    "   unremarkable. Below 5 means clear problems.\n"
    "5. Do NOT inflate scores. Generic or stock-photo-feeling images\n"
    "   should score 4-6 at most. Only distinctive, intentional,\n"
    "   professional visuals deserve 7+.\n"
    "\n"
    "Respond with JSON matching the schema provided."
)

VISUAL_ALL_DIMENSIONS_PROMPT = (
    "You are an expert visual advertising evaluator assessing an ad\n"
    f"creative for {THEME.brand_name}, a leading online learning platform.\n"
    "\n"
    "Evaluate this ad's image across ALL 4 visual dimensions below.\n"
    "For EACH dimension, first describe your observations, then assess\n"
    "against the rubric criteria, then provide a score (1-10).\n"
    "\n"
    "Be rigorous. Generic or stock-photo-feeling images should score\n"
    "4-6 at most. Only distinctive, intentional, professional visuals\n"
    "deserve 7+. Do NOT inflate scores.\n"
    "\n"
    "DIMENSIONS AND RUBRICS:\n"
    "\n"
    "{combined_rubrics}\n"
    "\n"
    "FEW-SHOT EXAMPLES (one per dimension):\n"
    "\n"
    "{combined_examples}\n"
    "\n"
    "INSTRUCTIONS:\n"
    "The ad image and copy text follow this prompt. Evaluate the image\n"
    "in context of the copy.\n"
    "\n"
    "For each dimension, provide your observations and rationale FIRST,\n"
    "then your score.\n"
    "\n"
    "Respond with JSON containing scores for all 4 visual dimensions,\n"
    "matching the schema provided."
)

# -------------------------------------------------------------------
# Response schemas for structured output
# -------------------------------------------------------------------

VISUAL_SINGLE_DIMENSION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "rationale": {
            "type": "string",
            "description": (
                "Observation of image elements, assessment against "
                "rubric criteria, and reasoning behind the score"
            ),
        },
        "score": {
            "type": "number",
            "description": "Score from 1-10 based on the rubric",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in this score from 0.0 to 1.0",
        },
    },
    "required": ["rationale", "score", "confidence"],
}

VISUAL_ALL_DIMENSIONS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        dim: {
            "type": "object",
            "properties": {
                "rationale": {
                    "type": "string",
                    "description": (f"Observation and reasoning for {dim} score"),
                },
                "score": {
                    "type": "number",
                    "description": f"Score from 1-10 for {dim}",
                },
                "confidence": {
                    "type": "number",
                    "description": ("Confidence in this score from 0.0 to 1.0"),
                },
            },
            "required": ["rationale", "score", "confidence"],
        }
        for dim in VISUAL_DIMENSIONS
    },
    "required": VISUAL_DIMENSIONS,
}


# -------------------------------------------------------------------
# Helper: build prompts from templates
# -------------------------------------------------------------------


def build_visual_single_dimension_prompt(dimension: str) -> str:
    """Build a multimodal prompt for evaluating a single visual dimension.

    Returns prompt text only. The image and ad copy text are assembled
    into the content list by the engine, not by this function.
    """
    return VISUAL_SINGLE_DIMENSION_PROMPT.format(
        dimension_name=dimension,
        rubric_text=VISUAL_RUBRICS[dimension],
        examples=VISUAL_FEW_SHOT_EXAMPLES[dimension],
    )


def build_visual_all_dimensions_prompt() -> str:
    """Build a multimodal prompt for evaluating all visual dimensions.

    Returns prompt text only. The image and ad copy text are assembled
    into the content list by the engine, not by this function.
    """
    combined_rubrics = "\n\n".join(
        f"--- {dim.upper()} ---\n{VISUAL_RUBRICS[dim]}" for dim in VISUAL_DIMENSIONS
    )

    combined_examples = "\n\n".join(
        f"--- {dim.upper()} ---\n{VISUAL_FEW_SHOT_EXAMPLES[dim]}" for dim in VISUAL_DIMENSIONS
    )

    return VISUAL_ALL_DIMENSIONS_PROMPT.format(
        combined_rubrics=combined_rubrics,
        combined_examples=combined_examples,
    )
