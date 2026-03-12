CREATE TABLE IF NOT EXISTS ads (
    id TEXT PRIMARY KEY,
    brief_id TEXT,
    primary_text TEXT NOT NULL,
    headline TEXT NOT NULL,
    description TEXT NOT NULL,
    cta_button TEXT NOT NULL,
    model_id TEXT,
    temperature REAL,
    generation_seed TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    ad_id TEXT NOT NULL REFERENCES ads(id),
    dimension TEXT NOT NULL,
    score REAL NOT NULL,
    rationale TEXT,
    confidence REAL,
    evaluator_model TEXT,
    eval_mode TEXT CHECK(eval_mode IN ('iteration', 'final')),
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS iterations (
    id TEXT PRIMARY KEY,
    source_ad_id TEXT NOT NULL REFERENCES ads(id),
    target_ad_id TEXT NOT NULL REFERENCES ads(id),
    cycle_number INTEGER NOT NULL,
    action_type TEXT NOT NULL CHECK(action_type IN ('component_fix', 'full_regen')),
    weak_dimension TEXT,
    feedback_prompt TEXT,
    delta_weighted_avg REAL,
    token_cost REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    component TEXT NOT NULL,
    action TEXT NOT NULL,
    rationale TEXT,
    context TEXT,
    agent_id TEXT
);

CREATE TABLE IF NOT EXISTS competitor_ads (
    id TEXT PRIMARY KEY,
    brand TEXT NOT NULL,
    primary_text TEXT,
    headline TEXT,
    cta_button TEXT,
    hook_type TEXT,
    emotional_angle TEXT,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS quality_snapshots (
    id TEXT PRIMARY KEY,
    cycle_number INTEGER NOT NULL,
    avg_weighted_score REAL,
    dimension_averages JSON,
    ads_above_threshold INTEGER,
    total_ads INTEGER,
    token_spend_usd REAL,
    quality_per_dollar REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

DROP TABLE IF EXISTS calibration_runs;
CREATE TABLE IF NOT EXISTS calibration_runs (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_version TEXT NOT NULL,
    alpha_overall REAL,
    spearman_rho REAL,
    mae_clarity REAL,
    mae_learner_benefit REAL,
    mae_cta_effectiveness REAL,
    mae_brand_voice REAL,
    mae_student_empathy REAL,
    mae_pedagogical_integrity REAL,
    ad_count INTEGER,
    passed INTEGER NOT NULL DEFAULT 0,
    details_json TEXT
);
