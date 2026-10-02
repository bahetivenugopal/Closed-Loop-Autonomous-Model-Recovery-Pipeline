-- ============================================================
-- Phase 4: Production-Grade Hardened Storage Schema (Non-destructive)
-- ============================================================

-- ============================================================
-- 1. Predictions Storage
-- ============================================================
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_version TEXT NOT NULL,
    prediction INT NOT NULL CHECK (prediction IN (0, 1)),
    probability FLOAT NOT NULL CHECK (probability >= 0.0 AND probability <= 1.0),
    application_date TIMESTAMP WITH TIME ZONE,
    features JSONB NOT NULL,
    request_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_application_date ON predictions(application_date DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_request_id ON predictions(request_id);

COMMENT ON TABLE predictions IS 'Durable store of all inference requests and feature vectors for drift analysis & replay.';

-- ============================================================
-- 2. Ground Truth Labels Storage
-- ============================================================
CREATE TABLE IF NOT EXISTS labels (
    prediction_id TEXT PRIMARY KEY REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    true_label INT NOT NULL CHECK (true_label IN (0, 1)),
    label_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    label_source TEXT DEFAULT 'manual',
    days_delayed INT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_labels_timestamp ON labels(label_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_labels_true_label ON labels(true_label);

COMMENT ON TABLE labels IS 'Ground truth labels linked to prediction_id for model evaluation and retraining gates.';

-- ============================================================
-- 3. Time-Series Monitoring Metrics
-- ============================================================
CREATE TABLE IF NOT EXISTS monitoring_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL,

    -- Window context
    lookback_hours INT NOT NULL,
    num_predictions INT NOT NULL,

    -- Proxy metrics (flattened for fast queries)
    positive_rate FLOAT,
    probability_mean FLOAT,
    probability_std FLOAT,
    entropy FLOAT,

    -- Drift summary (dataset-level ONLY)
    dataset_drift_detected BOOLEAN DEFAULT FALSE,
    feature_drift_ratio FLOAT DEFAULT 0.0,
    num_drifted_features INT DEFAULT 0,

    -- References to artifacts
    drift_summary_ref TEXT,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT monitoring_metrics_timestamp_unique UNIQUE (timestamp)
);

CREATE INDEX IF NOT EXISTS idx_monitoring_metrics_timestamp ON monitoring_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_monitoring_metrics_drift ON monitoring_metrics(dataset_drift_detected, feature_drift_ratio);

-- ============================================================
-- 4. Retraining & Decision Log
-- ============================================================
CREATE TABLE IF NOT EXISTS retraining_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL,

    -- Trigger context
    trigger_reason TEXT NOT NULL CHECK (
        trigger_reason IN ('scheduled', 'manual', 'drift_alert')
    ),

    -- Drift context (snapshot at decision time)
    feature_drift_ratio FLOAT,
    num_drifted_features INT,
    dataset_drift_detected BOOLEAN,
    drifted_features TEXT[],

    -- Data context
    labeled_samples INT,
    coverage_pct FLOAT,

    -- Decision outcome
    action TEXT NOT NULL CHECK (
        action IN ('train', 'skip', 'promote', 'reject')
    ),

    -- Gate details
    failed_gate TEXT,
    reason TEXT,

    -- Model context
    shadow_model_version INT,
    production_model_version INT,

    -- Metrics (if promotion decision)
    f1_improvement_pct FLOAT,
    brier_change FLOAT,

    -- References to artifacts
    drift_summary_ref TEXT,
    evaluation_report_ref TEXT,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT retraining_decisions_timestamp_unique UNIQUE (timestamp)
);

CREATE INDEX IF NOT EXISTS idx_retraining_decisions_action ON retraining_decisions(action);
CREATE INDEX IF NOT EXISTS idx_retraining_decisions_trigger ON retraining_decisions(trigger_reason);
CREATE INDEX IF NOT EXISTS idx_retraining_decisions_drift ON retraining_decisions(dataset_drift_detected);

-- ============================================================
-- 5. Model Lineage & Governance
-- ============================================================
CREATE TABLE IF NOT EXISTS model_versions (
    model_name TEXT NOT NULL,
    version INT NOT NULL,
    stage TEXT NOT NULL CHECK (
        stage IN ('Staging', 'Production', 'Archived', 'None')
    ),

    -- Lifecycle timestamps
    trained_at TIMESTAMP,
    promoted_at TIMESTAMP,
    archived_at TIMESTAMP,

    -- Training context
    trigger_reason TEXT,
    training_run_id TEXT,

    -- Performance snapshot
    f1_score FLOAT,
    brier_score FLOAT,
    num_samples INT,

    -- Drift context at training time
    feature_drift_ratio_at_training FLOAT,

    -- Decision reference
    decision_id UUID REFERENCES retraining_decisions(id),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (model_name, version)
);

CREATE INDEX IF NOT EXISTS idx_model_versions_stage ON model_versions(model_name, stage);
CREATE INDEX IF NOT EXISTS idx_model_versions_promoted ON model_versions(promoted_at DESC);

-- Governance invariant: exactly one production model per model_name
CREATE UNIQUE INDEX IF NOT EXISTS one_production_model
ON model_versions(model_name)
WHERE stage = 'Production';

-- ============================================================
-- 6. Views for Joins and Reports
-- ============================================================
CREATE OR REPLACE VIEW v_labeled_predictions AS
SELECT
    p.prediction_id,
    p.timestamp AS prediction_timestamp,
    p.model_version,
    p.prediction,
    p.probability,
    p.application_date,
    p.features,
    p.request_id,
    l.true_label,
    l.label_timestamp,
    l.label_source,
    l.days_delayed
FROM predictions p
INNER JOIN labels l ON p.prediction_id = l.prediction_id;

CREATE OR REPLACE VIEW v_recent_monitoring AS
SELECT
    timestamp,
    num_predictions,
    feature_drift_ratio,
    dataset_drift_detected,
    positive_rate,
    entropy
FROM monitoring_metrics
ORDER BY timestamp DESC
LIMIT 100;

CREATE OR REPLACE VIEW v_decision_history AS
SELECT
    d.timestamp,
    d.action,
    d.trigger_reason,
    d.feature_drift_ratio,
    d.labeled_samples,
    d.coverage_pct,
    d.failed_gate,
    d.reason,
    d.shadow_model_version,
    d.production_model_version
FROM retraining_decisions d
ORDER BY d.timestamp DESC;

CREATE OR REPLACE VIEW v_model_timeline AS
SELECT
    m.model_name,
    m.version,
    m.stage,
    m.trained_at,
    m.promoted_at,
    m.f1_score,
    d.feature_drift_ratio AS drift_at_training,
    d.action
FROM model_versions m
LEFT JOIN retraining_decisions d ON m.decision_id = d.id
ORDER BY m.version DESC;
