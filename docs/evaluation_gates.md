# Evaluation Gate Criteria

## Overview

The **Evaluation Gate** decides whether a freshly trained shadow model is allowed to replace the
current production model. All six gates must pass (fail-closed design) — the first failure
short-circuits the rest and the shadow model is archived.

**Philosophy: rejection is a success.** A rejection means the gate correctly prevented an inadequate
model from reaching production. It is logged as expected behaviour, not as a pipeline failure.

Source of truth: [`src/retraining/evaluation_gate.py`](../src/retraining/evaluation_gate.py).

---

## Gate summary

Gates are evaluated **in this order**, and evaluation stops at the first failure:

| # | Gate | Threshold (default) | Purpose |
|---|------|---------------------|---------|
| 1 | Sufficient samples | ≥ 200 evaluation samples | Statistical validity |
| 2 | Minimum label coverage | ≥ 30% labelled | Reliable, unbiased evaluation data |
| 3 | Promotion cooldown | ≥ 7 days since last promotion | Production stability |
| 4 | Metric improvement | F1 ≥ +2.0% | Measurable improvement |
| 5 | Calibration maintained | Brier degradation ≤ 0.01 | Probability accuracy |
| 6 | No segment regression | No segment F1 drop > 5% | Equitable performance |

---

## Gate details

### Gate 1: Sufficient samples (≥ 200)

Reads `shadow_metrics["num_samples"]`. With roughly a 10% positive class, 200 samples yields only
~20 positives — this is the floor below which an F1 comparison is noise.

### Gate 2: Minimum label coverage (≥ 30%)

Not every prediction receives a label, and the ones that do may not be a random sample. Below 30%
coverage the evaluation set is considered too selective to trust.

This gate **fails closed**: if coverage gating is enabled (`min_coverage_pct > 0`) but no
`coverage_stats` are supplied at all, the gate rejects immediately rather than skipping the check.

The gate accepts coverage in any of three shapes — `coverage_pct`, `label_coverage_pct`, or
`coverage_rate` (a 0–1 fraction, multiplied by 100).

### Gate 3: Promotion cooldown (≥ 7 days)

Prevents deployment churn and retraining storms, and leaves time for monitoring to catch problems
with the model that was just promoted.

The cooldown is computed by scanning `decision_*.json` files under
`/app/monitoring/retraining/decisions` for the most recent record with `action == "promote"`. If no
prior promotion exists, the gate passes.

> **Cooldown authority:** `EvaluationGate` is the *sole* enforcer. `ModelPromoter` does not
> re-check it — it trusts the gate's decision.

### Gate 4: Metric improvement (F1 ≥ +2.0%)

```python
f1_improvement_pct = (shadow_f1 - production_f1) / production_f1 * 100
```

Computed upstream by `ModelEvaluator.compare_models()` and passed in as `comparison`. A shadow model
that merely matches production is rejected — the bar is a real improvement, not parity.

### Gate 5: Calibration maintained (Brier degradation ≤ 0.01)

```python
brier_change = shadow_brier - production_brier   # lower Brier is better
```

A model can improve F1 while producing worse-calibrated probabilities. Since downstream credit
decisions use the probability, not just the 0/1 label, calibration is gated separately.

Brier score reference: 0.05 excellent · 0.10 good · 0.15 acceptable · 0.25 ≈ random.

### Gate 6: No segment regression (> 5% F1 drop)

Compares `segment_performance` between production and shadow, flagging any segment whose F1 falls by
more than `max_segment_regression_pct`. This stops a model from improving on average by trading away
performance on a subgroup.

> ⚠️ **Currently inactive by default.** `ModelEvaluator` only computes `segment_performance` when it
> is constructed with `segment_features`, and the Airflow DAG constructs it with none
> (`ModelEvaluator()`). With no segments defined on either side, the gate returns "pass" without
> comparing anything. To activate it, pass segment features when building the evaluator in
> [`airflow/dags/retraining_pipeline.py`](../airflow/dags/retraining_pipeline.py).

Segments that exist in production but are missing from the shadow model are **logged as a warning
and treated as non-blocking** (insufficient shadow data is not the same as a regression).

---

## Where the thresholds live

The thresholds are **not environment variables**. They are constructor arguments, set in
`run_evaluation_gate()` inside
[`airflow/dags/retraining_pipeline.py`](../airflow/dags/retraining_pipeline.py):

```python
gate = EvaluationGate(
    min_f1_improvement_pct=2.0,
    max_brier_degradation=0.01,
    max_segment_regression_pct=5.0,
    min_samples_for_decision=200,
    min_coverage_pct=30.0,
    promotion_cooldown_days=7,
)
```

Changing the policy means editing that call.

### Full constructor signature

```python
EvaluationGate(
    min_f1_improvement_pct: float = 2.0,
    max_brier_degradation: float = 0.01,
    max_segment_regression_pct: float = 5.0,
    min_samples_for_decision: int = 200,
    min_coverage_pct: float = 30.0,
    promotion_cooldown_days: int = 7,
    decisions_path: str = "/app/monitoring/retraining/decisions",
)
```

### Alternative policies

```python
# Conservative (high bar)
EvaluationGate(
    min_samples_for_decision=500,
    min_coverage_pct=50.0,
    promotion_cooldown_days=14,
    min_f1_improvement_pct=5.0,
    max_brier_degradation=0.005,
    max_segment_regression_pct=2.0,
)

# Aggressive (low bar — development only)
EvaluationGate(
    min_samples_for_decision=100,
    min_coverage_pct=10.0,
    promotion_cooldown_days=1,
    min_f1_improvement_pct=1.0,
    max_brier_degradation=0.05,
    max_segment_regression_pct=20.0,
)
```

---

## The one exception: first deployment

If no model is in the `Production` stage, there is nothing to compare against. The DAG detects this
in `evaluate_models_replay()`, sets `is_first_deployment = True`, and the gate is bypassed with an
auto-promotion. Gates only apply to *replacements*.

---

## Decision flow

```
Shadow Model
   │
   ├─ Gate 1  Sufficient samples ───── fail ─┐
   ├─ Gate 2  Label coverage ───────── fail ─┤
   ├─ Gate 3  Promotion cooldown ───── fail ─┤
   ├─ Gate 4  F1 improvement ───────── fail ─┼──► REJECT → archive shadow
   ├─ Gate 5  Calibration ──────────── fail ─┤    (logged as a successful gate operation)
   └─ Gate 6  Segment regression ───── fail ─┘
                   │
              all pass
                   │
                   ▼
            PROMOTE to Production
         (previous version → Archived)
```

---

## What gets recorded

Every decision is written twice, so there is both a queryable record and a portable artifact:

* **PostgreSQL** — a row in `retraining_decisions` (`action` = `train` / `skip` / `promote` /
  `reject`, plus `failed_gate`, `reason`, `f1_improvement_pct`, `brier_change`). See
  [database.md](database.md).
* **JSON on disk** — `monitoring/retraining/decisions/decision_<timestamp>.json`, written by
  `ModelPromoter`. These files are also what Gate 3 reads to compute the cooldown.

---

## FAQ

| Question | Answer |
|---|---|
| Why 30% label coverage? | Large enough to avoid a biased evaluation set, realistic given feedback delays |
| Why a 7-day cooldown? | Gives monitoring time to validate the model that was just promoted |
| Why 2% F1 improvement? | Parity is not worth the risk of a deployment; the bar is a real gain |
| Can the cooldown be overridden? | No. `EvaluationGate` is the sole authority |
| Does drift alone trigger retraining? | No. The DAG currently runs drift detection in **observation-only** mode — it logs drift but only the scheduled/data-sufficiency path proceeds to training |
| What about class imbalance? | F1 handles imbalance; Brier handles probability quality; segments (once enabled) handle group fairness |

---

## Related documentation

* [Architecture overview](architecture.md)
* [Retraining pipeline DAG](../airflow/dags/retraining_pipeline.py)
* [Gate implementation](../src/retraining/evaluation_gate.py)
* [Gate unit tests](../tests/unit/test_evaluation_gate.py)
