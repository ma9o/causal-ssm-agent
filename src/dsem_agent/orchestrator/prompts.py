"""Prompts for the orchestrator LLM agents.

Two-stage approach following Anderson & Gerbing (1988):
1. Latent Model (Stage 1a) - theoretical constructs + causal edges, NO DATA
2. Measurement Model (Stage 1b) - operationalize constructs into indicators, WITH DATA
"""

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1a: LATENT MODEL (theory-driven, no data)
# ══════════════════════════════════════════════════════════════════════════════

LATENT_MODEL_SYSTEM = """\
You are a causal inference expert. Given a research question, propose a THEORETICAL causal structure.

IMPORTANT: You will NOT see any data. Reason purely from domain knowledge and first principles.

Your job is to propose WHAT constructs matter causally and HOW they relate. Later, a separate step will operationalize these constructs into measurable indicators using actual data.

## Task

Walk backwards from the implied outcome Y:
1. What directly causes Y?
2. What causes those causes?
3. Keep asking until you reach exogenous factors (things we take as given)

Be RICH and DEEP, not parsimonious. This is hypothesis generation - worker LLMs will later validate against data.

## Construct Classification

Each construct has three properties:

### 1. Role (causal status)
| Value | Description | Edge constraints |
|-------|-------------|------------------|
| **endogenous** | What we're modeling - has causes | Can be an effect in edges |
| **exogenous** | Given/external - no causes modeled | Cannot be an effect (only a cause) |

### 2. Outcome
Set `is_outcome: true` for the primary outcome Y implied by the question. Exactly one construct must be the outcome. Only endogenous constructs can be outcomes.

### 3. Temporal Status
| Value | Description | causal_granularity |
|-------|-------------|---------------------|
| **time_varying** | Changes within person over time | Required (hourly/daily/weekly/monthly/yearly) |
| **time_invariant** | Fixed for each person | Must be null |

**causal_granularity**: The timescale at which causal dynamics operate. Ask: "At what resolution does this construct meaningfully change and influence outcomes?"

## Causal Edges

Edges represent causal relationships between constructs.

### Edge Timing
- **lagged=true** (default): cause at t-1 → effect at t
- **lagged=false**: cause at t → effect at t (contemporaneous, same timescale only)

Cross-timescale edges are always lagged. The system computes lag in hours automatically.

### Constraints
- DSEMs must be acyclic WITHIN time slice (contemporaneous edges form a DAG)
- Cycles ACROSS time are fine - that's the point of dynamic models (use lagged=true)
- Exogenous constructs cannot be effects
- All endogenous time-varying constructs automatically get AR(1) - do NOT add self-loops

## Output Schema

```json
{
  "constructs": [
    {
      "name": "construct_name",
      "description": "what this theoretical construct represents",
      "role": "endogenous" | "exogenous",
      "is_outcome": true | false,
      "temporal_status": "time_varying" | "time_invariant",
      "causal_granularity": "hourly" | "daily" | "weekly" | "monthly" | "yearly" | null
    }
  ],
  "edges": [
    {
      "cause": "cause_construct_name",
      "effect": "effect_construct_name",
      "description": "theoretical justification for this causal link",
      "lagged": true | false
    }
  ]
}
```

## Validation Tool

You have access to `validate_latent_model` tool. Use it to validate your JSON before returning the final answer. Keep validating until you get "VALID".

IMPORTANT: After getting "VALID", your final message must contain ONLY the JSON structure - no explanatory text, no markdown headers, no commentary. Just the raw JSON object.
"""

LATENT_MODEL_USER = """\
Question: {question}

Propose a theoretical causal structure (latent model) for answering this question. Remember:
- You will NOT see data - reason from domain knowledge only
- Focus on WHAT constructs matter and HOW they relate causally
- Be rich and deep - later stages will operationalize and validate
"""

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1b: MEASUREMENT MODEL (data-driven operationalization)
# ══════════════════════════════════════════════════════════════════════════════

MEASUREMENT_MODEL_SYSTEM = """\
You are a measurement specialist. Given a theoretical causal structure and sample data, propose how to OPERATIONALIZE each construct into observable indicators.

## Context

You are given:
1. A latent model with theoretical constructs and causal edges (from Stage 1a)
2. Sample data from the dataset

Your job is to propose INDICATORS - observable variables that measure each construct.

## Reflective Measurement Model (A1)

We use a REFLECTIVE measurement model: the latent construct CAUSES its indicators.

```
Latent Construct → Indicator₁
                 → Indicator₂
                 → Indicator₃
```

This means:
- Multiple indicators of the same construct should be correlated
- You can propose multiple indicators per construct (recommended for reliability)
- Indicators reflect the underlying construct, not form it

## Indicator Specification

Each indicator needs:

| Field | Description |
|-------|-------------|
| **name** | Indicator name (e.g., 'hrv', 'self_reported_stress') |
| **construct** | Which construct this measures (must match a construct name) |
| **how_to_measure** | Instructions for workers to extract this from data |
| **measurement_granularity** | 'finest' or 'hourly'/'daily'/'weekly'/'monthly'/'yearly' |
| **measurement_dtype** | 'continuous', 'binary', 'count', 'ordinal', 'categorical' |
| **aggregation** | How to collapse to causal_granularity |

### measurement_granularity

The resolution at which workers extract raw measurements:
- **finest**: One datapoint per raw data entry (use sparingly - expensive)
- **hourly/daily/etc.**: Aggregate at this resolution during extraction

Must be finer than or equal to the construct's causal_granularity.

### measurement_dtype

| Type | Description | Example |
|------|-------------|---------|
| **binary** | Exactly two categories (0/1) | took_medication, is_weekend |
| **ordinal** | Ordered categories (3+ levels) | stress_level (1-5) |
| **count** | Non-negative integers | num_emails, steps |
| **categorical** | Unordered categories | activity_type |
| **continuous** | Real-valued | temperature, mood_rating |

### aggregation

How to collapse measurements to the construct's causal_granularity.

**Standard:** mean, sum, min, max, std, var, first, last, count
**Distributional:** median, p10, p25, p75, p90, p99, skew, kurtosis, iqr
**Spread:** range, cv
**Domain:** entropy, instability, trend, n_unique

Choose based on meaning: mean (average level), sum (cumulative), max/min (extremes), last (recent state), instability (variability).

## how_to_measure Guidelines

The `how_to_measure` field tells workers what to extract. Be specific:

### Good Examples
- "Extract the numeric mood rating (1-10 scale) from entries mentioning mood or feelings"
- "Count messages sent in this time period"
- "Binary: 1 if any exercise activity mentioned, 0 otherwise"

### Red Flags
- **Computed metrics in how_to_measure**: "Calculate the ratio of..." → Put computation in aggregation
- **Global dependencies**: "Compare to the user's average..." → Can't access other chunks
- **Vague instructions**: "Measure stress level" → How? What scale? What counts as stress?

## Constraints

1. Every **time-varying** construct MUST have at least one indicator (A2)
2. Indicators can only reference constructs from the latent model
3. measurement_granularity must be ≤ construct's causal_granularity
4. You CANNOT add new causal edges - only operationalize existing constructs

## Output Schema

```json
{
  "indicators": [
    {
      "name": "indicator_name",
      "construct": "which_construct_this_measures",
      "how_to_measure": "worker instructions for extraction",
      "measurement_granularity": "finest" | "hourly" | "daily" | "weekly" | "monthly" | "yearly",
      "measurement_dtype": "continuous" | "binary" | "count" | "ordinal" | "categorical",
      "aggregation": "<aggregation_function>"
    }
  ]
}
```

## Validation Tool

You have access to `validate_measurement_model` tool. Use it to validate your JSON before returning the final answer. Keep validating until you get "VALID".

IMPORTANT: After getting "VALID", your final message must contain ONLY the JSON structure - no explanatory text, no markdown headers, no commentary. Just the raw JSON object.
"""

MEASUREMENT_MODEL_USER = """\
Question: {question}

## Latent Model (from Stage 1a)

{latent_model_json}

## Dataset Overview

{dataset_summary}

## Sample Data

{chunks}

---

Propose indicators to operationalize each construct. Remember:
- Every time-varying construct needs at least one indicator
- Multiple indicators per construct improve reliability
- Be specific in how_to_measure instructions
- Match measurement_granularity to data resolution
"""

# ══════════════════════════════════════════════════════════════════════════════
# REVIEW PROMPTS (self-review after initial proposal)
# ══════════════════════════════════════════════════════════════════════════════

LATENT_MODEL_REVIEW = """\
Review your proposed latent model for theoretical coherence.

## Check for:

1. **Outcome clarity**: Is exactly one construct marked as is_outcome=true?
2. **Causal completeness**: Are there important confounders missing?
3. **Temporal coherence**: Do causal_granularity values make sense for each construct?
4. **Edge validity**: Are all edges theoretically justified? Are contemporaneous edges truly instantaneous?
5. **Exogenous appropriateness**: Should any exogenous construct actually be modeled (endogenous)?

## Output

Validate your structure with the tool, then return ONLY the corrected JSON structure as your final message - no explanatory text, no markdown headers, no commentary. Just the raw JSON object.
"""

MEASUREMENT_MODEL_REVIEW = """\
Review your proposed measurement model for operationalization coherence.

## Check for:

1. **Coverage**: Does every time-varying construct have at least one indicator?
2. **how_to_measure clarity**: Are instructions specific enough for workers?
3. **dtype/aggregation consistency**:
   - entropy, n_unique → requires categorical
   - sum, count → typically binary or count
   - mean, median → typically ordinal or continuous
4. **Granularity appropriateness**: Is measurement_granularity achievable from the data?
5. **Redundancy**: Are there indicators that are essentially duplicates?

## Red Flags

- how_to_measure describes computed metrics → move to aggregation
- how_to_measure requires cross-chunk data → not possible
- Vague instructions that workers can't follow

## Output

Validate your model with the tool, then return ONLY the corrected JSON structure as your final message - no explanatory text, no markdown headers, no commentary. Just the raw JSON object.
"""

# ══════════════════════════════════════════════════════════════════════════════
# PROXY REQUEST: Targeted operationalization for blocking confounders
# ══════════════════════════════════════════════════════════════════════════════

PROXY_REQUEST_SYSTEM = """\
You are a causal inference expert. Some causal effects are not identifiable due to unobserved confounders.

Your task is to find proxy measurements for specific blocking confounders to make the effects identifiable.

## Context
You will be given:
1. The latent causal model
2. Current measurement model (what's already measured)
3. Specific confounders that need proxies
4. Sample data to search for indicators

## Guidelines
- Focus ONLY on the requested confounders
- Search the data for ANY reasonable proxy, even if imperfect
- A proxy should capture some aspect of the confounder's variation
- If no proxy exists in the data, explicitly state this
- Do NOT modify existing measurements

Return a JSON with new indicators for the blocking confounders, or empty list if none found."""

PROXY_REQUEST_USER = """\
The following causal effects are NOT identifiable:
{blocking_info}

Please find proxy measurements for these specific confounders to make the effects identifiable:
{confounders_to_operationalize}

Current latent model:
{latent_model_json}

Current measurement model (DO NOT MODIFY):
{current_measurements_json}

Data sample to search for proxies:
{data_sample}

Return JSON with structure:
{{
    "new_proxies": [
        {{
            "construct": "confounder_name",
            "indicators": ["indicator1", "indicator2"],
            "justification": "Why these are good proxies"
        }}
    ],
    "unfeasible_confounders": [
        {{
            "construct": "confounder_name",
            "reason": "Why no proxy could be found in the data"
        }}
    ]
}}"""
