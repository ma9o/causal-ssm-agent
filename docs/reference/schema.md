# Schema

## Source Files

- **Orchestrator schemas:** `src/causal_ssm_agent/orchestrator/schemas.py`
  - `Construct` - A theoretical entity in the causal model
  - `CausalEdge` - A directed causal edge between constructs
  - `LatentModel` - Theoretical causal structure (Stage 1a output)
  - `Indicator` - An observed measurement of a construct
  - `MeasurementModel` - Operationalization of constructs (Stage 1b output)
  - `CausalSpec` - Complete causal specification (latent + measurement)

- **Worker schemas:** `src/causal_ssm_agent/workers/schemas.py`
  - `WorkerOutput` - Extraction results from a single chunk
  - `Extraction` - Single indicator value extracted from data

**Terminology:** We avoid "structural" due to SEM/SCM ambiguity. Use measurement/latent for the SEM distinction and functional/topological for SCM.

The schema implements the two-stage pipeline (see [pipeline.md](pipeline.md)): latent model (Stage 1a) defines causal structure, measurement model (Stage 1b) operationalizes constructs into indicators.

---

## Validation Rules

### LatentModel Validation

1. **Exogenous edges:** If `role='exogenous'`, construct cannot appear as `effect` in any edge
2. **Contemporaneous same-scale only:** If `lagged=False`, cause and effect must have identical `temporal_scale`
3. **Single outcome:** Exactly one construct must have `is_outcome=True`
4. **Outcome must be endogenous:** If `is_outcome=True`, then `role='endogenous'`
5. **Time-varying requires granularity:** If `temporal_status='time_varying'`, then `temporal_scale` must be set
6. **Time-invariant forbids granularity:** If `temporal_status='time_invariant'`, then `temporal_scale` must be `None`

### MeasurementModel Validation

1. **Valid construct reference:** Each indicator's `construct_name` must exist in the latent model
2. **Valid aggregation:** `aggregation` must be a key in the aggregation registry

### CausalSpec Validation (cross-model)

1. **Constructs may lack indicators:** A construct can exist without indicators. Whether the target causal effect is identifiable is checked by y0 (Pearl's ID algorithm), not the schema.
2. **Indicator-construct consistency:** All indicators must reference valid constructs

---

## Key Schema Classes

### Construct

```python
class Construct(BaseModel):
    name: str                           # Unique identifier
    description: str                    # What this construct represents
    role: Literal["endogenous", "exogenous"]
    is_outcome: bool = False           # Target of causal query
    temporal_status: Literal["time_varying", "time_invariant"]
    temporal_scale: str | None     # hourly, daily, weekly, monthly, yearly, or None
```

### Indicator

```python
class Indicator(BaseModel):
    name: str                          # Unique identifier
    construct_name: str                # Which construct this measures
    how_to_measure: str                # Extraction instructions
    measurement_dtype: str             # continuous, binary, count, ordinal, categorical
    aggregation: str = "mean"          # How to aggregate to aggregation_window
```

### CausalEdge

```python
class CausalEdge(BaseModel):
    cause: str                         # Source construct name
    effect: str                        # Target construct name
    description: str                   # Causal mechanism
    lagged: bool = True                # False = contemporaneous, True = lagged
```
