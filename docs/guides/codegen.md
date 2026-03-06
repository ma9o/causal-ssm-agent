# TypeScript Codegen from Python Contracts

Python Pydantic models are the single source of truth for the shape of stage payloads. TypeScript types are generated from them automatically.

## Architecture

```
contracts.py (imports domain models)
     |
     v  .model_json_schema(mode="serialization")
export_schemas.py --> schemas/contracts.json
     |
     v  json-schema-to-typescript
generate.ts --> api-types/src/generated/models.ts
     |
     v  re-exported from
api-types/src/index.ts (+ hand-written stages.ts, run.ts)
```

Generated TS files are committed to git. A CI step runs codegen and checks `git diff` to catch drift.

## Running codegen

From the monorepo root:

```bash
# Full pipeline (export schemas + generate TS)
cd packages/api-types && bun run codegen

# Or step by step:
cd apps/data-pipeline && uv run python scripts/export_schemas.py
cd packages/api-types && bun run scripts/generate.ts
```

Verify everything is in sync:

```bash
bun run codegen:check
# Exits non-zero if generated types differ from committed versions
```

## When to run codegen

Run codegen after any change to:

- `apps/data-pipeline/src/causal_ssm_agent/flows/stages/contracts.py`
- Any domain model imported by contracts (e.g. `schemas.py`, `schemas_model.py`, `schemas_prior.py`, `schemas_inference.py`, `llm.py`, `posterior_predictive.py`, `parametric_id.py`)

## Adding a new field to a stage

1. Add the field to the Pydantic model in contracts.py (or the domain model it references)
2. Run `bun run codegen` from `packages/api-types/`
3. Commit both the Python change and the regenerated `src/generated/models.ts`

## Adding a new stage

1. Create a new `Stage<N>Contract` class in `contracts.py`
2. Register it in `STAGE_CONTRACTS` dict at the bottom of `contracts.py`
3. Run codegen
4. Add re-export alias in `packages/api-types/src/index.ts`

## Adding a new domain model

1. Create the Pydantic model in the appropriate file under `apps/data-pipeline/src/`
2. Reference it from a stage contract (directly or transitively)
3. Run codegen -- Pydantic's `model_json_schema()` automatically includes referenced models in `$defs`
4. If the model needs a frontend-friendly alias, add a re-export in `index.ts`

## What's generated vs hand-written

| File | Source |
|------|--------|
| `src/generated/models.ts` | Generated from Python. Do not edit. |
| `src/index.ts` | Hand-written re-exports with aliases |
| `src/run.ts` | Hand-written (frontend-only types) |
| `src/stages.ts` | Hand-written (stage metadata) |

## CI drift guard

The `codegen:check` task is registered in `turbo.json` with `cache: false`. It runs `bun run codegen` then checks `git diff --exit-code src/generated/`. If the generated output differs from what's committed, the task fails.

When CI fails on `codegen:check`:
1. Run `bun run codegen` locally
2. Review the diff in `src/generated/models.ts`
3. Commit the updated generated file alongside your Python changes

## Troubleshooting

**Field shows as optional in TS but required in Python**: Pydantic marks fields with defaults as optional in JSON Schema. The export script post-processes these via `_make_defaults_required()`, but only for non-nullable fields. If a field has `default=None` and allows `None`, it stays optional.

**Unwanted named type aliases in generated TS**: Pydantic adds `title` annotations to every field, which causes `json-schema-to-typescript` to emit named types like `type RHat = number`. The `stripFieldTitles()` function in `generate.ts` removes these.

**Circular imports between domain models**: Use deferred imports with `model_rebuild()`. See `schemas_inference.py` for an example with `TRuleResult`.

**tanstack-table column type errors**: Generated types may produce column definitions that need explicit casts: `as ColumnDef<T, unknown>[]`.
