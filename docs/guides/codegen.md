# TypeScript Codegen from Python Contracts

## How it works

[`stage_contracts.py`](../../apps/data-pipeline/src/causal_ssm_agent/flows/stage_contracts.py) and the domain models it imports are the single source of truth. The pipeline is:

1. `export_schemas.py` calls `.model_json_schema(mode="serialization")` → `schemas/contracts.json` + `schemas/tools.json`
2. `generate.ts` feeds those through [`json-schema-to-typescript`](https://github.com/bcherny/json-schema-to-typescript) → `src/generated/models.ts` + `src/generated/tools.ts`

Generated files are committed. [`docs:check`](../../package.json) verifies schema/typegen drift, generated docs drift, generated LaTeX images, and markdown lint.

## Running codegen

```bash
bun run docs:codegen # run every docs/codegen writer
bun run docs:check   # verify generated artifacts and markdown
```

Run after any change to `stage_contracts.py` or any Pydantic model it transitively imports.

## Docs LaTeX Images

GitHub math rendering is not reliable for every Markdown context, so docs LaTeX is code-generated into SVG image assets under [`docs/assets/generated/latex`](../assets/generated/latex). The source LaTeX is retained in nearby `docs-latex` metadata comments so the generator can be rerun.

```bash
bun run docs:codegen # rewrite docs math as generated SVG image embeds
bun run docs:check   # verify generated docs and SVG assets are in sync
```

Run [`docs:codegen`](../../package.json) after adding `$...$`, `$$...$$`, `\(...\)`, or `\[...\]` math to `README.md` or files under `docs/`. [`docs:check`](../../package.json) includes the LaTeX drift check before markdownlint.

## Changing the schema

The workflow is always: **edit Python → `bun run docs:codegen` → commit both**.

- **New/changed field**: edit the Pydantic model in `stage_contracts.py` (or the domain model it references).
- **New stage**: add a `Stage<N>Contract` in `stage_contracts.py`, register in `STAGE_CONTRACTS`, add re-export in `index.ts`.
- **New/changed tool**: update the `ToolContract` entry in `stage_contracts.py`.

## File ownership

| File | Source |
|------|--------|
| `src/generated/models.ts` | Generated — do not edit |
| `src/generated/tools.ts` | Generated — do not edit |
| `src/index.ts` | Hand-written re-exports |
| `src/run.ts`, `src/stages.ts` | Hand-written |

## Troubleshooting

- **Optional vs required mismatch**: `_make_defaults_required()` in the export script promotes defaulted fields to required, but nullable fields (`default=None`) stay optional.
- **Spurious named type aliases** (e.g. `type RHat = number`): `stripFieldTitles()` in `generate.ts` strips Pydantic's per-field `title` annotations that cause these.
- **Circular imports**: use deferred imports + `model_rebuild()` (see `schemas_inference.py`).
- **tanstack-table column errors**: cast generated column defs as `ColumnDef<T, unknown>[]`.
