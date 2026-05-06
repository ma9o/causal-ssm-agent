# Code Generation

`bun run docs:codegen` runs two generators:

- **TypeScript contracts**: [`stage_contracts.py`](../../apps/data-pipeline/src/causal_ssm_agent/flows/stage_contracts.py) and the domain models it imports are the source of truth. `export_schemas.py` calls `.model_json_schema(mode="serialization")` → `schemas/contracts.json` + `schemas/tools.json`; `generate.ts` feeds those through [`json-schema-to-typescript`](https://github.com/bcherny/json-schema-to-typescript) → `src/generated/models.ts` + `src/generated/tools.ts`.
- **Docs LaTeX images**: math in `README.md` and `docs/` (`$...$`, `$$...$$`, `\(...\)`, `\[...\]`) is rewritten as SVG embeds under [`docs/assets/generated/latex`](../assets/generated/latex), with source retained in nearby `docs-latex` metadata comments. GitHub math rendering is unreliable across Markdown contexts.

```bash
bun run docs:codegen # regenerate everything
bun run docs:check   # verify drift and lint markdown
```

Generated files are committed. Run `docs:codegen` after editing `stage_contracts.py` (or any Pydantic model it transitively imports) or adding math to `README.md`/`docs/`.

## Changing the schema

Workflow: **edit Python → `bun run docs:codegen` → commit both**.

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
