# Code Generation

Generated API artifacts and generated documentation have separate ownership and commands.

## API Artifacts

`bun run codegen` runs two independent branches:

- **Contracts**: [`artifact_contracts.py`](../../apps/data-pipeline/src/nof1_causal_lab/flows/artifact_contracts.py) and the domain models it imports are the source of truth. `export_schemas.py` writes the JSON schemas; `generate.ts` then feeds them through [`json-schema-to-typescript`](https://github.com/bcherny/json-schema-to-typescript) to write the TypeScript models, tools, tool results, and metadata.
- **Agent API**: `export_agent_api.py` writes the OpenAPI schema and the generated `nof1-episode-api` skill from the FastAPI application.

```bash
bun run codegen       # regenerate API artifacts
bun run codegen:check # verify API artifact drift
```

Generated API files are committed. Run `codegen` after editing `artifact_contracts.py`, any Pydantic model it transitively imports, or the agent API.

## Documentation Artifacts

`bun run docs:codegen` updates the generated distribution reference sections, then rewrites math in `README.md` and `docs/` (`$...$`, `$$...$$`, `\(...\)`, `\[...\]`) as SVG embeds under [`docs/assets/generated/latex`](../assets/generated/latex). The source is retained in nearby `docs-latex` metadata comments because GitHub math rendering is unreliable across Markdown contexts.

```bash
bun run docs:codegen # regenerate documentation artifacts
bun run docs:check   # verify documentation drift, Markdown, and spelling
```

`bun run check` runs both drift checks alongside the repository's lint, type, test, and build tasks.

## Changing the schema

Workflow: **edit Python → `bun run codegen` → commit both**.

- **New/changed field**: edit the Pydantic model in `artifact_contracts.py` (or the domain model it references).
- **New artifact contract**: add the contract class in the owning transition package, register it in `ARTIFACT_CONTRACTS`, add re-export in `index.ts`.
- **New/changed tool**: update the `ToolContract` entry in `artifact_contracts.py`.

## File ownership

| File | Source |
|------|--------|
| `src/generated/models.ts` | Generated — do not edit |
| `src/generated/tools.ts` | Generated — do not edit |
| `src/index.ts` | Hand-written re-exports |
| `src/run.ts`, `src/transitions.ts` | Hand-written |

## Troubleshooting

- **Optional vs required mismatch**: `_make_defaults_required()` in the export script promotes defaulted fields to required, but nullable fields (`default=None`) stay optional.
- **Spurious named type aliases** (e.g. `type RHat = number`): `stripFieldTitles()` in `generate.ts` strips Pydantic's per-field `title` annotations that cause these.
- **Circular imports**: use deferred imports + `model_rebuild()` (see `schemas_inference.py`).
- **tanstack-table column errors**: cast generated column defs as `ColumnDef<T, unknown>[]`.
