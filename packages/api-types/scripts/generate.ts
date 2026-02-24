/**
 * Generate TypeScript types from JSON Schema exported by Python.
 *
 * Usage:
 *   cd packages/api-types
 *   bun run scripts/generate.ts
 */

import { compile } from "json-schema-to-typescript";
import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { resolve, dirname } from "path";

const ROOT = dirname(dirname(resolve(import.meta.filename)));
const SCHEMA_PATH = resolve(ROOT, "schemas", "contracts.json");
const OUTPUT_PATH = resolve(ROOT, "src", "generated", "models.ts");

/**
 * Reduce a schema node to just its `$ref` if it has one.
 *
 * Pydantic emits `{"$ref": "#/$defs/Foo", "description": "..."}` for
 * fields with doc-strings. The sibling `description` (or `title`, `default`,
 * etc.) next to `$ref` causes json-schema-to-typescript to treat it as a
 * distinct anonymous type — generating duplicates like `LatentModel1`.
 * Per JSON Schema 2020-12, `$ref` siblings are valid but the TS codegen
 * library doesn't handle them well, so we strip them.
 */
function collapseRefs(schema: any): any {
  if (typeof schema !== "object" || schema === null) return schema;
  if (Array.isArray(schema)) return schema.map(collapseRefs);

  // If this object has a $ref, keep only the $ref
  if ("$ref" in schema) {
    return { $ref: schema.$ref };
  }

  const result: any = {};
  for (const [key, value] of Object.entries(schema)) {
    result[key] = collapseRefs(value);
  }
  return result;
}

/**
 * Strip field-level "title" from JSON Schema properties.
 *
 * Pydantic adds "title": "Field Name" to every field, which causes
 * json-schema-to-typescript to generate named type aliases for each
 * field (e.g., `type RHat = number | number[]`). This makes the
 * generated types hard to use with generic TS libraries like tanstack-table.
 *
 * We keep titles on top-level $defs (the actual model names) but strip
 * them from individual properties.
 */
function stripFieldTitles(schema: any, isTopLevel = true): any {
  if (typeof schema !== "object" || schema === null) return schema;

  if (Array.isArray(schema)) {
    return schema.map((item) => stripFieldTitles(item, false));
  }

  const result: any = {};
  for (const [key, value] of Object.entries(schema)) {
    if (key === "properties" && typeof value === "object" && value !== null) {
      // Strip titles from property definitions
      const cleanProps: any = {};
      for (const [propName, propSchema] of Object.entries(value as Record<string, any>)) {
        const cleaned = { ...propSchema };
        delete cleaned.title;
        cleanProps[propName] = stripFieldTitles(cleaned, false);
      }
      result[key] = cleanProps;
    } else if (key === "$defs" && typeof value === "object" && value !== null) {
      // Keep titles on $defs (model-level names) but recurse into their contents
      const cleanDefs: any = {};
      for (const [defName, defSchema] of Object.entries(value as Record<string, any>)) {
        cleanDefs[defName] = stripFieldTitles(defSchema, true);
      }
      result[key] = cleanDefs;
    } else if (key === "items") {
      // Recurse into array items but strip their title
      const cleaned = typeof value === "object" ? { ...value } : value;
      if (typeof cleaned === "object" && cleaned !== null && !isTopLevel) {
        delete cleaned.title;
      }
      result[key] = stripFieldTitles(cleaned, false);
    } else if (key === "anyOf" || key === "oneOf") {
      // Strip titles from union members
      result[key] = (value as any[]).map((item: any) => {
        const cleaned = typeof item === "object" ? { ...item } : item;
        if (typeof cleaned === "object" && cleaned !== null) {
          delete cleaned.title;
        }
        return stripFieldTitles(cleaned, false);
      });
    } else {
      result[key] = value;
    }
  }

  return result;
}

async function main() {
  const rawSchema = JSON.parse(readFileSync(SCHEMA_PATH, "utf-8"));
  const schema = stripFieldTitles(collapseRefs(rawSchema));

  const ts = await compile(schema, "CausalSSMContracts", {
    bannerComment:
      "/* eslint-disable */\n" +
      "/**\n" +
      " * AUTO-GENERATED — DO NOT EDIT\n" +
      " *\n" +
      " * Generated from Python Pydantic models via:\n" +
      " *   cd apps/data-pipeline && uv run python scripts/export_schemas.py\n" +
      " *   cd packages/api-types && bun run scripts/generate.ts\n" +
      " *\n" +
      " * Source of truth: apps/data-pipeline/src/causal_ssm_agent/flows/stages/contracts.py\n" +
      " */",
    additionalProperties: false,
    strictIndexSignatures: true,
    enableConstEnums: false,
    unknownAny: false,
    style: {
      semi: true,
      singleQuote: false,
    },
  });

  mkdirSync(dirname(OUTPUT_PATH), { recursive: true });
  writeFileSync(OUTPUT_PATH, ts);

  // Count interfaces generated
  const count = (ts.match(/export (interface|type)/g) || []).length;
  console.log(`Generated ${count} types/interfaces → ${OUTPUT_PATH}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
