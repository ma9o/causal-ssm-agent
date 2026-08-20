import { randomUUID } from "node:crypto";
import { mkdir, rename, writeFile } from "node:fs/promises";
import type { IncomingMessage, ServerResponse } from "node:http";
import { dirname, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { chromium, type Page } from "playwright";
import { logger } from "storybook/internal/node-logger";
import type { ServerApp } from "storybook/internal/types";
import {
  SVG_MATERIALIZER_ENDPOINT,
  SVG_MATERIALIZER_TAG,
  type SvgMaterializerResponse,
} from "./constants.ts";

const MAX_REQUEST_BYTES = 24 * 1024 * 1024;
const STORY_ID = /^[a-z0-9][a-z0-9_-]*$/;
const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");
const OUTPUT_ROOT = resolve(REPO_ROOT, "scratchpad/storybook-svg");
const STORY_INDEX_ATTEMPTS = 50;
const STORY_INDEX_RETRY_MS = 200;
const STORY_RENDER_TIMEOUT_MS = 30_000;

class RequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

interface MaterializationRequest {
  storyId: string;
  svg: string;
}

interface StoryIndexEntry {
  id: string;
  type: string;
  tags?: string[];
}

interface StoryIndex {
  entries: Record<string, StoryIndexEntry>;
}

async function readMaterializationRequest(
  request: IncomingMessage,
): Promise<MaterializationRequest> {
  if (!request.headers["content-type"]?.startsWith("application/json")) {
    throw new RequestError("Expected an application/json request.", 415);
  }

  const chunks: Buffer[] = [];
  let size = 0;
  for await (const rawChunk of request) {
    const chunk = Buffer.isBuffer(rawChunk) ? rawChunk : Buffer.from(rawChunk);
    size += chunk.byteLength;
    if (size > MAX_REQUEST_BYTES) {
      throw new RequestError("Materialized SVG exceeds the 24 MiB request limit.", 413);
    }
    chunks.push(chunk);
  }

  let value: unknown;
  try {
    value = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new RequestError("Request body is not valid JSON.", 400);
  }
  if (typeof value !== "object" || value == null) {
    throw new RequestError("Request body must be an object.", 400);
  }
  const { storyId, svg } = value as Partial<MaterializationRequest>;
  if (typeof storyId !== "string" || !STORY_ID.test(storyId)) {
    throw new RequestError("Story id contains unsupported path characters.", 400);
  }
  if (typeof svg !== "string" || !svg.trimStart().startsWith("<svg")) {
    throw new RequestError("Request does not contain a serialized SVG root.", 400);
  }
  return { storyId, svg };
}

function writeJson(response: ServerResponse, status: number, payload: unknown): void {
  response.statusCode = status;
  response.setHeader("content-type", "application/json; charset=utf-8");
  response.end(JSON.stringify(payload));
}

async function handleMaterialization(
  request: IncomingMessage,
  response: ServerResponse,
): Promise<void> {
  if (request.method !== "POST") {
    response.setHeader("allow", "POST");
    writeJson(response, 405, { error: "Use POST to materialize an SVG." });
    return;
  }

  try {
    const { storyId, svg } = await readMaterializationRequest(request);
    await mkdir(OUTPUT_ROOT, { recursive: true });
    const filename = `${storyId}.svg`;
    const target = resolve(OUTPUT_ROOT, filename);
    const temporary = resolve(OUTPUT_ROOT, `.${filename}.${randomUUID()}.tmp`);
    await writeFile(temporary, svg, "utf8");
    await rename(temporary, target);
    const payload: SvgMaterializerResponse = {
      relativePath: `scratchpad/storybook-svg/${filename}`,
    };
    writeJson(response, 200, payload);
  } catch (error) {
    const status = error instanceof RequestError ? error.status : 500;
    const message = error instanceof Error ? error.message : String(error);
    writeJson(response, status, { error: message });
  }
}

function storybookOrigin(app: ServerApp): string {
  const address = app.server.address();
  if (address == null || typeof address === "string") {
    throw new Error("Storybook did not expose a TCP address for SVG materialization.");
  }
  return `http://127.0.0.1:${address.port}`;
}

async function taggedStoryIds(origin: string): Promise<string[]> {
  let lastError: unknown;
  for (let attempt = 0; attempt < STORY_INDEX_ATTEMPTS; attempt += 1) {
    try {
      const response = await fetch(`${origin}/index.json`);
      if (!response.ok) {
        throw new Error(`Story index request failed (${response.status}).`);
      }
      const index = (await response.json()) as StoryIndex;
      if (typeof index.entries !== "object" || index.entries == null) {
        throw new Error("Storybook returned an invalid story index.");
      }
      const storyIds = Object.values(index.entries)
        .filter((entry) => entry.type === "story" && entry.tags?.includes(SVG_MATERIALIZER_TAG))
        .map((entry) => entry.id)
        .sort();
      if (storyIds.length === 0) {
        throw new Error(`No stories are tagged ${JSON.stringify(SVG_MATERIALIZER_TAG)}.`);
      }
      return storyIds;
    } catch (error) {
      lastError = error;
      await delay(STORY_INDEX_RETRY_MS);
    }
  }
  const message = lastError instanceof Error ? lastError.message : String(lastError);
  throw new Error(`Storybook's SVG materialization index did not become ready: ${message}`);
}

async function materializeStory(page: Page, origin: string, storyId: string): Promise<string> {
  const storyUrl = new URL("/iframe.html", origin);
  storyUrl.searchParams.set("id", storyId);
  storyUrl.searchParams.set("viewMode", "story");
  await page.goto(storyUrl.href, { waitUntil: "domcontentloaded" });
  const control = page.locator("[data-svg-materializer]");
  await control.waitFor({ state: "attached", timeout: STORY_RENDER_TIMEOUT_MS });
  await page.waitForFunction(
    () => {
      const state = document.querySelector<HTMLElement>("[data-svg-materializer]")?.dataset.state;
      return state === "saved" || state === "error";
    },
    undefined,
    { timeout: STORY_RENDER_TIMEOUT_MS },
  );
  const state = await control.getAttribute("data-state");
  const status = await control.getAttribute("data-status");
  if (state !== "saved" || status == null) {
    throw new Error(`${storyId}: ${status ?? "SVG materialization failed without a status."}`);
  }
  return status;
}

async function materializeTaggedStories(app: ServerApp): Promise<void> {
  const origin = storybookOrigin(app);
  const storyIds = await taggedStoryIds(origin);
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1200 } });
    for (const storyId of storyIds) {
      const output = await materializeStory(page, origin, storyId);
      logger.info(`[svg-materializer] ${storyId} -> ${output}`);
    }
  } finally {
    await browser.close();
  }
}

export function installSvgMaterializer(app: ServerApp): ServerApp {
  app.use((request, response, next) => {
    const path = new URL(request.url ?? "/", "http://storybook.local").pathname;
    if (path !== SVG_MATERIALIZER_ENDPOINT) {
      next();
      return;
    }
    void handleMaterialization(request, response);
  });
  app.server.once("listening", () => {
    void materializeTaggedStories(app).catch((error: unknown) => {
      const message = error instanceof Error ? error.stack : String(error);
      logger.error(`[svg-materializer] Startup capture failed:\n${message}`);
    });
  });
  return app;
}
