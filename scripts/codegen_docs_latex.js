#!/usr/bin/env bun

const { createHash } = require("node:crypto");
const { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } = require("node:fs");
const { dirname, relative, resolve, sep } = require("node:path");

const { mathjax } = require("mathjax-full/js/mathjax.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");
const { SVG } = require("mathjax-full/js/output/svg.js");

const repoRoot = resolve(__dirname, "..");
const docsRoot = resolve(repoRoot, "docs");
const assetRoot = resolve(docsRoot, "assets", "generated", "latex");
const generatedBlockPattern =
  /<!-- docs-latex:start ([A-Za-z0-9_-]+) -->([\s\S]*?)<!-- docs-latex:end -->/g;
const checkOnly = process.argv.includes("--check");
const displayExToPx = 12;

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({ packages: AllPackages });
const svg = new SVG({ fontCache: "none" });
const mathDocument = mathjax.document("", { InputJax: tex, OutputJax: svg });

const renderedAssets = new Map();

function normalizePath(filePath) {
  return filePath.split(sep).join("/");
}

function markdownFilesIn(dir) {
  const files = [];

  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const entryPath = resolve(dir, entry.name);

    if (entry.isDirectory()) {
      files.push(...markdownFilesIn(entryPath));
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      files.push(entryPath);
    }
  }

  return files;
}

function docsMarkdownFiles() {
  return [resolve(repoRoot, "README.md"), ...markdownFilesIn(docsRoot)].sort((a, b) =>
    normalizePath(relative(repoRoot, a)).localeCompare(normalizePath(relative(repoRoot, b))),
  );
}

function isEscaped(text, index) {
  let backslashes = 0;
  for (let cursor = index - 1; cursor >= 0 && text[cursor] === "\\"; cursor -= 1) {
    backslashes += 1;
  }
  return backslashes % 2 === 1;
}

function encodeMeta(meta) {
  return Buffer.from(JSON.stringify({ display: meta.display, latex: meta.latex }), "utf8").toString("base64url");
}

function decodeMeta(encoded) {
  const raw = Buffer.from(encoded, "base64url").toString("utf8");
  const meta = JSON.parse(raw);

  if (typeof meta.display !== "boolean" || typeof meta.latex !== "string") {
    throw new Error("Invalid docs-latex metadata block.");
  }

  return { display: meta.display, latex: meta.latex };
}

function hashMeta(meta) {
  return createHash("sha256").update(encodeMeta(meta)).digest("hex").slice(0, 20);
}

function xmlEscape(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function markdownAlt(meta) {
  const normalized = meta.latex.replace(/\s+/g, " ").trim();
  const clipped = normalized.length > 120 ? `${normalized.slice(0, 117)}...` : normalized;

  return `LaTeX: ${clipped}`.replaceAll("[", "\\[").replaceAll("]", "\\]");
}

function renderSvg(meta) {
  const node = mathDocument.convert(meta.latex, { display: meta.display });
  const svgNode = adaptor.childNodes(node).find((child) => adaptor.kind(child) === "svg");

  if (!svgNode) {
    throw new Error(`MathJax did not produce SVG for: ${meta.latex}`);
  }

  const rawSvg = adaptor.outerHTML(svgNode);
  const title = xmlEscape(markdownAlt(meta));
  const description = xmlEscape(meta.latex);
  const theme = "<style>svg{color:#000}</style>";
  const metadata = `${theme}<title>${title}</title><desc>${description}</desc>`;

  const sizedSvg = meta.display
    ? rawSvg.replace(/\b(width|height)="([0-9.]+)ex"/g, (_, attr, value) => {
        const pixels = Math.ceil(Number.parseFloat(value) * displayExToPx);
        return `${attr}="${pixels}"`;
      })
    : rawSvg;

  return `${sizedSvg.replace(/<svg\b([^>]*)>/, `<svg$1>${metadata}`)}\n`;
}

function assetPathFor(meta) {
  const assetPath = resolve(assetRoot, `${meta.display ? "display" : "inline"}-${hashMeta(meta)}.svg`);
  const svgText = renderSvg(meta);
  const existing = renderedAssets.get(assetPath);

  if (existing !== undefined && existing !== svgText) {
    throw new Error(`Hash collision while rendering ${assetPath}`);
  }

  renderedAssets.set(assetPath, svgText);
  return assetPath;
}

function embedFor(meta, markdownPath) {
  const assetPath = assetPathFor(meta);
  const link = normalizePath(relative(dirname(markdownPath), assetPath));
  const encoded = encodeMeta(meta);
  const alt = markdownAlt(meta);

  if (!meta.display) {
    const image = `![${alt}](<${link}>)`;
    return `<!-- docs-latex:start ${encoded} -->${image}<!-- docs-latex:end -->`;
  }

  return [
    `<!-- docs-latex:start ${encoded} -->`,
    '<p align="center">',
    `  <img src="${link}" alt="${xmlEscape(alt)}">`,
    "</p>",
    "<!-- docs-latex:end -->",
  ].join("\n");
}

function findClosingDelimiter(text, start, delimiter) {
  for (let cursor = start; cursor < text.length; cursor += 1) {
    if (text.startsWith(delimiter, cursor) && !isEscaped(text, cursor)) {
      return cursor;
    }
  }

  return -1;
}

function findInlineDollarClose(text, start) {
  for (let cursor = start; cursor < text.length; cursor += 1) {
    if (text[cursor] === "\n") {
      return -1;
    }

    if (text[cursor] === "$" && !isEscaped(text, cursor) && text[cursor + 1] !== "$") {
      return cursor;
    }
  }

  return -1;
}

function isValidInlineDollarOpen(text, index) {
  const previous = text[index - 1] ?? "";
  const next = text[index + 1] ?? "";

  return (
    next !== "$" &&
    next !== "" &&
    !/\s/.test(next) &&
    !/[0-9]/.test(next) &&
    !/[A-Za-z0-9]/.test(previous)
  );
}

function isValidInlineDollarClose(text, index) {
  const previous = text[index - 1] ?? "";
  const next = text[index + 1] ?? "";

  return previous !== "" && !/\s/.test(previous) && !/[A-Za-z0-9]/.test(next);
}

function replaceLatexInUnprotectedText(text, markdownPath) {
  let output = "";
  let cursor = 0;

  while (cursor < text.length) {
    if (text.startsWith("$$", cursor) && !isEscaped(text, cursor)) {
      const close = findClosingDelimiter(text, cursor + 2, "$$");
      if (close !== -1) {
        const latex = text.slice(cursor + 2, close).trim();
        output += latex ? embedFor({ display: true, latex }, markdownPath) : text.slice(cursor, close + 2);
        cursor = close + 2;
        continue;
      }
    }

    if (text.startsWith("\\[", cursor) && !isEscaped(text, cursor)) {
      const close = findClosingDelimiter(text, cursor + 2, "\\]");
      if (close !== -1) {
        const latex = text.slice(cursor + 2, close).trim();
        output += latex ? embedFor({ display: true, latex }, markdownPath) : text.slice(cursor, close + 2);
        cursor = close + 2;
        continue;
      }
    }

    if (text.startsWith("\\(", cursor) && !isEscaped(text, cursor)) {
      const close = findClosingDelimiter(text, cursor + 2, "\\)");
      if (close !== -1) {
        const latex = text.slice(cursor + 2, close).trim();
        output += latex ? embedFor({ display: false, latex }, markdownPath) : text.slice(cursor, close + 2);
        cursor = close + 2;
        continue;
      }
    }

    if (text[cursor] === "$" && !isEscaped(text, cursor) && isValidInlineDollarOpen(text, cursor)) {
      const close = findInlineDollarClose(text, cursor + 1);
      if (close !== -1 && isValidInlineDollarClose(text, close)) {
        const latex = text.slice(cursor + 1, close).trim();
        output += latex ? embedFor({ display: false, latex }, markdownPath) : text.slice(cursor, close + 1);
        cursor = close + 1;
        continue;
      }
    }

    output += text[cursor];
    cursor += 1;
  }

  return output;
}

function collectFenceRanges(text) {
  const ranges = [];
  let offset = 0;
  let openFence = null;

  for (const line of text.match(/[^\n]*(?:\n|$)/g) ?? []) {
    if (line === "") {
      break;
    }

    const match = /^( {0,3})(`{3,}|~{3,})/.exec(line);
    if (match) {
      const marker = match[2];
      if (openFence === null) {
        openFence = { char: marker[0], length: marker.length, start: offset };
      } else if (marker[0] === openFence.char && marker.length >= openFence.length) {
        ranges.push({ start: openFence.start, end: offset + line.length });
        openFence = null;
      }
    }

    offset += line.length;
  }

  if (openFence !== null) {
    ranges.push({ start: openFence.start, end: text.length });
  }

  return ranges;
}

function isInsideRanges(index, ranges) {
  return ranges.some((range) => index >= range.start && index < range.end);
}

function collectInlineCodeRanges(text, protectedRanges) {
  const ranges = [];
  let cursor = 0;

  while (cursor < text.length) {
    if (text[cursor] !== "`" || isInsideRanges(cursor, protectedRanges)) {
      cursor += 1;
      continue;
    }

    let length = 1;
    while (text[cursor + length] === "`") {
      length += 1;
    }

    let close = cursor + length;
    while (close < text.length) {
      if (
        text.startsWith("`".repeat(length), close) &&
        !isInsideRanges(close, protectedRanges)
      ) {
        ranges.push({ start: cursor, end: close + length });
        cursor = close + length;
        break;
      }
      close += 1;
    }

    if (close >= text.length) {
      cursor += length;
    }
  }

  return ranges;
}

function mergeRanges(ranges) {
  const sorted = [...ranges].sort((a, b) => a.start - b.start || a.end - b.end);
  const merged = [];

  for (const range of sorted) {
    const previous = merged.at(-1);
    if (previous && range.start <= previous.end) {
      previous.end = Math.max(previous.end, range.end);
    } else {
      merged.push({ ...range });
    }
  }

  return merged;
}

function replaceLatexOutsideCode(text, markdownPath) {
  const fenceRanges = collectFenceRanges(text);
  const inlineCodeRanges = collectInlineCodeRanges(text, fenceRanges);
  const protectedRanges = mergeRanges([...fenceRanges, ...inlineCodeRanges]);
  let output = "";
  let cursor = 0;

  for (const range of protectedRanges) {
    output += replaceLatexInUnprotectedText(text.slice(cursor, range.start), markdownPath);
    output += text.slice(range.start, range.end);
    cursor = range.end;
  }

  output += replaceLatexInUnprotectedText(text.slice(cursor), markdownPath);
  return output;
}

function processMarkdown(text, markdownPath) {
  let output = "";
  let cursor = 0;
  let match;

  generatedBlockPattern.lastIndex = 0;
  while ((match = generatedBlockPattern.exec(text)) !== null) {
    output += replaceLatexOutsideCode(text.slice(cursor, match.index), markdownPath);
    output += embedFor(decodeMeta(match[1]), markdownPath);
    cursor = match.index + match[0].length;
  }

  output += replaceLatexOutsideCode(text.slice(cursor), markdownPath);
  return output;
}

function syncAssets() {
  const changedAssets = [];
  if (!checkOnly) {
    mkdirSync(assetRoot, { recursive: true });
  }

  for (const [assetPath, svgText] of renderedAssets.entries()) {
    const existing = existsSync(assetPath) ? readFileSync(assetPath, "utf8") : null;
    if (existing !== svgText) {
      changedAssets.push(normalizePath(relative(repoRoot, assetPath)));
      if (!checkOnly) {
        writeFileSync(assetPath, svgText);
      }
    }
  }

  if (!existsSync(assetRoot)) {
    return changedAssets.sort();
  }

  for (const entry of readdirSync(assetRoot, { withFileTypes: true })) {
    if (!entry.isFile() || !entry.name.endsWith(".svg")) {
      continue;
    }

    const assetPath = resolve(assetRoot, entry.name);
    if (!renderedAssets.has(assetPath)) {
      changedAssets.push(normalizePath(relative(repoRoot, assetPath)));
      if (!checkOnly) {
        rmSync(assetPath);
      }
    }
  }

  return changedAssets.sort();
}

function main() {
  const changedDocs = [];

  for (const markdownPath of docsMarkdownFiles()) {
    const original = readFileSync(markdownPath, "utf8");
    const updated = processMarkdown(original, markdownPath);

    if (updated !== original) {
      changedDocs.push(normalizePath(relative(repoRoot, markdownPath)));
      if (!checkOnly) {
        writeFileSync(markdownPath, updated);
      }
    }
  }

  const changedAssets = syncAssets();

  if (checkOnly && (changedDocs.length > 0 || changedAssets.length > 0)) {
    console.error("Docs LaTeX image codegen is out of date. Run `bun run docs:latex`.");
    for (const filePath of [...changedDocs, ...changedAssets]) {
      console.error(`  ${filePath}`);
    }
    process.exit(1);
  }

  const mode = checkOnly ? "checked" : "generated";
  console.log(
    `Docs LaTeX images ${mode}: ${renderedAssets.size} equation(s), ${changedDocs.length + changedAssets.length} changed file(s).`,
  );
}

main();
