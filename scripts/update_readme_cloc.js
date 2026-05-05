#!/usr/bin/env bun

const { execFileSync } = require("node:child_process");
const { mkdtempSync, readFileSync, rmSync, writeFileSync } = require("node:fs");
const { tmpdir } = require("node:os");
const { basename, resolve } = require("node:path");

const repoRoot = resolve(__dirname, "..");
const readmePath = resolve(repoRoot, "README.md");
const startMarker = "<!-- cloc:start -->";
const endMarker = "<!-- cloc:end -->";
const vcsCommand = "git ls-files --cached --others --exclude-standard";

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function escapeMarkdownCell(value) {
  return value.replaceAll("|", "\\|");
}

function normalizeFilePath(filePath) {
  return filePath.replace(/^\.\//, "");
}

function percentile(sortedValues, p) {
  if (sortedValues.length === 0) {
    return 0;
  }

  const index = Math.max(0, Math.ceil(sortedValues.length * p) - 1);
  return sortedValues[index];
}

function formatFileSizeDistribution(fileCodeLines) {
  const sortedCodeLines = [...fileCodeLines].sort((a, b) => a - b);
  const p50 = percentile(sortedCodeLines, 0.5);
  const p90 = percentile(sortedCodeLines, 0.9);
  const max = sortedCodeLines.at(-1) ?? 0;

  return [p50, p90, max].map(formatNumber).join("&nbsp;/&nbsp;");
}

function formatFileLink(filePath) {
  const normalizedFilePath = normalizeFilePath(filePath);

  return `[${escapeMarkdownCell(basename(normalizedFilePath))}](<${normalizedFilePath}>)`;
}

function formatLargestFiles(files) {
  return files.map((file) => formatFileLink(file.path)).join(", ");
}

function renderRow(label, stats, bold = false) {
  const values = [
    escapeMarkdownCell(label),
    formatNumber(stats.nFiles),
    formatNumber(stats.blank),
    formatNumber(stats.comment),
    formatNumber(stats.code),
    formatFileSizeDistribution(stats.fileCodeLines),
    formatLargestFiles(stats.files),
  ];

  if (bold) {
    return `| ${values.map((value) => `**${value}**`).join(" | ")} |`;
  }

  return `| ${values.join(" | ")} |`;
}

function buildTable(report) {
  const rows = Object.entries(report)
    .filter(([language]) => language !== "header" && language !== "SUM")
    .map(([language, stats]) => renderRow(language, stats));

  return [
    "| Language | Files | Blank | Comment | Code | p50&nbsp;/&nbsp;p90&nbsp;/&nbsp;max | Top files |",
    "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ...rows,
    renderRow("Total", report.SUM, true),
  ].join("\n");
}

function buildBlock(report) {
  return [
    startMarker,
    "",
    "## Lines of Code",
    "",
    buildTable(report),
    "",
    endMarker,
    "",
  ].join("\n");
}

function replaceGeneratedBlock(readme, block) {
  const hasStart = readme.includes(startMarker);
  const hasEnd = readme.includes(endMarker);

  if (hasStart !== hasEnd) {
    throw new Error("README.md has only one cloc marker; remove the stale marker pair and rerun.");
  }

  const withoutBlock = hasStart
    ? readme.replace(new RegExp(`${startMarker}[\\s\\S]*?${endMarker}\\n?`), "")
    : readme;

  return `${withoutBlock.trimEnd()}\n\n${block}`;
}

function buildLanguageReport(report) {
  const languages = new Map();
  const total = {
    nFiles: 0,
    blank: 0,
    comment: 0,
    code: 0,
    fileCodeLines: [],
    files: [],
  };

  for (const [filePath, stats] of Object.entries(report)) {
    if (filePath === "header" || filePath === "SUM") {
      continue;
    }

    const language = stats.language;
    const languageStats = languages.get(language) ?? {
      nFiles: 0,
      blank: 0,
      comment: 0,
      code: 0,
      fileCodeLines: [],
      files: [],
    };

    languageStats.nFiles += 1;
    languageStats.blank += stats.blank;
    languageStats.comment += stats.comment;
    languageStats.code += stats.code;
    languageStats.fileCodeLines.push(stats.code);
    languageStats.files.push({ path: filePath, code: stats.code });

    total.nFiles += 1;
    total.blank += stats.blank;
    total.comment += stats.comment;
    total.code += stats.code;
    total.fileCodeLines.push(stats.code);
    total.files.push({ path: filePath, code: stats.code });

    languages.set(language, languageStats);
  }

  for (const stats of [...languages.values(), total]) {
    stats.files = stats.files
      .sort((fileA, fileB) => fileB.code - fileA.code || fileA.path.localeCompare(fileB.path))
      .slice(0, 3);
  }

  const sortedLanguages = [...languages.entries()].sort(
    ([languageA, statsA], [languageB, statsB]) =>
      statsB.code - statsA.code || languageA.localeCompare(languageB),
  );

  return {
    ...Object.fromEntries(sortedLanguages),
    SUM: total,
  };
}

const tempDir = mkdtempSync(resolve(tmpdir(), "causal-ssm-cloc-"));
const excludeListPath = resolve(tempDir, "exclude-list.txt");

writeFileSync(excludeListPath, "README.md\n");

let report;
try {
  const rawReport = execFileSync(
    "cloc",
    [
      "--by-file",
      `--vcs=${vcsCommand}`,
      `--exclude-list-file=${excludeListPath}`,
      "--json",
      "--quiet",
    ],
    {
      cwd: repoRoot,
      encoding: "utf8",
    },
  );
  report = buildLanguageReport(JSON.parse(rawReport));
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}

const readme = readFileSync(readmePath, "utf8");

writeFileSync(readmePath, replaceGeneratedBlock(readme, buildBlock(report)));
