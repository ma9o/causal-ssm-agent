import type { ReactNode } from "react";

const GITHUB_BASE = "https://github.com/ma9o/causal-ssm-agent/blob/master/apps/data-pipeline";

/**
 * Detect file paths ending in .md (e.g. `docs/modeling/functional_spec.md`)
 * and wrap them in links to the GitHub repo.
 */
const MD_PATH_RE = /(?<!\S)((?:[\w.-]+\/)*[\w.-]+\.md)(?!\S)/g;

export function linkifyDocRefs(text: string): ReactNode {
  const parts: ReactNode[] = [];
  let lastIndex = 0;

  for (const match of text.matchAll(MD_PATH_RE)) {
    const path = match[1];
    const start = match.index;

    if (start > lastIndex) {
      parts.push(text.slice(lastIndex, start));
    }

    parts.push(
      <a
        key={start}
        href={`${GITHUB_BASE}/${path}`}
        target="_blank"
        rel="noopener noreferrer"
        className="underline decoration-muted-foreground/40 underline-offset-2 hover:decoration-foreground/60 transition-colors"
      >
        {path}
      </a>,
    );

    lastIndex = start + match[0].length;
  }

  if (parts.length === 0) return text;

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return <>{parts}</>;
}
