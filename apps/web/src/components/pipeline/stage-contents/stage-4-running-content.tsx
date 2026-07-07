"use client";

import { DagEdge } from "@/components/dag/core/dag-edge";
import { DagNodeShell } from "@/components/dag/core/dag-node";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { useDagLayout } from "@/lib/hooks/use-dag-layout";
import {
  type Stage4AdmissionCheckResult,
  type Stage4AdmissionConstructState,
  type Stage4AdmissionConstructStatus,
  type Stage4AdmissionParameter,
  type Stage4AdmissionReport,
  type Stage4AdmissionReplayState,
  useStage4Admission,
} from "@/lib/hooks/use-stage4-admission";
import type { DagGraphInput } from "@/lib/utils/dag-graph-layout";
import { cn } from "@/lib/utils";
import {
  AlertTriangle,
  CheckCircle2,
  Circle,
  Clock,
  FlaskConical,
  Loader2,
  RotateCcw,
  XCircle,
} from "lucide-react";
import { type KeyboardEvent, useMemo, useState } from "react";

const HARD_CHECKS = new Set(["C1a finiteness", "C5a location reach"]);

function titleize(value: string): string {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function constructLabel(construct: Stage4AdmissionConstructState | null | undefined): string {
  if (!construct) return "Construct admission";
  return construct.label ?? titleize(construct.name);
}

function formatParamValue(value: number): string {
  return Number.isInteger(value) ? String(value) : Number.parseFloat(value.toFixed(3)).toString();
}

/** How long a single check took to run: "420ms", "1.4s", "12s". */
function formatCheckDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const seconds = ms / 1000;
  return `${seconds >= 10 ? Math.round(seconds) : seconds.toFixed(1)}s`;
}

/** "Normal(0, 1)" from an authored prior; distribution families are already display-cased. */
function formatPriorSummary(param: Stage4AdmissionParameter): string {
  if (!param.distribution) return "--";
  const values = Object.values(param.params);
  return values.length > 0
    ? `${param.distribution}(${values.map(formatParamValue).join(", ")})`
    : param.distribution;
}

function StatusIcon({ status }: { status: Stage4AdmissionConstructStatus }) {
  if (status === "active") return <Loader2 className="h-3.5 w-3.5 animate-spin" />;
  if (status === "checking") return <FlaskConical className="h-3.5 w-3.5" />;
  if (status === "revising") return <RotateCcw className="h-3.5 w-3.5" />;
  if (status === "blocked") return <XCircle className="h-3.5 w-3.5" />;
  if (status === "admitted_with_consequences") {
    return <AlertTriangle className="h-3.5 w-3.5" />;
  }
  if (status === "admitted") return <CheckCircle2 className="h-3.5 w-3.5" />;
  return <Circle className="h-3.5 w-3.5" />;
}

function checkMode(result: Stage4AdmissionCheckResult): "hard" | "soft" {
  return result.mode ?? (HARD_CHECKS.has(result.check) ? "hard" : "soft");
}

function reportKeyFor(constructName: string, report: Stage4AdmissionReport, index: number): string {
  return `${constructName}:${report.attempt}:${index}`;
}

function progressCounts(state: Stage4AdmissionReplayState) {
  const admitted = state.constructs.filter(
    (construct) =>
      construct.status === "admitted" || construct.status === "admitted_with_consequences",
  ).length;
  const revising = state.constructs.filter((construct) => construct.status === "revising").length;
  const blocked = state.constructs.filter((construct) => construct.status === "blocked").length;
  return { admitted, revising, blocked, total: state.constructs.length };
}

function getFeaturedConstruct(state: Stage4AdmissionReplayState) {
  return (
    state.constructs.find((construct) => construct.name === state.activeConstruct) ??
    (state.latestReport
      ? state.constructs.find((construct) => construct.name === state.latestReport?.name)
      : undefined) ??
    [...state.constructs].reverse().find((construct) => construct.status !== "pending") ??
    state.constructs[0] ??
    null
  );
}

function truncateDagLabel(label: string): string {
  return label.length > 20 ? `${label.slice(0, 17)}...` : label;
}

// Compact node geometry for the inline DAG strip. ELK lays the nodes out; these
// are just the box dimensions it packs.
const DAG_NODE_W = 152;
const DAG_NODE_H = 40;
const DAG_PAD = 10;

/** Admission status → theme color token, shared by the node border and its dot. */
function statusColorVar(status: Stage4AdmissionConstructStatus, isSelected: boolean): string {
  if (isSelected) return "var(--primary)";
  switch (status) {
    case "active":
    case "checking":
      return "var(--primary)";
    case "revising":
    case "admitted_with_consequences":
      return "var(--warning)";
    case "admitted":
      return "var(--success)";
    case "blocked":
      return "var(--destructive)";
    default:
      return "var(--muted-foreground)";
  }
}

/** The single status signal used everywhere: an icon (specific state) in the status color. */
function StatusIndicator({ status }: { status: Stage4AdmissionConstructStatus }) {
  return (
    <span className="shrink-0" style={{ color: statusColorVar(status, false) }}>
      <StatusIcon status={status} />
    </span>
  );
}

/** Worst outcome of a set of checks → status color: any hard fail = red, any soft fail = amber, else green. */
function resultsStatus(
  results: readonly Stage4AdmissionCheckResult[],
): Stage4AdmissionConstructStatus {
  if (results.some((result) => !result.passed && checkMode(result) === "hard")) return "blocked";
  if (results.some((result) => !result.passed)) return "revising";
  return "admitted";
}

/**
 * An attempt's outcome on the shared status color language, mirroring the check-row colors:
 * a hard fail is a hard block (red), a soft-only fail is a revise/decide (amber),
 * admitted-with-consequences is amber, and a clean admit is green.
 */
function reportStatus(report: Stage4AdmissionReport): Stage4AdmissionConstructStatus {
  if (!report.admitted) return resultsStatus(report.results);
  return report.annotations.length > 0 ? "admitted_with_consequences" : "admitted";
}

/**
 * The status used to COLOR a construct in the queue and DAG. While it is still being worked
 * (pending/active/checking) we show the live status; once it has attempts we reflect the last
 * attempt's actual outcome, so a hard-failed last attempt reads red rather than amber "revising".
 */
function constructColorStatus(
  construct: Stage4AdmissionConstructState,
): Stage4AdmissionConstructStatus {
  if (
    construct.status === "pending" ||
    construct.status === "active" ||
    construct.status === "checking"
  ) {
    return construct.status;
  }
  const last = construct.reports[construct.reports.length - 1];
  return last ? reportStatus(last) : construct.status;
}

/**
 * One entry in a construct's check timeline: either its own admission attempt, or a coupled
 * subsystem recheck that re-validated it when a *later* construct closed a feedback loop.
 */
type Stage4TimelineEntry = {
  key: string;
  status: Stage4AdmissionConstructStatus;
  results: Stage4AdmissionCheckResult[];
} & (
  | { kind: "attempt"; attempt: number }
  | { kind: "recheck"; originator: string; closingEdges: string[] }
);

/**
 * A construct's own attempts, followed by the coupled rechecks that re-validate it. A recheck
 * lives on the loop-closing construct's report (its `coupled_recheck`); we surface it under every
 * *other* cycle member it re-checks, attributed to the originator that closed the loop.
 */
function buildTimeline(
  state: Stage4AdmissionReplayState,
  construct: Stage4AdmissionConstructState,
): Stage4TimelineEntry[] {
  const attempts: Stage4TimelineEntry[] = construct.reports.map((report, index) => ({
    kind: "attempt",
    key: reportKeyFor(construct.name, report, index),
    status: reportStatus(report),
    results: report.results,
    attempt: report.attempt,
  }));
  const rechecks: Stage4TimelineEntry[] = [];
  for (const other of state.constructs) {
    if (other.name === construct.name) continue;
    other.reports.forEach((report, index) => {
      const recheck = report.coupled_recheck;
      if (recheck?.constructs.includes(construct.name)) {
        rechecks.push({
          kind: "recheck",
          key: `recheck:${other.name}:${report.attempt}:${index}`,
          status: resultsStatus(recheck.results),
          results: recheck.results,
          originator: other.name,
          closingEdges: recheck.closing_edges ?? [],
        });
      }
    });
  }
  return [...attempts, ...rechecks];
}

/** Status → tinted card surface (border + subtle bg fill). One rule for every status-bearing list item. */
function statusTintClasses(status: Stage4AdmissionConstructStatus): string {
  switch (status) {
    case "active":
    case "checking":
      return "border-primary/30 bg-primary/5";
    case "revising":
    case "admitted_with_consequences":
      return "border-warning/40 bg-warning/10";
    case "admitted":
      return "border-success/30 bg-success/5";
    case "blocked":
      return "border-destructive/30 bg-destructive/5";
    default:
      return "border-border bg-muted/20";
  }
}

function MiniConstructDag({
  state,
  selectedName,
  onSelectConstruct,
}: {
  state: Stage4AdmissionReplayState;
  selectedName: string | null | undefined;
  onSelectConstruct: (constructName: string) => void;
}) {
  const constructByName = new Map(state.constructs.map((construct) => [construct.name, construct]));
  const orderByName = new Map(state.constructs.map((construct, index) => [construct.name, index]));

  // Key the ELK layout on the topology only (construct names + edges) so status
  // ticks recolor in place instead of triggering a full re-layout. Construct and
  // edge endpoints are `[a-z0-9_]` identifiers, so `|`/`>` are safe delimiters.
  const nodeKey = state.constructs.map((construct) => construct.name).join("|");
  const edgeKey = (state.plan?.edges ?? []).map((edge) => `${edge.cause}>${edge.effect}`).join("|");
  const graph = useMemo<DagGraphInput>(() => {
    const names = nodeKey ? nodeKey.split("|") : [];
    const nameSet = new Set(names);
    return {
      direction: "RIGHT",
      nodes: names.map((id) => ({ id, width: DAG_NODE_W, height: DAG_NODE_H })),
      edges: (edgeKey ? edgeKey.split("|") : []).flatMap((pair, index) => {
        const [source, target] = pair.split(">");
        return nameSet.has(source) && nameSet.has(target)
          ? [{ id: `edge-${index}`, source, target }]
          : [];
      }),
    };
  }, [nodeKey, edgeKey]);

  const { nodes, edges, width, height, isLayouting } = useDagLayout(graph);
  const geoByName = new Map(nodes.map((node) => [node.id, node]));

  return (
    <div className="max-h-[320px] overflow-auto rounded-lg border bg-muted/20">
      {isLayouting ? (
        <div className="flex h-[180px] items-center justify-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          Laying out DAG...
        </div>
      ) : (
        <svg
          width={width + DAG_PAD * 2}
          height={height + DAG_PAD * 2}
          viewBox={`0 0 ${width + DAG_PAD * 2} ${height + DAG_PAD * 2}`}
          role="img"
          aria-label="Retained Stage 1b construct DAG"
          className="block"
        >
          <g transform={`translate(${DAG_PAD} ${DAG_PAD})`}>
            <g>
              {edges.map((edge) => {
                const source = constructByName.get(edge.source);
                const target = constructByName.get(edge.target);
                const isFeedback =
                  (orderByName.get(edge.source) ?? 0) > (orderByName.get(edge.target) ?? 0);
                const isActive =
                  source?.status !== "pending" &&
                  source?.status !== "blocked" &&
                  target?.status !== "pending";
                const color = isFeedback
                  ? "var(--warning)"
                  : isActive
                    ? "var(--primary)"
                    : "var(--muted-foreground)";
                return (
                  <DagEdge
                    key={edge.id}
                    points={edge.points}
                    color={color}
                    width={1.6}
                    dashed={isFeedback}
                    opacity={isFeedback ? 0.6 : isActive ? 0.55 : 0.28}
                  />
                );
              })}
            </g>
            <g>
              {state.constructs.map((construct) => {
                const geo = geoByName.get(construct.name);
                if (!geo) return null;
                const isSelected = construct.name === selectedName;
                const label = constructLabel(construct);
                const colorStatus = constructColorStatus(construct);
                const isPending = colorStatus === "pending";
                const handleKeyDown = (event: KeyboardEvent<SVGGElement>) => {
                  if (event.key !== "Enter" && event.key !== " ") return;
                  event.preventDefault();
                  onSelectConstruct(construct.name);
                };
                return (
                  <g
                    key={construct.name}
                    transform={`translate(${geo.x} ${geo.y})`}
                    role="button"
                    tabIndex={0}
                    aria-label={`Show ${label}`}
                    aria-pressed={isSelected}
                    className="cursor-pointer outline-none"
                    onClick={() => onSelectConstruct(construct.name)}
                    onKeyDown={handleKeyDown}
                  >
                    <DagNodeShell
                      width={geo.width}
                      height={geo.height}
                      accent={
                        isSelected || !isPending
                          ? statusColorVar(colorStatus, isSelected)
                          : undefined
                      }
                      highlighted={isSelected}
                    >
                      <title>{label}</title>
                      <circle
                        cx={15}
                        cy={geo.height / 2}
                        r={isSelected ? 4.5 : 3.5}
                        fill={statusColorVar(colorStatus, isSelected)}
                        fillOpacity={isPending && !isSelected ? 0.4 : 1}
                      />
                      <text
                        x={28}
                        y={geo.height / 2}
                        fontSize={11}
                        fontWeight={500}
                        fill="var(--foreground)"
                        dominantBaseline="middle"
                      >
                        {truncateDagLabel(label)}
                      </text>
                    </DagNodeShell>
                  </g>
                );
              })}
            </g>
          </g>
        </svg>
      )}
    </div>
  );
}

function ConstructQueueItem({
  construct,
  index,
  isSelected,
  onSelect,
}: {
  construct: Stage4AdmissionConstructState;
  index: number;
  isSelected: boolean;
  onSelect: (constructName: string) => void;
}) {
  return (
    <li>
      <button
        type="button"
        aria-pressed={isSelected}
        className={cn(
          "grid w-full grid-cols-[2rem_minmax(0,1fr)_auto] items-center gap-2 rounded-lg border px-3 py-2.5 text-left transition-colors",
          statusTintClasses(constructColorStatus(construct)),
          isSelected && "ring-1 ring-primary",
        )}
        onClick={() => onSelect(construct.name)}
      >
        <div className="flex h-6 w-6 items-center justify-center rounded-md border border-border/60 bg-background/60 text-xs tabular-nums text-muted-foreground">
          {index + 1}
        </div>
        <div className="min-w-0">
          <div className="flex min-w-0 items-center gap-2">
            <span className="truncate text-sm font-medium" title={constructLabel(construct)}>
              {constructLabel(construct)}
            </span>
          </div>
        </div>
        {construct.reports.length > 0 ? (
          <span
            className="rounded-md border border-border/60 bg-background/60 px-1.5 py-0.5 text-xs tabular-nums text-muted-foreground"
            aria-label={`${construct.reports.length} ${construct.reports.length === 1 ? "attempt" : "attempts"}`}
            title={`${construct.reports.length} ${construct.reports.length === 1 ? "attempt" : "attempts"}`}
          >
            {construct.reports.length}×
          </span>
        ) : (
          <span className="text-xs tabular-nums text-muted-foreground/50" aria-hidden>
            —
          </span>
        )}
      </button>
    </li>
  );
}

function ConstructDetail({ construct }: { construct: Stage4AdmissionConstructState | null }) {
  if (!construct) {
    return <p className="text-sm text-muted-foreground">Waiting for the construct order.</p>;
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="truncate text-sm font-semibold" title={constructLabel(construct)}>
          {constructLabel(construct)}
        </h3>
      </div>

      <div className="space-y-3 text-sm">
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">Authored parameters</div>
          {construct.parameters && construct.parameters.length > 0 ? (
            <div className="overflow-hidden rounded-md border">
              <table className="w-full table-fixed text-xs">
                <thead>
                  <tr className="border-b bg-muted/30 text-[10px] uppercase tracking-wide text-muted-foreground">
                    <th className="w-[55%] px-2 py-1 text-left font-medium">Parameter</th>
                    <th className="px-2 py-1 text-right font-medium">Prior</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {construct.parameters.map((param) => (
                    <tr key={param.name}>
                      <td className="px-2 py-1.5 align-top">
                        <span className="block truncate font-mono" title={param.name}>
                          {param.name}
                        </span>
                      </td>
                      <td className="px-2 py-1.5 text-right align-top font-mono tabular-nums text-muted-foreground">
                        <span className="block truncate" title={formatPriorSummary(param)}>
                          {formatPriorSummary(param)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="rounded-md border bg-muted/25 px-3 py-2 text-xs text-muted-foreground">
              none
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function AttemptHistory({
  entries,
  selectedKey,
  onSelect,
}: {
  entries: Stage4TimelineEntry[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
}) {
  const attemptCount = entries.filter((entry) => entry.kind === "attempt").length;

  return (
    <div className="space-y-3 border-t border-border pt-4">
      <div className="flex items-center justify-between gap-3">
        <h4 className="text-sm font-semibold">Attempts</h4>
        <span className="text-xs tabular-nums text-muted-foreground">
          {attemptCount} {attemptCount === 1 ? "attempt" : "attempts"}
        </span>
      </div>
      {entries.length === 0 ? (
        <p className="text-xs text-muted-foreground">Not checked yet.</p>
      ) : (
        <ol className="space-y-2">
          {entries.map((entry) => {
            const isSelected = entry.key === selectedKey;
            return (
              <li key={entry.key}>
                <button
                  type="button"
                  aria-pressed={isSelected}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-md border px-3 py-2 text-left text-xs transition-colors",
                    statusTintClasses(entry.status),
                    isSelected && "ring-1 ring-primary",
                  )}
                  onClick={() => onSelect(entry.key)}
                >
                  <StatusIndicator status={entry.status} />
                  {entry.kind === "attempt" ? (
                    <span className="font-medium tabular-nums">Attempt {entry.attempt}</span>
                  ) : (
                    <span className="min-w-0 truncate">
                      <span className="font-medium">Coupled recheck</span>
                      <span
                        className="ml-1.5 text-muted-foreground"
                        title={`Triggered when ${entry.originator} closed the feedback loop`}
                      >
                        from {entry.originator}
                      </span>
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

function CheckRow({ result }: { result: Stage4AdmissionCheckResult }) {
  return (
    <li
      className={cn(
        "rounded-lg border p-3",
        result.passed
          ? "border-success/25 bg-success/5"
          : checkMode(result) === "hard"
            ? "border-destructive/30 bg-destructive/5"
            : "border-warning/30 bg-warning/10",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium">{result.check}</span>
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            {result.target ? `${result.target}: ` : ""}
            {result.value}
          </div>
        </div>
        <div
          className="flex shrink-0 items-center gap-1 rounded-md border bg-card px-2 py-1 text-xs tabular-nums text-muted-foreground"
          title="Time this check took to run"
        >
          <Clock className="h-3 w-3" aria-hidden />
          {formatCheckDuration(result.duration_ms)}
        </div>
      </div>
      {!result.passed && (
        <div className="mt-2 space-y-1 text-xs leading-relaxed text-muted-foreground">
          {result.note && <p>{result.note}</p>}
          {result.diagnosis?.slice(0, 2).map((diagnosis) => (
            <p key={diagnosis}>{diagnosis}</p>
          ))}
        </div>
      )}
    </li>
  );
}

function ReachabilityPanel({ entry }: { entry: Stage4TimelineEntry | null }) {
  if (!entry) {
    return (
      <div className="space-y-3">
        <div>
          <h3 className="text-sm font-semibold">Awaiting report</h3>
        </div>
        <div className="flex items-center gap-2 rounded-md border bg-muted/20 px-3 py-3 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          Waiting for the first prior-predictive check.
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div>
        <h3 className="text-sm font-semibold">
          {entry.kind === "recheck"
            ? `Loop closed by ${entry.originator}`
            : `Attempt ${entry.attempt}`}
        </h3>
        {entry.kind === "recheck" && entry.closingEdges.length > 0 && (
          <div className="mt-0.5 text-xs text-muted-foreground">
            Closing edge{entry.closingEdges.length > 1 ? "s" : ""}: {entry.closingEdges.join(", ")}
          </div>
        )}
      </div>
      <ol className="max-h-[520px] space-y-2 overflow-auto pr-1">
        {entry.results.map((result, index) => (
          <CheckRow key={`${result.check}-${result.target}-${index}`} result={result} />
        ))}
      </ol>
    </div>
  );
}

export function Stage4AdmissionRunningView({
  state,
}: {
  state: Stage4AdmissionReplayState | null;
}) {
  const [selectedConstructName, setSelectedConstructName] = useState<string | null>(null);
  const [reportSelection, setReportSelection] = useState<{
    selectedKey: string | null;
    latestKeyAtSelection: string | null;
  }>({ selectedKey: null, latestKeyAtSelection: null });
  const counts = state ? progressCounts(state) : { admitted: 0, revising: 0, blocked: 0, total: 0 };
  const liveFeaturedConstruct = state ? getFeaturedConstruct(state) : null;
  const featuredConstruct =
    state?.constructs.find((construct) => construct.name === selectedConstructName) ??
    liveFeaturedConstruct;
  const timeline = useMemo(
    () => (state && featuredConstruct ? buildTimeline(state, featuredConstruct) : []),
    [state, featuredConstruct],
  );
  const latestEntry = timeline[timeline.length - 1] ?? null;
  const explicitlySelectedEntry =
    timeline.find((entry) => entry.key === reportSelection.selectedKey) ?? null;
  const followsLatest =
    reportSelection.selectedKey === null ||
    reportSelection.selectedKey === reportSelection.latestKeyAtSelection;
  const selectedEntry = followsLatest ? latestEntry : (explicitlySelectedEntry ?? latestEntry);
  const progress = counts.total > 0 ? Math.round((counts.admitted / counts.total) * 100) : 0;
  const handleSelectConstruct = (constructName: string) => {
    setSelectedConstructName(constructName);
    setReportSelection({ selectedKey: null, latestKeyAtSelection: null });
  };
  const handleSelectEntry = (key: string) => {
    setReportSelection({ selectedKey: key, latestKeyAtSelection: latestEntry?.key ?? null });
  };

  if (!state?.plan || state.constructs.length === 0) {
    return (
      <div className="flex items-center gap-2 py-3 text-sm text-muted-foreground">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Preparing construct admission order...
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <Card size="sm">
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="text-sm font-semibold">Construct admission</div>
              <div className="mt-1 text-sm text-muted-foreground">
                {counts.admitted} of {counts.total} constructs admitted
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              {counts.revising > 0 && <Badge variant="warning">{counts.revising} revising</Badge>}
              {counts.blocked > 0 && <Badge variant="destructive">{counts.blocked} blocked</Badge>}
              {state.done && <Badge variant="success">done</Badge>}
            </div>
          </div>
          <div className="h-2 overflow-hidden rounded-md bg-muted">
            <div
              className="h-full rounded-md bg-primary transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <MiniConstructDag
            state={state}
            selectedName={featuredConstruct?.name}
            onSelectConstruct={handleSelectConstruct}
          />
          {state.error && (
            <div className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive">
              {state.error}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Master → detail → report reads left to right across one connected surface. */}
      <div className="grid divide-y divide-border overflow-hidden rounded-xl bg-card ring-1 ring-foreground/10 xl:grid-cols-[minmax(280px,0.82fr)_minmax(360px,1fr)_minmax(360px,1.1fr)] xl:divide-x xl:divide-y-0">
        <section className="space-y-3 p-4">
          <h3 className="text-sm font-semibold">Construct queue</h3>
          <ol className="max-h-[720px] space-y-2 overflow-auto pr-1">
            {state.constructs.map((construct, index) => (
              <ConstructQueueItem
                key={construct.name}
                construct={construct}
                index={index}
                isSelected={construct.name === featuredConstruct?.name}
                onSelect={handleSelectConstruct}
              />
            ))}
          </ol>
        </section>

        <section className="space-y-4 p-4">
          <ConstructDetail construct={featuredConstruct} />
          {featuredConstruct && (
            <AttemptHistory
              entries={timeline}
              selectedKey={selectedEntry?.key ?? null}
              onSelect={handleSelectEntry}
            />
          )}
        </section>

        <section className="p-4">
          <ReachabilityPanel entry={selectedEntry} />
        </section>
      </div>
    </div>
  );
}

export default function Stage4RunningContent({ workspaceId }: { workspaceId: string }) {
  const state = useStage4Admission(workspaceId);
  return <Stage4AdmissionRunningView state={state} />;
}
