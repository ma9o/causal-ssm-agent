import type { Stage4Graph } from "@/lib/stage4-runtime";

export type Stage4SectionId =
  | "model_decisions"
  | "global_review"
  | "measurement_prior"
  | "dynamics_prior"
  | "effect_prior"
  | "correlation_prior"
  | "repair_barrier"
  | "global_prior_review"
  | "done";

export type Stage4SectionEdgeKind = "forward" | "phase_advance" | "repair_transition";

export interface Stage4Point {
  x: number;
  y: number;
}

export interface Stage4SectionEdge {
  from: Stage4SectionId;
  to: Stage4SectionId;
  kind: Stage4SectionEdgeKind;
}

export interface Stage4SectionRect {
  left: number;
  right: number;
  top: number;
  bottom: number;
  centerX: number;
  centerY: number;
}

export const STAGE4_SECTION_ORDER: Array<{
  id: Stage4SectionId;
  label: string;
  tooltip: string;
}> = [
  {
    id: "model_decisions",
    label: "Model Decisions",
    tooltip: "Chooses likelihood functions and loading constraints for each indicator in the causal model.",
  },
  {
    id: "global_review",
    label: "Global Review",
    tooltip: "Reviews the full model specification for consistency before locking it.",
  },
  {
    id: "measurement_prior",
    label: "Measurement Priors",
    tooltip: "Sets priors on measurement parameters that link latent constructs to observed indicators.",
  },
  {
    id: "dynamics_prior",
    label: "Dynamics Priors",
    tooltip: "Sets priors on autoregressive and mean-reversion parameters for each latent construct.",
  },
  {
    id: "effect_prior",
    label: "Effect Priors",
    tooltip: "Sets priors on cross-construct causal effect strengths for each edge in the DAG.",
  },
  {
    id: "correlation_prior",
    label: "Correlation Priors",
    tooltip: "Sets priors on initial correlations between latent constructs at baseline.",
  },
  {
    id: "repair_barrier",
    label: "Repair Barrier",
    tooltip: "Checkpoint that validates repaired blocks before proceeding to the final review.",
  },
  {
    id: "global_prior_review",
    label: "Prior Review",
    tooltip: "Reviews the full prior system for cross-section consistency and flags pathologies.",
  },
  {
    id: "done",
    label: "Complete",
    tooltip: "All parameterized blocks have been accepted and the model specification is finalized.",
  },
];

export const STAGE4_SECTION_NODE_WIDTH = 320;
export const STAGE4_SECTION_NODE_HEIGHT = 102;

export const STAGE4_SECTION_NODE_POSITION: Record<Stage4SectionId, Stage4Point> = {
  model_decisions: { x: 36, y: 28 },
  global_review: { x: 386, y: 28 },
  measurement_prior: { x: 736, y: 28 },
  dynamics_prior: { x: 736, y: 278 },
  effect_prior: { x: 386, y: 278 },
  correlation_prior: { x: 36, y: 278 },
  repair_barrier: { x: 36, y: 528 },
  global_prior_review: { x: 386, y: 528 },
  done: { x: 736, y: 528 },
};

const OPTIONAL_SECTIONS = new Set<Stage4SectionId>([
  "measurement_prior",
  "dynamics_prior",
  "effect_prior",
  "correlation_prior",
  "global_prior_review",
]);

const KIND_ORDER: Record<Stage4SectionEdgeKind, number> = {
  phase_advance: 0,
  forward: 1,
  repair_transition: 2,
};

interface RouteNode extends Stage4Point {
  id: string;
}

interface RouteCandidate {
  handle: Stage4Point;
  boundary: Stage4Point;
}

const SECTION_INDEX = new Map(STAGE4_SECTION_ORDER.map((section, index) => [section.id, index]));
const X_POSITIONS = [...new Set(Object.values(STAGE4_SECTION_NODE_POSITION).map((pos) => pos.x))].sort(
  (left, right) => left - right,
);
const Y_POSITIONS = [...new Set(Object.values(STAGE4_SECTION_NODE_POSITION).map((pos) => pos.y))].sort(
  (left, right) => left - right,
);
const COLUMN_BY_X = new Map(X_POSITIONS.map((x, index) => [x, index]));
const ROW_BY_Y = new Map(Y_POSITIONS.map((y, index) => [y, index]));
const X_BOUNDARIES = [
  X_POSITIONS[0]! - 24,
  ...X_POSITIONS.slice(0, -1).map((x, index) => {
    const leftRight = x + STAGE4_SECTION_NODE_WIDTH;
    return Math.round((leftRight + X_POSITIONS[index + 1]!) / 2);
  }),
  X_POSITIONS[X_POSITIONS.length - 1]! + STAGE4_SECTION_NODE_WIDTH + 24,
];
const Y_BOUNDARIES = [
  Y_POSITIONS[0]! - 16,
  ...Y_POSITIONS.slice(0, -1).map((y, index) => {
    const topBottom = y + STAGE4_SECTION_NODE_HEIGHT;
    return Math.round((topBottom + Y_POSITIONS[index + 1]!) / 2);
  }),
  Y_POSITIONS[Y_POSITIONS.length - 1]! + STAGE4_SECTION_NODE_HEIGHT + 16,
];

const INTERSECTIONS: RouteNode[] = X_BOUNDARIES.flatMap((x, xIndex) =>
  Y_BOUNDARIES.map((y, yIndex) => ({ id: `i:${xIndex}:${yIndex}`, x, y })),
);

function pointKey(point: Stage4Point): string {
  return `${point.x},${point.y}`;
}

function isBoundaryX(x: number): boolean {
  return X_BOUNDARIES.includes(x);
}

function isBoundaryY(y: number): boolean {
  return Y_BOUNDARIES.includes(y);
}

function distance(left: Stage4Point, right: Stage4Point): number {
  return Math.abs(left.x - right.x) + Math.abs(left.y - right.y);
}

function countBends(points: Stage4Point[]): number {
  let bends = 0;
  for (let index = 1; index < points.length - 1; index++) {
    const prev = points[index - 1]!;
    const current = points[index]!;
    const next = points[index + 1]!;
    const dx1 = Math.sign(current.x - prev.x);
    const dy1 = Math.sign(current.y - prev.y);
    const dx2 = Math.sign(next.x - current.x);
    const dy2 = Math.sign(next.y - current.y);
    if (dx1 !== dx2 || dy1 !== dy2) bends++;
  }
  return bends;
}

function simplifyPolyline(points: Stage4Point[]): Stage4Point[] {
  const deduped: Stage4Point[] = [];
  for (const point of points) {
    const previous = deduped[deduped.length - 1];
    if (!previous || previous.x !== point.x || previous.y !== point.y) {
      deduped.push(point);
    }
  }

  const simplified: Stage4Point[] = [];
  for (const point of deduped) {
    const prev = simplified[simplified.length - 1];
    const prevPrev = simplified[simplified.length - 2];
    if (
      prev &&
      prevPrev &&
      ((prevPrev.x === prev.x && prev.x === point.x) ||
        (prevPrev.y === prev.y && prev.y === point.y))
    ) {
      simplified[simplified.length - 1] = point;
      continue;
    }
    simplified.push(point);
  }

  return simplified;
}

function buildRouteGraph(nodes: RouteNode[]): Map<string, Array<{ id: string; weight: number }>> {
  const adjacency = new Map<string, Array<{ id: string; weight: number }>>();
  for (const node of nodes) {
    adjacency.set(node.id, []);
  }

  for (let leftIndex = 0; leftIndex < nodes.length; leftIndex++) {
    const left = nodes[leftIndex]!;
    for (let rightIndex = leftIndex + 1; rightIndex < nodes.length; rightIndex++) {
      const right = nodes[rightIndex]!;
      const alignedOnVerticalBoundary = left.x === right.x && isBoundaryX(left.x);
      const alignedOnHorizontalBoundary = left.y === right.y && isBoundaryY(left.y);
      if (!alignedOnVerticalBoundary && !alignedOnHorizontalBoundary) continue;
      const weight = distance(left, right);
      adjacency.get(left.id)?.push({ id: right.id, weight });
      adjacency.get(right.id)?.push({ id: left.id, weight });
    }
  }

  return adjacency;
}

function shortestPath(nodes: RouteNode[], startId: string, endId: string): Stage4Point[] | null {
  if (startId === endId) {
    const startNode = nodes.find((node) => node.id === startId);
    return startNode ? [{ x: startNode.x, y: startNode.y }] : null;
  }

  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const adjacency = buildRouteGraph(nodes);
  const distances = new Map<string, number>(nodes.map((node) => [node.id, Number.POSITIVE_INFINITY]));
  const previous = new Map<string, string | null>(nodes.map((node) => [node.id, null]));
  const queue = new Set(nodes.map((node) => node.id));

  distances.set(startId, 0);

  while (queue.size > 0) {
    let currentId: string | null = null;
    let currentDistance = Number.POSITIVE_INFINITY;
    for (const candidateId of queue) {
      const candidateDistance = distances.get(candidateId) ?? Number.POSITIVE_INFINITY;
      if (candidateDistance < currentDistance) {
        currentDistance = candidateDistance;
        currentId = candidateId;
      }
    }

    if (currentId === null || currentDistance === Number.POSITIVE_INFINITY) {
      break;
    }

    queue.delete(currentId);
    if (currentId === endId) break;

    for (const neighbor of adjacency.get(currentId) ?? []) {
      if (!queue.has(neighbor.id)) continue;
      const nextDistance = currentDistance + neighbor.weight;
      if (nextDistance < (distances.get(neighbor.id) ?? Number.POSITIVE_INFINITY)) {
        distances.set(neighbor.id, nextDistance);
        previous.set(neighbor.id, currentId);
      }
    }
  }

  if ((distances.get(endId) ?? Number.POSITIVE_INFINITY) === Number.POSITIVE_INFINITY) {
    return null;
  }

  const pathIds: string[] = [];
  let currentId: string | null = endId;
  while (currentId) {
    pathIds.push(currentId);
    currentId = previous.get(currentId) ?? null;
  }
  pathIds.reverse();

  return pathIds
    .map((id) => nodeById.get(id))
    .filter((node): node is RouteNode => Boolean(node))
    .map((node) => ({ x: node.x, y: node.y }));
}

function getSectionGrid(sectionId: Stage4SectionId): { col: number; row: number } {
  const pos = STAGE4_SECTION_NODE_POSITION[sectionId];
  const col = COLUMN_BY_X.get(pos.x);
  const row = ROW_BY_Y.get(pos.y);
  if (col === undefined || row === undefined) {
    throw new Error(`Unknown Stage 4 section position for ${sectionId}`);
  }
  return { col, row };
}

function sectionRect(sectionId: Stage4SectionId): Stage4SectionRect {
  const pos = STAGE4_SECTION_NODE_POSITION[sectionId];
  return {
    left: pos.x,
    right: pos.x + STAGE4_SECTION_NODE_WIDTH,
    top: pos.y,
    bottom: pos.y + STAGE4_SECTION_NODE_HEIGHT,
    centerX: pos.x + STAGE4_SECTION_NODE_WIDTH / 2,
    centerY: pos.y + STAGE4_SECTION_NODE_HEIGHT / 2,
  };
}

function getRouteCandidates(sectionId: Stage4SectionId): RouteCandidate[] {
  const rect = sectionRect(sectionId);
  const { col, row } = getSectionGrid(sectionId);

  return [
    {
      handle: { x: rect.left, y: rect.centerY },
      boundary: { x: X_BOUNDARIES[col]!, y: rect.centerY },
    },
    {
      handle: { x: rect.right, y: rect.centerY },
      boundary: { x: X_BOUNDARIES[col + 1]!, y: rect.centerY },
    },
    {
      handle: { x: rect.centerX, y: rect.top },
      boundary: { x: rect.centerX, y: Y_BOUNDARIES[row]! },
    },
    {
      handle: { x: rect.centerX, y: rect.bottom },
      boundary: { x: rect.centerX, y: Y_BOUNDARIES[row + 1]! },
    },
  ];
}

function routeBetweenBoundaries(source: Stage4Point, target: Stage4Point): Stage4Point[] | null {
  const routeNodes: RouteNode[] = [
    ...INTERSECTIONS,
    { id: `source:${pointKey(source)}`, x: source.x, y: source.y },
    { id: `target:${pointKey(target)}`, x: target.x, y: target.y },
  ];

  return shortestPath(routeNodes, `source:${pointKey(source)}`, `target:${pointKey(target)}`);
}

export function getStage4SectionId(kind: string): Stage4SectionId {
  switch (kind) {
    case "indicator_decision":
    case "loading_decision":
    case "model_spec_lock":
      return "model_decisions";
    case "global_review":
      return "global_review";
    case "measurement_prior":
      return "measurement_prior";
    case "dynamics_prior":
      return "dynamics_prior";
    case "effect_prior":
      return "effect_prior";
    case "correlation_prior":
      return "correlation_prior";
    case "repair_barrier":
      return "repair_barrier";
    case "global_prior_review":
      return "global_prior_review";
    case "done":
      return "done";
    default:
      throw new Error(`Unknown Stage 4 node kind: ${kind}`);
  }
}

export function isOptionalStage4Section(sectionId: Stage4SectionId): boolean {
  return OPTIONAL_SECTIONS.has(sectionId);
}

export function getStage4SectionRect(sectionId: Stage4SectionId): Stage4SectionRect {
  return sectionRect(sectionId);
}

export function deriveStage4SectionEdges(graph: Stage4Graph | null): Stage4SectionEdge[] {
  if (!graph) return [];

  const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
  const edges = new Map<string, Stage4SectionEdge>();

  for (const edge of graph.edges) {
    const sourceNode = nodeById.get(edge.from);
    const targetNode = nodeById.get(edge.to);
    if (!sourceNode || !targetNode) continue;

    const from = getStage4SectionId(sourceNode.kind);
    const to = getStage4SectionId(targetNode.kind);
    if (from === to) continue;

    const kind = edge.kind as Stage4SectionEdgeKind;
    const key = `${from}->${to}:${kind}`;
    if (!edges.has(key)) {
      edges.set(key, { from, to, kind });
    }
  }

  return [...edges.values()].sort((left, right) => {
    const fromDelta = (SECTION_INDEX.get(left.from) ?? 0) - (SECTION_INDEX.get(right.from) ?? 0);
    if (fromDelta !== 0) return fromDelta;
    const toDelta = (SECTION_INDEX.get(left.to) ?? 0) - (SECTION_INDEX.get(right.to) ?? 0);
    if (toDelta !== 0) return toDelta;
    return KIND_ORDER[left.kind] - KIND_ORDER[right.kind];
  });
}

export function routeStage4SectionEdge(
  from: Stage4SectionId,
  to: Stage4SectionId,
): Stage4Point[] {
  const sourceCandidates = getRouteCandidates(from);
  const targetCandidates = getRouteCandidates(to);

  let bestPath: Stage4Point[] | null = null;
  let bestScore = Number.POSITIVE_INFINITY;

  for (const source of sourceCandidates) {
    for (const target of targetCandidates) {
      const boundaryPath = routeBetweenBoundaries(source.boundary, target.boundary);
      if (!boundaryPath) continue;

      const fullPath = simplifyPolyline([source.handle, ...boundaryPath, target.handle]);
      const length = fullPath.reduce((total, point, index) => {
        if (index === 0) return 0;
        return total + distance(fullPath[index - 1]!, point);
      }, 0);
      const score = length + countBends(fullPath) * 24;

      if (score < bestScore) {
        bestScore = score;
        bestPath = fullPath;
      }
    }
  }

  if (!bestPath) {
    const sourceRect = sectionRect(from);
    const targetRect = sectionRect(to);
    return [
      { x: sourceRect.centerX, y: sourceRect.centerY },
      { x: targetRect.centerX, y: targetRect.centerY },
    ];
  }

  return bestPath;
}
