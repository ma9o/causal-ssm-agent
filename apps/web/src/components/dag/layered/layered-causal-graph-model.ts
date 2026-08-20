import type {
  CausalDesign,
  KnownInput,
  LatentStructure,
  MeasurementStructure,
  PosteriorData,
  ScientificOnlyConstruct,
  StatisticalModelSpecData,
  StructuralDisposition,
  StructuralPlan,
} from "@nof1-causal-lab/api-types";
import type { AnalysisSimulationResult, EdgePosterior } from "../intervention-dag-types";

export const CAUSAL_GRAPH_LAYER_ORDER = [
  "structure",
  "measurement",
  "design",
  "specification",
  "fit",
  "simulation",
] as const;

export type CausalGraphLayerId = (typeof CAUSAL_GRAPH_LAYER_ORDER)[number];

export interface MeasurementGraphLayer {
  measurement: MeasurementStructure;
  knownInputs: KnownInput[];
  scientificOnlyConstructs: ScientificOnlyConstruct[];
}

export interface DesignGraphLayer {
  causalDesign: CausalDesign;
  structuralPlan: StructuralPlan;
}

export interface SpecificationGraphLayer {
  modelSpec: StatisticalModelSpecData;
}

export interface FitGraphLayer {
  posterior: PosteriorData;
  edgePosteriors: Record<string, EdgePosterior>;
  persistencePosteriors: Record<string, EdgePosterior>;
}

export interface SimulationGraphLayer {
  result: AnalysisSimulationResult;
}

/**
 * Artifact-backed data available to one causal graph.
 *
 * Structure is the sole topology owner. Optional fields are cumulative backend
 * materializations whose renderers may annotate that topology but never rebuild it.
 */
export interface LayeredCausalGraphModel {
  structure: LatentStructure;
  measurement?: MeasurementGraphLayer;
  design?: DesignGraphLayer;
  specification?: SpecificationGraphLayer;
  fit?: FitGraphLayer;
  simulation?: SimulationGraphLayer;
}

export function assertCumulativeGraphLayers(model: LayeredCausalGraphModel): void {
  const requirements: Array<{
    layer: Exclude<CausalGraphLayerId, "structure" | "measurement">;
    available: boolean;
    requires: Exclude<CausalGraphLayerId, "structure" | "simulation">;
    requirementAvailable: boolean;
  }> = [
    {
      layer: "design",
      available: model.design != null,
      requires: "measurement",
      requirementAvailable: model.measurement != null,
    },
    {
      layer: "specification",
      available: model.specification != null,
      requires: "design",
      requirementAvailable: model.design != null,
    },
    {
      layer: "fit",
      available: model.fit != null,
      requires: "specification",
      requirementAvailable: model.specification != null,
    },
    {
      layer: "simulation",
      available: model.simulation != null,
      requires: "fit",
      requirementAvailable: model.fit != null,
    },
  ];

  for (const requirement of requirements) {
    if (requirement.available && !requirement.requirementAvailable) {
      throw new Error(
        `The '${requirement.layer}' graph layer requires '${requirement.requires}' to be materialized.`,
      );
    }
  }
}

export function availableGraphLayers(model: LayeredCausalGraphModel): CausalGraphLayerId[] {
  assertCumulativeGraphLayers(model);
  return CAUSAL_GRAPH_LAYER_ORDER.filter((layer) => layer === "structure" || model[layer] != null);
}

export function causalEdgeKey(cause: string, effect: string, lagged: boolean): string {
  return `${cause}→${effect}@${lagged ? "lag1" : "same"}`;
}

export type EdgeDesignDisposition = Extract<
  StructuralDisposition,
  "retained_edge" | "projected_edge"
>;

/** Resolve total backend edge dispositions by semantic edge identity. */
export function deriveEdgeDesignDispositions(
  structuralPlan: StructuralPlan,
): Map<string, EdgeDesignDisposition> {
  const dispositionBySourceId = new Map(
    structuralPlan.dispositions.map((item) => [item.source_id, item] as const),
  );
  const result = new Map<string, EdgeDesignDisposition>();

  for (const [sourceId, edge] of Object.entries(structuralPlan.semantics.edges)) {
    const sourceDisposition = dispositionBySourceId.get(sourceId);
    if (!sourceDisposition || sourceDisposition.source_kind !== "edge") {
      throw new Error(`StructuralPlan is missing an edge disposition for '${sourceId}'.`);
    }
    if (
      sourceDisposition.disposition !== "retained_edge" &&
      sourceDisposition.disposition !== "projected_edge"
    ) {
      throw new Error(
        `StructuralPlan gave edge '${sourceId}' the invalid disposition '${sourceDisposition.disposition}'.`,
      );
    }
    const key = causalEdgeKey(edge.cause, edge.effect, edge.lagged);
    if (result.has(key)) {
      throw new Error(`StructuralPlan contains duplicate semantic edge '${key}'.`);
    }
    result.set(key, sourceDisposition.disposition);
  }

  return result;
}
