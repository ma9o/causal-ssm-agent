import type {
  Stage1aData,
  Stage1bData,
  Stage4Data,
  Stage5bData,
} from "@nof1-causal-lab/api-types";
import type { UIMessage } from "ai";
import type { EdgePosterior } from "@/components/dag/intervention-dag-types";
import { parseFixedEffect } from "@/lib/utils/ssm-latex";
import { extractLatestStage6FollowUpSimulation } from "./stage-6-follow-up";
import type { Stage6DagScene } from "./stage-6-showcase";

function normalizeConstructLabel(value: string): string {
  return value.trim().toLowerCase().replace(/[\s_]+/g, " ");
}

function resolveConstructName(label: string, constructNames: string[]): string | null {
  const normalizedLabel = normalizeConstructLabel(label);
  return (
    constructNames.find(
      (constructName) => normalizeConstructLabel(constructName) === normalizedLabel,
    ) ?? null
  );
}

function parseFixedEffectDescription(
  description: string,
  constructNames: string[],
): { source: string; target: string } | null {
  const match = /^Effect of (.+?) on (.+?)(?: \(|$)/.exec(description);
  if (!match) {
    return null;
  }

  const source = resolveConstructName(match[1], constructNames);
  const target = resolveConstructName(match[2], constructNames);
  if (!source || !target) {
    return null;
  }

  return { source, target };
}

function buildEdgePosteriors({
  stage1a,
  stage4,
  stage5b,
}: {
  stage1a?: Stage1aData;
  stage4?: Stage4Data;
  stage5b?: Stage5bData;
}): Record<string, EdgePosterior> {
  if (!stage1a) {
    return {};
  }

  const constructNames = stage1a.latent_model.constructs.map((construct) => construct.name);
  const parametersByName = new Map(
    (stage4?.model_spec.parameters ?? []).map((parameter) => [parameter.name, parameter]),
  );
  const marginals = stage5b?.posterior_marginals ?? [];
  const edgePosteriors: Record<string, EdgePosterior> = {};

  for (const marginal of marginals) {
    if (!marginal.parameter.startsWith("beta_")) {
      continue;
    }

    const parameter = parametersByName.get(marginal.parameter);
    const parsed =
      (parameter?.description
        ? parseFixedEffectDescription(parameter.description, constructNames)
        : null) ?? parseFixedEffect(marginal.parameter, constructNames);
    if (!parsed) {
      continue;
    }

    edgePosteriors[`${parsed.source}→${parsed.target}`] = {
      mean: marginal.mean,
      ci_lower: marginal.hdi_3,
      ci_upper: marginal.hdi_97,
    };
  }

  return edgePosteriors;
}

export function buildStage6DagScene({
  stage1a,
  stage1b,
  stage4,
  stage5b,
  refinementMessages = [],
  height = "600px",
}: {
  stage1a?: Stage1aData;
  stage1b?: Stage1bData;
  stage4?: Stage4Data;
  stage5b?: Stage5bData;
  refinementMessages?: UIMessage[];
  height?: string;
}): Stage6DagScene | undefined {
  if (!stage1a) {
    return undefined;
  }

  const followUpSimulation = extractLatestStage6FollowUpSimulation(refinementMessages);
  const edgePosteriors = buildEdgePosteriors({ stage1a, stage4, stage5b });

  return {
    constructs: stage1a.latent_model.constructs,
    edges: stage1a.latent_model.edges,
    indicators: stage1b?.causal_spec.measurement.indicators,
    edgePosteriors,
    requestedHorizonDays: followUpSimulation?.input?.query?.horizon_days ?? undefined,
    simulationResult: followUpSimulation?.output ?? undefined,
    height,
  };
}
