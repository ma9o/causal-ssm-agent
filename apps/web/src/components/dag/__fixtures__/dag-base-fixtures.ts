import type { Stage1aData, Stage1bData } from "@nof1-causal-lab/api-types";
import stage1aFixture from "../../../../../../data/DEMO/run/stage-1a.json";
import stage1bFixture from "../../../../../../data/DEMO/run/stage-1b.json";
import type { ConstructStatus } from "../construct-node";

const stage1a = stage1aFixture as unknown as Stage1aData;
const stage1b = stage1bFixture as unknown as Stage1bData;

export const constructs = stage1a.latent_model.constructs;
export const edges = stage1a.latent_model.edges;
export const spec = stage1b.causal_spec;
export const indicators = spec.measurement.indicators;

function deriveIdentifiabilityView(): {
  nodeStatuses: Record<string, ConstructStatus>;
  blockingEdges: Set<string>;
} {
  const statuses: Record<string, ConstructStatus> = {};

  const marginalized = new Set<string>();
  for (const s of Object.values(spec.identifiability?.identifiable_treatments ?? {})) {
    for (const c of s?.marginalized_confounders ?? []) marginalized.add(c);
  }

  const blocking = new Set<string>();
  const blockingEdges = new Set<string>();
  for (const [treatment, s] of Object.entries(
    spec.identifiability?.non_identifiable_treatments ?? {},
  )) {
    blocking.add(treatment);
    for (const c of s?.confounders ?? []) {
      blocking.add(c);
      blockingEdges.add(`${c} ${treatment}`);
    }
  }

  for (const c of spec.latent.constructs) {
    if (blocking.has(c.name)) statuses[c.name] = "blocking";
    else if (marginalized.has(c.name)) statuses[c.name] = "marginalized";
    else statuses[c.name] = "observed";
  }
  return { nodeStatuses: statuses, blockingEdges };
}

const identifiabilityView = deriveIdentifiabilityView();
export const nodeStatuses = identifiabilityView.nodeStatuses;
export const blockingEdges = identifiabilityView.blockingEdges;
