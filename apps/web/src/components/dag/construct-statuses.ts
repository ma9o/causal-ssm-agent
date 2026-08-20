import type { CausalDesign, StructuralPlan } from "@nof1-causal-lab/api-types";
import type { ConstructStatus } from "./structure-dag";

/** Map the backend's total construct dispositions onto the DAG presentation vocabulary. */
export function deriveConstructStatuses(
  design: CausalDesign,
  structuralPlan: StructuralPlan,
): Record<string, ConstructStatus> {
  const blocking = new Set<string>();
  for (const [treatment, status] of Object.entries(
    design.identifiability?.non_identifiable_treatments ?? {},
  )) {
    blocking.add(treatment);
    for (const construct of status?.confounders ?? []) {
      blocking.add(construct);
    }
  }

  const dispositionById = new Map(
    structuralPlan.dispositions.map((item) => [item.source_id, item] as const),
  );
  const statusByName = new Map<string, ConstructStatus>();
  for (const [sourceId, construct] of Object.entries(structuralPlan.semantics.constructs)) {
    const item = dispositionById.get(sourceId);
    if (!item || item.source_kind !== "construct") {
      throw new Error(`StructuralPlan is missing a construct disposition for '${construct.name}'.`);
    }

    let status: ConstructStatus;
    switch (item.disposition) {
      case "retained_state":
      case "known_input":
        status = "observed";
        break;
      case "marginalized":
      case "identification_only":
        status = "marginalized";
        break;
      default:
        throw new Error(
          `StructuralPlan gave construct '${construct.name}' the invalid disposition '${item.disposition}'.`,
        );
    }
    statusByName.set(construct.name, status);
  }

  return Object.fromEntries(
    design.latent.constructs.map((construct) => {
      const status = statusByName.get(construct.name);
      if (!status) {
        throw new Error(`StructuralPlan has no semantics for construct '${construct.name}'.`);
      }
      return [construct.name, blocking.has(construct.name) ? "blocking" : status];
    }),
  );
}
