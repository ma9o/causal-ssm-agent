import type { Stage4Data } from "@causal-ssm/api-types";
import type { ModelMessage } from "ai";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStage4Data(value: unknown): value is Stage4Data {
  return (
    isRecord(value) &&
    isRecord(value.model_spec) &&
    Array.isArray(value.model_spec.likelihoods) &&
    Array.isArray(value.model_spec.parameters) &&
    isRecord(value.authored_priors)
  );
}

function isModelSpec(value: unknown): value is Stage4Data["model_spec"] {
  return (
    isRecord(value) &&
    Array.isArray(value.likelihoods) &&
    Array.isArray(value.parameters)
  );
}

function formatLikelihoodCards(data: Stage4Data): string {
  const rows = data.model_spec.likelihoods.map((likelihood) => {
    const sourceCount = likelihood.sources?.length ?? 0;
    const sourceNote = sourceCount > 0 ? `, ${sourceCount} source${sourceCount === 1 ? "" : "s"}` : "";
    return `- \`${likelihood.variable}\`: \`${likelihood.distribution}\` + \`${likelihood.link}\`${sourceNote}\n  reasoning: ${likelihood.reasoning}`;
  });
  return rows.length > 0 ? rows.join("\n") : "(none)";
}

function formatLoadingConstraints(data: Stage4Data): string {
  const rows = data.model_spec.parameters
    .filter((parameter) => parameter.role === "loading")
    .map((parameter) => {
      return `- \`${parameter.name}\`: \`${parameter.constraint}\`\n  description: ${parameter.description}`;
    });
  return rows.length > 0 ? rows.join("\n") : "(no loading parameters)";
}

function formatPriorCards(data: Stage4Data): string {
  const rows = data.model_spec.parameters.map((parameter) => {
    const current = data.authored_priors[parameter.name];
    const currentLine = current
      ? `current prior: \`${current.distribution}\` with ${JSON.stringify(current.params)}`
      : "current prior: MISSING";
    return `- \`${parameter.name}\`: role=\`${parameter.role}\`, constraint=\`${parameter.constraint}\`\n  ${currentLine}\n  description: ${parameter.description}`;
  });
  return rows.length > 0 ? rows.join("\n") : "(none)";
}

function formatSearchQueryContext(data: Stage4Data): string {
  const rows = Object.entries(data.search_queries ?? {}).map(([parameter, query]) => {
    return `- \`${parameter}\`: ${query}`;
  });
  if (rows.length === 0) {
    return "";
  }
  return `## Literature Searches Already Used\n\n${rows.join("\n")}\n`;
}

function mergeStage4Context(
  persisted: Stage4Data,
  pendingStagePatch: Record<string, unknown>,
): Stage4Data {
  return {
    ...persisted,
    ...(isModelSpec(pendingStagePatch.model_spec)
      ? { model_spec: pendingStagePatch.model_spec }
      : {}),
    ...(isRecord(pendingStagePatch.authored_priors)
      ? { authored_priors: pendingStagePatch.authored_priors as Stage4Data["authored_priors"] }
      : {}),
    ...(Array.isArray(pendingStagePatch.resolved_priors)
      ? { resolved_priors: pendingStagePatch.resolved_priors as Stage4Data["resolved_priors"] }
      : {}),
    ...(isRecord(pendingStagePatch.search_queries)
      ? { search_queries: pendingStagePatch.search_queries as Stage4Data["search_queries"] }
      : {}),
    ...(isRecord(pendingStagePatch.prior_predictive_samples)
      ? {
          prior_predictive_samples:
            pendingStagePatch.prior_predictive_samples as Stage4Data["prior_predictive_samples"],
        }
      : {}),
  };
}

const STAGE4_REFINEMENT_SYSTEM = `You are refining an existing Stage 4 model specification and prior set for a continuous-time state-space model.

This is the live refinement path. Reason over the full current Stage 4 state at once.

Most of the specification has already been determined from the causal structure. Your job is to revise the decisions that require statistical judgment and update priors where needed.

Use the current accepted Stage 4 state in the user message for reasoning, but do not rewrite it as if it were undecided. Do not add or remove constructs, edges, indicators, or parameters unless the user explicitly asks.

Use validate_model with the live refinement contract:
- Submit {"model_spec": <complete ModelSpec>} when revising likelihood choices or loading constraints.
- Submit {"priors": {...}} when revising priors.
- Never mix model_spec and priors in the same call.
- Accepted state is retained server-side. Resend only the fields you changed.

Once you get "VALID", stop immediately.`;

function buildStage4RefinementUserMessage(
  data: Stage4Data,
  pendingStagePatch: Record<string, unknown>,
): string {
  const pendingFields = Object.keys(pendingStagePatch);
  const pendingOverlayNote =
    pendingFields.length > 0
      ? `Unsaved patch fields already applied to the context below: ${pendingFields.map((field) => `\`${field}\``).join(", ")}.`
      : "No unsaved stage patch is currently applied.";
  const searchQuerySection = formatSearchQueryContext(data);

  return `LIVE STAGE 4 CONTEXT (machine-generated)

This live refinement prompt is intentionally broad. All current Stage 4 decisions are shown together so you can repair or revise the current authored state.

${pendingOverlayNote}

## Your Decisions

### 1. Likelihood Choices

Current accepted likelihood choices are shown below. If you revise them, your next \`validate_model\` call must submit a complete \`model_spec\`.

${formatLikelihoodCards(data)}

### 2. Loading Constraints

Current accepted loading constraints are shown below. If you revise them, include the revised loading parameter entries in the complete \`model_spec\` you submit.

${formatLoadingConstraints(data)}

### 3. Parameter Prior Cards

Current prior state for every parameter is shown below. If you revise priors, submit only the changed priors.

${formatPriorCards(data)}

${searchQuerySection}---

\`validate_model\` is stateful. You do not need to resend unchanged fields after a rejection.

Typical sequence:

1. Revise likelihood choices and/or loading constraints by submitting a complete \`model_spec\` only.
2. Revise priors by submitting \`priors\` only.

Never combine \`model_spec\` and \`priors\` in the same tool call. After a failure, only resend the fields you changed.`;
}

export function buildRefinementContextMessages(
  stageId: string,
  stageData: unknown,
  pendingStagePatch: Record<string, unknown>,
): ModelMessage[] {
  if (stageId !== "stage-4" || !isStage4Data(stageData)) {
    return [];
  }

  const merged = mergeStage4Context(stageData, pendingStagePatch);
  return [
    { role: "system", content: STAGE4_REFINEMENT_SYSTEM },
    {
      role: "user",
      content: buildStage4RefinementUserMessage(merged, pendingStagePatch),
    },
  ];
}
