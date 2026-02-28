import { apiFetch } from "./client";

interface PrefectDeployment {
  id: string;
  name: string;
}

interface PrefectFlowRun {
  id: string;
  state: { type: string; name: string };
}

export async function getDeploymentId(name = "causal-inference"): Promise<string> {
  const deployments = await apiFetch<PrefectDeployment[]>("/prefect/deployments/filter", {
    method: "POST",
    body: JSON.stringify({
      deployments: { name: { any_: [name] } },
    }),
  });
  if (deployments.length === 0) {
    throw new Error(`Deployment "${name}" not found`);
  }
  return deployments[0].id;
}

export async function triggerRun(
  deploymentId: string,
  parameters: Record<string, unknown>,
): Promise<string> {
  const run = await apiFetch<PrefectFlowRun>(
    `/prefect/deployments/${deploymentId}/create_flow_run`,
    {
      method: "POST",
      body: JSON.stringify({ parameters }),
    },
  );
  return run.id;
}

