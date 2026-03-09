const PREFECT_API = process.env.PREFECT_API_URL ?? "http://localhost:4200/api";

interface FlowRun {
  id: string;
  state: { type: string; name: string };
  parameters: Record<string, unknown>;
}

interface TaskRun {
  id: string;
  name: string;
  state: { type: string; name: string };
}

async function prefectFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${PREFECT_API}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Prefect API ${init?.method ?? "GET"} ${path}: ${res.status} ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function getFlowRun(runId: string): Promise<FlowRun> {
  return prefectFetch<FlowRun>(`/flow_runs/${runId}`);
}

export async function getTaskRuns(flowRunId: string): Promise<TaskRun[]> {
  return prefectFetch<TaskRun[]>("/task_runs/filter", {
    method: "POST",
    body: JSON.stringify({
      flow_runs: { id: { any_: [flowRunId] } },
    }),
  });
}

export async function getDeploymentId(name = "causal-inference"): Promise<string> {
  const deployments = await prefectFetch<Array<{ id: string }>>(
    "/deployments/filter",
    {
      method: "POST",
      body: JSON.stringify({
        deployments: { name: { any_: [name] } },
      }),
    },
  );
  if (!deployments.length) throw new Error(`Deployment "${name}" not found`);
  return deployments[0].id;
}

export async function triggerRun(
  deploymentId: string,
  parameters: Record<string, unknown>,
): Promise<string> {
  const run = await prefectFetch<{ id: string }>(
    `/deployments/${deploymentId}/create_flow_run`,
    {
      method: "POST",
      body: JSON.stringify({ parameters }),
    },
  );
  return run.id;
}
