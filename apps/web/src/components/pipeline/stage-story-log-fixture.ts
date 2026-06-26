import type { PrefectLogEntry } from "@/lib/prefect-log-client";

const STORY_LOG_MESSAGES = [
  { level: 20, message: "Starting stage execution..." },
  { level: 20, message: "Loading workspace artifacts" },
  { level: 20, message: "Found 1,247 records across 3 data sources" },
  { level: 20, message: "Validating schemas against stage requirements" },
  {
    level: 30,
    message: "Column 'mood_score' has 12% missing values; carrying forward last observed value",
  },
  { level: 20, message: "Normalizing timestamps to UTC" },
  { level: 20, message: "Encoding categorical variables" },
  { level: 20, message: "Submitting model request for stage reasoning" },
  { level: 10, message: "POST /v1/responses model=gpt-5.4 tokens_est=4320" },
  { level: 20, message: "Model response received (3.2s, 1,847 output tokens)" },
  { level: 20, message: "Parsing structured payload" },
  { level: 20, message: "Constructing causal DAG candidate" },
  { level: 20, message: "Checking DAG acyclicity" },
  { level: 20, message: "DAG is acyclic" },
  { level: 20, message: "Running d-separation checks against observed correlations" },
  {
    level: 30,
    message: "Implied independence Sleep->Appetite not supported (p=0.003); flagging for review",
  },
  { level: 20, message: "Projecting temporal graph for identification analysis" },
  { level: 20, message: "Checking identification via ID algorithm" },
  { level: 20, message: "Back-door adjustment set found for exposure->outcome query" },
  { level: 20, message: "Estimating nuisance models for confounder adjustment" },
  { level: 40, message: "Optimizer hit max_iter=200 before convergence" },
  { level: 20, message: "Retrying optimizer with max_iter=500" },
  { level: 20, message: "Optimizer converged on retry (iter=342)" },
  { level: 20, message: "Fitting continuous-time latent state model" },
  { level: 20, message: "Evaluating posterior predictive diagnostics" },
  {
    level: 30,
    message: "One latent factor shows weak identifiability; widening uncertainty interval",
  },
  { level: 20, message: "Computing causal effect summary" },
  { level: 20, message: "Effect estimate finalized" },
  { level: 20, message: "Writing artifacts to workspace storage" },
  { level: 20, message: "Uploading DAG preview asset" },
  { level: 20, message: "Persisting stage metadata" },
  { level: 20, message: "Stage completed successfully" },
] satisfies Pick<PrefectLogEntry, "level" | "message">[];

export function makeStageStoryLog(
  index: number,
  totalCount = STORY_LOG_MESSAGES.length,
): PrefectLogEntry {
  const entry = STORY_LOG_MESSAGES[index % STORY_LOG_MESSAGES.length];
  const timestamp = new Date(Date.now() - (totalCount - index) * 800);
  return {
    id: `story-log-${index}`,
    created: timestamp.toISOString(),
    name: "prefect.flow_runs",
    level: entry.level,
    message: entry.message,
    timestamp: timestamp.toISOString(),
    flow_run_id: "story-flow-run",
    task_run_id: null,
  };
}

export function createStageStoryLogs(count = STORY_LOG_MESSAGES.length): PrefectLogEntry[] {
  return Array.from({ length: count }, (_, index) => makeStageStoryLog(index, count));
}
