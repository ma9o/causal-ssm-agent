import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage2Data } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { StageSection } from "../stage-section";
import Stage2Content from "./stage-2-content";
import { Stage2RunningView } from "./stage-2-running-content";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-2.json";

const stage = STAGES.find((s) => s.id === "stage-2")!;
const data = fixture as Stage2Data;

/* ── Mock data for running state ── */

function makeWorkers(completed: number, running: number, failed: number, pending: number) {
  let idx = 0;
  return [
    ...Array.from({ length: completed }, (_, i) => ({ id: `w-done-${i}`, name: `worker-${idx++}`, state: "completed" as const })),
    ...Array.from({ length: running }, (_, i) => ({ id: `w-run-${i}`, name: `worker-${idx++}`, state: "running" as const })),
    ...Array.from({ length: failed }, (_, i) => ({ id: `w-fail-${i}`, name: `worker-${idx++}`, state: "failed" as const })),
    ...Array.from({ length: pending }, (_, i) => ({ id: `w-pend-${i}`, name: `worker-${idx++}`, state: "pending" as const })),
  ];
}

const mockWorkers = makeWorkers(8, 3, 1, 4);
const mockWorkers1k = makeWorkers(620, 45, 12, 323);

const mockLogs = [
  { id: "l1", created: "", name: "", level: 20, message: "Starting extraction for indicator: revenue_growth", timestamp: "2025-01-15T10:30:01Z", flow_run_id: "", task_run_id: null },
  { id: "l2", created: "", name: "", level: 20, message: "Worker worker-0 completed — 3 LLM calls, 12 extractions", timestamp: "2025-01-15T10:30:05Z", flow_run_id: "", task_run_id: null },
  { id: "l3", created: "", name: "", level: 30, message: "Rate limit approaching: 380/450 rpm", timestamp: "2025-01-15T10:30:08Z", flow_run_id: "", task_run_id: null },
  { id: "l4", created: "", name: "", level: 20, message: "Worker worker-1 completed — 5 LLM calls, 18 extractions", timestamp: "2025-01-15T10:30:12Z", flow_run_id: "", task_run_id: null },
  { id: "l5", created: "", name: "", level: 40, message: "Worker worker-11 failed: context window exceeded (32k tokens)", timestamp: "2025-01-15T10:30:15Z", flow_run_id: "", task_run_id: null },
  { id: "l6", created: "", name: "", level: 20, message: "Worker worker-5 completed — 2 LLM calls, 8 extractions", timestamp: "2025-01-15T10:30:18Z", flow_run_id: "", task_run_id: null },
];

const meta = {
  title: "Pipeline/Stages/2 – Data Extraction",
  component: Stage2Content,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof Stage2Content>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Pending: Story = {
  render: () => (
    <StageSection number={stage.number} title={stage.label} status="pending" context={stage.description} />
  ),
};

export const Running: Story = {
  render: () => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="running"
      context={stage.description}
      loadingHint={stage.loadingHint}
      runningContent={<Stage2RunningView workers={mockWorkers} logs={mockLogs} rpm={285} />}
    />
  ),
};

export const RunningHighRpm: Story = {
  name: "Running (High RPM)",
  render: () => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="running"
      context={stage.description}
      loadingHint={stage.loadingHint}
      runningContent={<Stage2RunningView workers={mockWorkers} logs={mockLogs} rpm={420} />}
    />
  ),
};

export const Running1kWorkers: Story = {
  name: "Running (1000 Workers)",
  render: () => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="running"
      context={stage.description}
      loadingHint={stage.loadingHint}
      runningContent={<Stage2RunningView workers={mockWorkers1k} logs={mockLogs} rpm={440} />}
    />
  ),
};

export const Completed: Story = {
  render: () => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="completed"
      outcome={data.outcome}
      context={stage.description}
      elapsedMs={45_200}
    >
      <Stage2Content data={data} />
    </StageSection>
  ),
};

export const Failed: Story = {
  render: () => (
    <StageSection number={stage.number} title={stage.label} status="failed" context={stage.description} />
  ),
};
