import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import {
  demoBaselineTrace,
  edgePosteriors,
  identifiableTreatments,
  nodeStatuses,
  persistencePosteriors,
} from "../__fixtures__/baseline_report-materialized-fixture";
import { constructs, edges, indicators, knownInputs } from "../__fixtures__/dag-base-fixtures";
import { buildBaselineReportScenarios } from "../../pipeline/output-views/baseline-report-scenarios";
import { InteractiveDag } from "./interactive-dag";

const scenarios = buildBaselineReportScenarios({ trace: demoBaselineTrace });
const firstScenario = scenarios[0]?.result;
const comparisonScenario = scenarios[scenarios.length - 1]?.result;

if (!firstScenario || !comparisonScenario) {
  throw new Error("The DEMO trace must contain materialized simulation scenarios.");
}

const graphArgs = {
  constructs,
  edges,
  indicators,
  knownInputs,
  edgePosteriors,
  persistencePosteriors,
  identifiableTreatments,
  nodeStatuses,
};

const meta: Meta<typeof InteractiveDag> = {
  title: "DAG/Interactive/Living DAG",
  component: InteractiveDag,
  parameters: { layout: "fullscreen" },
  decorators: [
    (Story) => (
      <div style={{ background: "#fafbfc", minHeight: "100vh" }}>
        <div style={{ maxWidth: 1320, margin: "0 auto", padding: "18px 20px 60px" }}>
          <Story />
        </div>
      </div>
    ),
  ],
};
export default meta;

type Story = StoryObj<typeof InteractiveDag>;

/** A production-shaped result with its backend reference and action rollouts. */
export const ReferenceAndAction: Story = {
  args: { ...graphArgs, result: firstScenario },
};

/** Read-only living DAG with an intervention applied (no do() editing). */
export const Intervention: Story = {
  args: { ...graphArgs, result: comparisonScenario },
};
