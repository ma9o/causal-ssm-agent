import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { constructs, edges, indicators } from "../__fixtures__/dag-base-fixtures";
import { makeMockSimulate, synthesizeMockScenarios } from "./dev-mock-scenario";
import { InteractiveDag } from "./interactive-dag";

const outcome = constructs.find((c) => c.is_outcome)?.name ?? "affective_state";
const { baseline, interventions } = synthesizeMockScenarios(constructs, edges, indicators, outcome);
const intervention = interventions[0]?.result ?? baseline.result;

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

/** No-intervention baseline: the reference world, every node on its baseline line. */
export const Baseline: Story = {
  args: { constructs, edges, indicators, result: baseline.result },
};

/** Read-only living DAG with an intervention applied (no do() editing). */
export const ReadOnly: Story = {
  args: { constructs, edges, indicators, result: intervention },
};

/** Interactive: type a do() value on any node to re-simulate (mock backend). */
export const Interactive: Story = {
  args: {
    constructs,
    edges,
    indicators,
    result: intervention,
    onSimulate: makeMockSimulate(intervention),
  },
};
