import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { TooltipProvider } from "@/components/ui/tooltip";
import { RetryIndicator } from "./retry-indicator";

const mockRetries = [
  {
    attempt: 1,
    failed_params: ["sigma_cardiovascular_risk", "rho_metabolic_health"],
    feedback: "Prior for sigma_cardiovascular_risk too wide — posterior unconstrained. Tighten to HalfNormal(0.5).",
  },
  {
    attempt: 2,
    failed_params: ["rho_metabolic_health"],
    feedback: "rho_metabolic_health hitting boundary at 1.0 — use Beta(2, 2) to keep away from unit root.",
  },
];

const meta = {
  title: "Stages/ModelSpec/RetryIndicator",
  component: RetryIndicator,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-md mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof RetryIndicator>;

export default meta;
type Story = StoryObj<typeof meta>;

export const WithRetries: Story = {
  render: () => <RetryIndicator retries={mockRetries} />,
};

export const Empty: Story = {
  render: () => <RetryIndicator retries={[]} />,
};
