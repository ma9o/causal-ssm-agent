import type { Preview } from "@storybook/nextjs-vite";
import "katex/dist/katex.min.css";
import "../src/app/globals.css";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: false, refetchOnWindowFocus: false } },
});

const preview: Preview = {
  decorators: [
    (Story) =>
      React.createElement(QueryClientProvider, { client: queryClient }, React.createElement(Story)),
  ],
  parameters: {
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
    options: {
      // Pipeline surfaces first (in stage order, Panel before its widgets),
      // then the reusable layers. Everything unlisted falls back to alphabetical.
      storySort: {
        method: "alphabetical",
        order: [
          "Pipeline",
          [
            "StageHeader",
            "StageSection",
            "Stages",
            [
              "0 – Preprocess",
              ["Panel", "*"],
              "1a – Latent Model",
              ["Panel", "*"],
              "1b – Measurement",
              ["Panel", "*"],
              "2 – Data Extraction",
              ["Panel", "*"],
              "3 – Validation",
              ["Panel", "*"],
              "4 – Model Specification",
              ["Panel", "*"],
              "5b – Inference & Diagnostics",
              ["Panel", "*"],
              "6 – Treatment Effects",
              ["Panel", "*"],
              "*",
            ],
          ],
          "Charts",
          "UI",
          "Landing",
          "*",
        ],
      },
    },
  },
};

export default preview;
