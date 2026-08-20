import type { StorybookConfig } from "@storybook/nextjs-vite";
import { mergeConfig } from "vite";

const config: StorybookConfig = {
  stories: ["../src/**/*.stories.@(ts|tsx)"],
  addons: [
    "@storybook/addon-docs",
    "@storybook/addon-vitest",
    import.meta.resolve("./svg-materializer/preset.ts"),
  ],
  framework: "@storybook/nextjs-vite",
  staticDirs: ["../public"],
  async viteFinal(baseConfig) {
    return mergeConfig(baseConfig, {
      define: {
        __dirname: JSON.stringify(""),
      },
    });
  },
};

export default config;
