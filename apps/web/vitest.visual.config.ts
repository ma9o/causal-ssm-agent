import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { storybookTest } from "@storybook/addon-vitest/vitest-plugin";
import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

const root = dirname(fileURLToPath(import.meta.url));

// Standalone visual-regression config — intentionally NOT wired into vitest.config.ts so that
// `bun run test` stays node-only and fast. Run explicitly via `bun run test:visual`.
// Only stories tagged `visual` are turned into tests (see `tags.include`), each one screenshotted
// by the afterEach hook in .storybook/vitest.setup.ts.
export default defineConfig({
  plugins: [
    storybookTest({
      configDir: join(root, ".storybook"),
      tags: { include: ["visual"], exclude: [], skip: [] },
    }),
  ],
  resolve: {
    alias: { "@": join(root, "src") },
  },
  test: {
    name: "visual",
    setupFiles: [join(root, ".storybook/vitest.setup.ts")],
    browser: {
      enabled: true,
      headless: true,
      provider: playwright(),
      instances: [{ browser: "chromium", viewport: { width: 1280, height: 720 } }],
      expect: {
        toMatchScreenshot: {
          comparatorName: "pixelmatch",
          // Allow up to 2% of pixels to differ — absorbs sub-pixel anti-aliasing jitter
          // without masking real layout/geometry regressions.
          comparatorOptions: { allowedMismatchedPixelRatio: 0.02 },
        },
      },
    },
  },
});
