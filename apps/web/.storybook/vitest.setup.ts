import { setProjectAnnotations } from "@storybook/nextjs-vite";
import { page } from "@vitest/browser/context";
import { afterEach, beforeAll, expect } from "vitest";
import * as projectAnnotations from "./preview";

// Apply the same decorators/parameters Storybook uses (QueryClientProvider, globals.css)
// so stories render identically under Vitest's browser runner.
const project = setProjectAnnotations([projectAnnotations]);
beforeAll(project.beforeAll);

// Every story selected into the visual project is pixel-diffed against its committed baseline
// in __screenshots__/. Each story test renders only that one story into the page, so capturing
// document.body captures exactly the component. The screenshot name is derived automatically
// from the running story's test name.
afterEach(async () => {
  await expect.element(page.elementLocator(document.body)).toMatchScreenshot();
});
