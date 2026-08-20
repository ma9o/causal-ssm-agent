import { fileURLToPath } from "node:url";
import type { PresetProperty } from "storybook/internal/types";
import { installSvgMaterializer } from "./server.ts";

export const experimental_devServer: PresetProperty<"experimental_devServer"> = (app) =>
  installSvgMaterializer(app);

export const previewAnnotations: PresetProperty<"previewAnnotations"> = (entries = []) => [
  ...entries,
  fileURLToPath(import.meta.resolve("./preview.tsx")),
];
