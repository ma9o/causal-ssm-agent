import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import type {
  Stage0Data,
  Stage1aData,
  Stage1bData,
  Stage2Data,
  Stage4Data,
  Stage4bData,
  Stage5aData,
  Stage5bData,
  Stage6Data,
} from "@causal-ssm/api-types";
import { normalizeStage3Data } from "@/components/stages/validation/__fixtures__/normalize-stage3";
import { type AllStageData, generateMarkdown } from "./generate-markdown";

// --- Doctolib fixtures ---
import doctolibS0 from "../../../../../data/DOCTOLIB/run/stage-0.json";
import doctolibS1a from "../../../../../data/DOCTOLIB/run/stage-1a.json";
import doctolibS1b from "../../../../../data/DOCTOLIB/run/stage-1b.json";
import doctolibS2 from "../../../../../data/DOCTOLIB/run/stage-2.json";
import doctolibS3 from "../../../../../data/DOCTOLIB/run/stage-3.json";
import doctolibS4 from "../../../../../data/DOCTOLIB/run/stage-4.json";
import doctolibS4b from "../../../../../data/DOCTOLIB/run/stage-4b.json";
import doctolibS5a from "../../../../../data/DOCTOLIB/run/stage-5a.json";
import doctolibS5b from "../../../../../data/DOCTOLIB/run/stage-5b.json";
import doctolibS6 from "../../../../../data/DOCTOLIB/run/stage-6.json";

// --- Default fixtures ---
import defaultS0 from "../../../../../data/DEFAULT/run/stage-0.json";
import defaultS1a from "../../../../../data/DEFAULT/run/stage-1a.json";
import defaultS1b from "../../../../../data/DEFAULT/run/stage-1b.json";
import defaultS2 from "../../../../../data/DEFAULT/run/stage-2.json";
import defaultS3 from "../../../../../data/DEFAULT/run/stage-3.json";
import defaultS4 from "../../../../../data/DEFAULT/run/stage-4.json";
import defaultS4b from "../../../../../data/DEFAULT/run/stage-4b.json";
import defaultS5a from "../../../../../data/DEFAULT/run/stage-5a.json";
import defaultS5b from "../../../../../data/DEFAULT/run/stage-5b.json";
import defaultS6 from "../../../../../data/DEFAULT/run/stage-6.json";

function buildAllStageData(
  s0: unknown, s1a: unknown, s1b: unknown, s2: unknown, s3: unknown,
  s4: unknown, s4b: unknown, s5a: unknown, s5b: unknown, s6: unknown,
): AllStageData {
  return {
    "stage-0": s0 as Stage0Data,
    "stage-1a": s1a as Stage1aData,
    "stage-1b": s1b as Stage1bData,
    "stage-2": s2 as Stage2Data,
    "stage-3": normalizeStage3Data(s3),
    "stage-4": s4 as Stage4Data,
    "stage-4b": s4b as Stage4bData,
    "stage-5a": s5a as Stage5aData,
    "stage-5b": s5b as Stage5bData,
    "stage-6": s6 as Stage6Data,
  };
}

const datasets: Record<string, AllStageData> = {
  doctolib: buildAllStageData(
    doctolibS0, doctolibS1a, doctolibS1b, doctolibS2, doctolibS3,
    doctolibS4, doctolibS4b, doctolibS5a, doctolibS5b, doctolibS6,
  ),
  default: buildAllStageData(
    defaultS0, defaultS1a, defaultS1b, defaultS2, defaultS3,
    defaultS4, defaultS4b, defaultS5a, defaultS5b, defaultS6,
  ),
};

function MarkdownReport({ datasetName }: { datasetName: string }) {
  const data = datasets[datasetName] ?? {};
  const md = generateMarkdown(data, datasetName);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(md).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const handleDownload = () => {
    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pipeline-report-${datasetName}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 12, alignItems: "center" }}>
        <strong style={{ fontSize: 14 }}>
          {datasetName} &mdash; {md.length.toLocaleString()} chars, {md.split("\n").length} lines
        </strong>
        <button
          onClick={handleCopy}
          style={{
            padding: "4px 12px",
            fontSize: 12,
            border: "1px solid #ccc",
            borderRadius: 4,
            cursor: "pointer",
            background: copied ? "#d4edda" : "#f8f9fa",
          }}
        >
          {copied ? "Copied!" : "Copy"}
        </button>
        <button
          onClick={handleDownload}
          style={{
            padding: "4px 12px",
            fontSize: 12,
            border: "1px solid #ccc",
            borderRadius: 4,
            cursor: "pointer",
            background: "#f8f9fa",
          }}
        >
          Download .md
        </button>
      </div>
      <pre
        style={{
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          fontFamily: "ui-monospace, 'Cascadia Code', 'Source Code Pro', Menlo, Consolas, monospace",
          fontSize: 11,
          lineHeight: 1.5,
          padding: 16,
          background: "#f6f8fa",
          border: "1px solid #d0d7de",
          borderRadius: 6,
          maxHeight: "80vh",
          overflow: "auto",
        }}
      >
        {md}
      </pre>
    </div>
  );
}

const meta = {
  title: "Export/Markdown Report",
  component: MarkdownReport,
  parameters: {
    layout: "padded",
  },
} satisfies Meta<typeof MarkdownReport>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Doctolib: Story = {
  args: { datasetName: "doctolib" },
};

export const Default: Story = {
  args: { datasetName: "default" },
};
