"use client";

import type { Stage6Data } from "@causal-ssm/api-types";
import type { ReactNode } from "react";
import Stage6Content from "./stage-6-content";

export default function Stage6Showcase({
  data,
  dag,
}: {
  data: Stage6Data;
  dag?: ReactNode;
}) {
  return (
    <div className="space-y-4">
      <Stage6Content
        data={data}
      />
      {dag ? (
          dag
      ) : null}
    </div>
  );
}
