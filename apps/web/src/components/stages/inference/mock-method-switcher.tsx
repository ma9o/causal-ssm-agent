"use client";

import { Badge } from "@/components/ui/badge";
import type { Stage5bData } from "@nof1-causal-lab/api-types";
import { useState } from "react";

type InferenceMethod = "map" | "aux_gibbs" | "particle_mgrad";

const METHODS: { id: InferenceMethod; label: string; disabled: boolean }[] = [
  { id: "map", label: "MAP", disabled: false },
  { id: "aux_gibbs", label: "Aux Gibbs", disabled: false },
  { id: "particle_mgrad", label: "Particle-mGRAD", disabled: true },
];

interface MockMethodSwitcherProps {
  workspaceId: string;
  baseData: Stage5bData;
  onDataChange: (data: Stage5bData) => void;
}

export function MockMethodSwitcher({ workspaceId, baseData, onDataChange }: MockMethodSwitcherProps) {
  const [active, setActive] = useState<InferenceMethod>("map");
  const [auxGibbsData, setAuxGibbsData] = useState<Stage5bData | null>(null);

  const handleSwitch = (method: InferenceMethod) => {
    if (method === active) return;
    setActive(method);
    if (method === "map") {
      onDataChange(baseData);
    } else if (method === "aux_gibbs") {
      if (auxGibbsData) {
        onDataChange(auxGibbsData);
      } else {
        fetch(`/api/results/${workspaceId}/stage-5b-aux-gibbs`)
          .then((r) => {
            if (!r.ok) throw new Error(`Aux Gibbs fetch failed: ${r.status}`);
            return r.json();
          })
          .then((d) => {
            setAuxGibbsData(d);
            onDataChange(d);
          })
          .catch((e) => console.error("Failed to fetch Aux Gibbs data:", e));
      }
    }
  };

  return (
    <div className="flex items-center gap-1.5 rounded-md border border-dashed border-muted-foreground/30 bg-muted/30 px-2.5 py-1.5">
      <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground/60">
        Mock
      </span>
      <div className="flex gap-1">
        {METHODS.map((m) => (
          <button
            key={m.id}
            type="button"
            onClick={() => !m.disabled && handleSwitch(m.id)}
            disabled={m.disabled}
            className="focus-visible:outline-none"
          >
            <Badge
              variant={active === m.id ? "default" : "outline"}
              className={
                m.disabled
                  ? "cursor-not-allowed opacity-40"
                  : active === m.id
                    ? "cursor-default"
                    : "cursor-pointer hover:bg-accent"
              }
            >
              {m.label}
            </Badge>
          </button>
        ))}
      </div>
    </div>
  );
}
