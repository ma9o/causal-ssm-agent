"use client";

import { Badge } from "@/components/ui/badge";
import type { PosteriorData } from "@nof1-causal-lab/api-types";
import { useState } from "react";

type InferenceMethod = "map" | "aux_kalman_mcmc" | "pit_particle_mgrad";

const METHODS: { id: InferenceMethod; label: string; disabled: boolean }[] = [
  { id: "map", label: "MAP", disabled: false },
  { id: "aux_kalman_mcmc", label: "Auxiliary Kalman MCMC", disabled: false },
  { id: "pit_particle_mgrad", label: "PIT Particle-mGRAD", disabled: true },
];

interface MockMethodSwitcherProps {
  workspaceId: string;
  baseData: PosteriorData;
  onDataChange: (data: PosteriorData) => void;
}

export function MockMethodSwitcher({
  workspaceId,
  baseData,
  onDataChange,
}: MockMethodSwitcherProps) {
  const [active, setActive] = useState<InferenceMethod>("map");
  const [auxKalmanMCMCData, setAuxKalmanMCMCData] = useState<PosteriorData | null>(null);

  const handleSwitch = (method: InferenceMethod) => {
    if (method === active) return;
    setActive(method);
    if (method === "map") {
      onDataChange(baseData);
    } else if (method === "aux_kalman_mcmc") {
      if (auxKalmanMCMCData) {
        onDataChange(auxKalmanMCMCData);
      } else {
        fetch(`/api/artifacts/${workspaceId}/posterior/view`)
          .then((r) => {
            if (!r.ok) throw new Error(`Auxiliary Kalman MCMC fetch failed: ${r.status}`);
            return r.json();
          })
          .then((d) => {
            setAuxKalmanMCMCData(d);
            onDataChange(d);
          })
          .catch((e) => console.error("Failed to fetch Auxiliary Kalman MCMC data:", e));
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
