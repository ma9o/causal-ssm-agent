"use client";

import { Badge } from "@/components/ui/badge";
import type { Stage5bData } from "@causal-ssm/api-types";
import { useState } from "react";

type InferenceMethod = "laplace_em" | "nuts_da" | "particle_filter";

const METHODS: { id: InferenceMethod; label: string; disabled: boolean }[] = [
  { id: "laplace_em", label: "Laplace-EM", disabled: false },
  { id: "nuts_da", label: "NUTS-DA", disabled: false },
  { id: "particle_filter", label: "Particle Filter", disabled: true },
];

interface MockMethodSwitcherProps {
  userId: string;
  baseData: Stage5bData;
  onDataChange: (data: Stage5bData) => void;
}

export function MockMethodSwitcher({ userId, baseData, onDataChange }: MockMethodSwitcherProps) {
  const [active, setActive] = useState<InferenceMethod>("laplace_em");
  const [nutsdaData, setNutsdaData] = useState<Stage5bData | null>(null);

  const handleSwitch = (method: InferenceMethod) => {
    if (method === active) return;
    setActive(method);
    if (method === "laplace_em") {
      onDataChange(baseData);
    } else if (method === "nuts_da") {
      if (nutsdaData) {
        onDataChange(nutsdaData);
      } else {
        fetch(`/api/results/${userId}/stage-5b-nutsda`)
          .then((r) => (r.ok ? r.json() : null))
          .then((d) => {
            if (d) {
              setNutsdaData(d);
              onDataChange(d);
            }
          })
          .catch(() => {});
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
