"use client";

import { Badge } from "@/components/ui/badge";
import { useState, useEffect } from "react";
import type { Stage5Data } from "@causal-ssm/api-types";

type InferenceMethod = "svi" | "nuts_da" | "particle_filter";

const METHODS: { id: InferenceMethod; label: string; disabled: boolean }[] = [
  { id: "nuts_da", label: "NUTS-DA", disabled: false },
  { id: "particle_filter", label: "Particle Filter", disabled: true },
  { id: "svi", label: "SVI", disabled: false },
];

interface MockMethodSwitcherProps {
  baseData: Stage5Data;
  onDataChange: (data: Stage5Data) => void;
}

export function MockMethodSwitcher({ baseData, onDataChange }: MockMethodSwitcherProps) {
  const [active, setActive] = useState<InferenceMethod>("nuts_da");
  const [nutsdaData, setNutsdaData] = useState<Stage5Data | null>(null);

  useEffect(() => {
    fetch("/api/results/mock-run-001/stage-5-nutsda")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        setNutsdaData(d);
        if (d) onDataChange(d);
      })
      .catch(() => {});
  }, [onDataChange]);

  const handleSwitch = (method: InferenceMethod) => {
    if (method === active) return;
    setActive(method);
    if (method === "svi") {
      onDataChange(baseData);
    } else if (method === "nuts_da" && nutsdaData) {
      onDataChange(nutsdaData);
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
