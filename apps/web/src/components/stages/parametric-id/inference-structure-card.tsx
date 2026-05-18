import { Card, CardContent } from "@/components/ui/card";
import type { InferenceStructureResult } from "@nof1-causal-lab/api-types";
import type { ReactNode } from "react";

interface InferenceStructureCardProps {
  inferenceStructure: InferenceStructureResult;
}

interface PartitionSummaryProps {
  label: string;
  kalmanCount: number;
  particleCount: number;
  kalmanLabel: string;
  particleLabel: string;
}

function PartitionSummary({
  label,
  kalmanCount,
  particleCount,
  kalmanLabel,
  particleLabel,
}: PartitionSummaryProps) {
  return (
    <SummaryItem
      label={label}
      value={
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-success" />
          <span className="tabular-nums">{kalmanCount}</span>
          <span className="text-muted-foreground">{kalmanLabel}</span>
          <span className="text-muted-foreground/40">/</span>
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-warning" />
          <span className="tabular-nums">{particleCount}</span>
          <span className="text-muted-foreground">{particleLabel}</span>
        </span>
      }
    />
  );
}

function SummaryItem({ label, value }: { label: string; value: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-foreground">{value}</span>
    </span>
  );
}

function SummaryDivider() {
  return <span className="text-muted-foreground/40">|</span>;
}

const PATH_LABELS = {
  kalman: "Kalman",
  composed: "Kalman + Particle",
  particle: "Particle",
} as const;

const METHOD_LABELS = {
  map: "MAP",
  svi: "SVI",
  aux_gibbs: "Auxiliary Gibbs",
  particle_mgrad: "Particle-mGRAD",
} as const;

export function InferenceStructureCard({ inferenceStructure }: InferenceStructureCardProps) {
  const firstPass = inferenceStructure.first_pass_rb;
  const latentKalman = firstPass.latent_variables.filter((v) => v.method === "kalman");
  const latentParticle = firstPass.latent_variables.filter((v) => v.method === "particle");
  const obsKalman = firstPass.obs_variables.filter((v) => v.method === "kalman");
  const obsParticle = firstPass.obs_variables.filter((v) => v.method === "particle");

  return (
    <Card>
      <CardContent className="py-1">
        <div className="flex justify-center overflow-x-auto">
          <div className="flex min-w-max items-center gap-2 whitespace-nowrap text-sm">
            <span className="font-medium text-foreground">Inference Structure</span>
            <SummaryDivider />
            <SummaryItem
              label="Likelihood path"
              value={PATH_LABELS[inferenceStructure.likelihood_path]}
            />
            <SummaryDivider />
            <SummaryItem label="Method" value={METHOD_LABELS[inferenceStructure.auto_method]} />
            <SummaryDivider />
            <span className="inline-flex items-center gap-1.5">
              <span className="text-muted-foreground">First-pass Rao-Blackwellization</span>
              <span className="text-foreground">
                {firstPass.status === "active" ? "Active" : "Inactive"}
              </span>
            </span>
            {firstPass.status === "active" ? (
              <>
                <SummaryDivider />
                <PartitionSummary
                  label="Latents"
                  kalmanCount={latentKalman.length}
                  particleCount={latentParticle.length}
                  kalmanLabel="Kalman"
                  particleLabel="Particle"
                />
                <SummaryDivider />
                <PartitionSummary
                  label="Observed channels"
                  kalmanCount={obsKalman.length}
                  particleCount={obsParticle.length}
                  kalmanLabel="Kalman"
                  particleLabel="Particle"
                />
              </>
            ) : null}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
