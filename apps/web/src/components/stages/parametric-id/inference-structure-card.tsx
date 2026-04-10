import { Card, CardContent } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { InferenceStructureResult } from "@causal-ssm/api-types";
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
  nuts: "NUTS",
  laplace_em: "Laplace-EM",
  svi: "SVI",
} as const;

const FIRST_PASS_REASON_LABELS = {
  disabled_in_spec: "Disabled in the model spec.",
  interval_summary_support: "Disabled because interval-summary observations require support-aware particle likelihoods.",
  no_executable_partition: "No executable first-pass split exists for the current latent and observation coupling.",
  likelihood_override: "Disabled because the model explicitly requests the Kalman backend.",
} as const;

export function InferenceStructureCard({ inferenceStructure }: InferenceStructureCardProps) {
  const firstPass = inferenceStructure.first_pass_rb;
  const latentKalman = firstPass.latent_variables.filter((v) => v.method === "kalman");
  const latentParticle = firstPass.latent_variables.filter((v) => v.method === "particle");
  const obsKalman = firstPass.obs_variables.filter((v) => v.method === "kalman");
  const obsParticle = firstPass.obs_variables.filter((v) => v.method === "particle");
  const firstPassReason =
    firstPass.inactive_reason == null ? null : FIRST_PASS_REASON_LABELS[firstPass.inactive_reason];

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
            <SummaryItem
              label="Method"
              value={METHOD_LABELS[inferenceStructure.auto_method]}
            />
            <SummaryDivider />
            <span className="inline-flex items-center gap-1.5">
              <span className="text-muted-foreground">First-pass Rao-Blackwellization</span>
              <span className="text-foreground">
                {firstPass.status === "active" ? "Active" : "Inactive"}
              </span>
              {firstPass.status === "inactive" && firstPassReason ? (
                <Tooltip>
                  <TooltipTrigger
                    render={<button type="button" />}
                    aria-label="Show Rao-Blackwellization inactive reason"
                    className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-border text-[10px] font-semibold text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                  >
                    ?
                  </TooltipTrigger>
                  <TooltipContent>
                    <span className="max-w-xs text-xs leading-relaxed">{firstPassReason}</span>
                  </TooltipContent>
                </Tooltip>
              ) : null}
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
