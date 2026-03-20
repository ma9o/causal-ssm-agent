import { Card, CardContent } from "@/components/ui/card";
import type { InferenceStructureResult } from "@causal-ssm/api-types";

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
    <div className="space-y-2">
      <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="flex h-2.5 overflow-hidden rounded-full bg-muted">
        {kalmanCount > 0 && <div className="h-full bg-success" style={{ flex: kalmanCount }} />}
        {particleCount > 0 && <div className="h-full bg-warning" style={{ flex: particleCount }} />}
      </div>
      <div className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-sm text-muted-foreground">
        <span className="inline-block h-2.5 w-2.5 rounded-full bg-success" />
        <span className="tabular-nums text-foreground">{kalmanCount}</span>
        <span>{kalmanLabel}</span>
        <span className="mx-1 text-muted-foreground/40">|</span>
        <span className="inline-block h-2.5 w-2.5 rounded-full bg-warning" />
        <span className="tabular-nums text-foreground">{particleCount}</span>
        <span>{particleLabel}</span>
      </div>
    </div>
  );
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
      <CardContent className="space-y-4 py-4">
        <div className="font-medium">Inference Structure</div>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-1">
            <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Likelihood path</div>
            <div className="text-sm text-foreground">{PATH_LABELS[inferenceStructure.likelihood_path]}</div>
          </div>
          <div className="space-y-1">
            <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Auto method</div>
            <div className="text-sm text-foreground">{METHOD_LABELS[inferenceStructure.auto_method]}</div>
          </div>
        </div>
        <div className="space-y-3">
          <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            First-pass Rao-Blackwellization
          </div>
          {firstPass.status === "active" ? (
            <div className="grid gap-4 md:grid-cols-2">
              <PartitionSummary
                label="Latents"
                kalmanCount={latentKalman.length}
                particleCount={latentParticle.length}
                kalmanLabel="Kalman"
                particleLabel="Particle"
              />
              <PartitionSummary
                label="Observed channels"
                kalmanCount={obsKalman.length}
                particleCount={obsParticle.length}
                kalmanLabel="Kalman-side"
                particleLabel="Particle-side"
              />
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">
              <span className="font-medium text-foreground">Inactive.</span>{" "}
              {firstPassReason}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
