"use client";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import { formatNumber } from "@/lib/utils/format";
import type { Stage4bData } from "@causal-ssm/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import { AlertTriangle, CheckCircle2 } from "lucide-react";
import type { ReactNode } from "react";
import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

type MapGeometryResult = NonNullable<Stage4bData["parametric_id"]["map_geometry"]>;
type MAPCurvatureResult = MapGeometryResult["likelihood_curvature"];
type CurvatureDirection = MAPCurvatureResult["weak_directions"][number];
type CurvatureDirectionLoading = CurvatureDirection["top_loadings"][number];
type CurvatureParameterEntry = MAPCurvatureResult["per_parameter"][number];
type MAPOptimizationRun = MapGeometryResult["starts"][number];
type GeometryVerdict = {
  title: string;
  description: string;
  alertVariant: "default" | "warning" | "destructive";
  badgeVariant: "success" | "warning" | "destructive";
  badgeLabel: string;
};
type ParameterChip = {
  parameter: string;
  label: string;
  status?: "pass" | "warn" | "fail";
};

const directionCol = createColumnHelper<CurvatureDirection>();
const parameterCol = createColumnHelper<CurvatureParameterEntry>();
const startCol = createColumnHelper<MAPOptimizationRun>();

function formatMetric(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  return Math.abs(value) < 0.01 || Math.abs(value) >= 1_000
    ? value.toExponential(1)
    : formatNumber(value, 4);
}

function statusToSeverity(status: string): "fail" | "warn" | undefined {
  if (status === "fail") return "fail";
  if (status === "warn") return "warn";
  return undefined;
}

function statusToBadgeVariant(status: string): "secondary" | "warning" | "destructive" | "success" {
  if (status === "fail") return "destructive";
  if (status === "warn") return "warning";
  if (status === "pass") return "success";
  return "secondary";
}

function parameterStatusRank(status: string | undefined) {
  if (status === "fail") return 0;
  if (status === "warn") return 1;
  return 2;
}

function buildParameterLabelMap(result: MapGeometryResult) {
  const labelMap = new Map<string, string>();
  for (const entry of result.likelihood_curvature.per_parameter) {
    labelMap.set(entry.parameter, entry.interpretable_parameter);
  }
  for (const entry of result.posterior_curvature.per_parameter) {
    labelMap.set(entry.parameter, entry.interpretable_parameter);
  }
  return labelMap;
}

function getGeometryVerdict(
  result: MapGeometryResult,
  remainingWeakParameters: CurvatureParameterEntry[],
): GeometryVerdict {
  if (result.posterior_curvature.negative_direction_count > 0) {
    return {
      title: "Unstable Local Mode",
      description:
        "The posterior Hessian still has negative-curvature directions at the selected MAP, so the local Gaussian geometry is not trustworthy yet.",
      alertVariant: "destructive",
      badgeVariant: "destructive",
      badgeLabel: "mode instability",
    };
  }

  if (result.boundary_parameters.length > 0) {
    return {
      title: "Boundary-Adjacent Solution",
      description:
        "At least one selected MAP parameter sits near a support or prior bound, so local curvature can be distorted by boundary effects.",
      alertVariant: "warning",
      badgeVariant: "warning",
      badgeLabel: "boundary pathology",
    };
  }

  if (remainingWeakParameters.some((entry) => entry.normalized_status === "fail")) {
    return {
      title: "Weak After Priors",
      description:
        "Some parameters remain weak even after prior regularization, so the local posterior geometry is still underconstrained.",
      alertVariant: "destructive",
      badgeVariant: "destructive",
      badgeLabel: "still weak after priors",
    };
  }

  if (
    remainingWeakParameters.length > 0 ||
    result.prior_rescued_parameters.length > 0 ||
    result.n_successful_starts < result.n_starts
  ) {
    return {
      title: "Prior-Dependent Geometry",
      description:
        "The local mode is usable, but some stability comes from priors or from a subset of successful optimization starts rather than clean data-driven curvature alone.",
      alertVariant: "warning",
      badgeVariant: "warning",
      badgeLabel: "prior dependent",
    };
  }

  return {
    title: "Stable Local Mode",
    description:
      "No weak posterior directions, no boundary warnings, and no sign of local non-convexity were detected at the selected MAP.",
    alertVariant: "default",
    badgeVariant: "success",
    badgeLabel: "stable local mode",
  };
}

function SummaryCard({
  title,
  description,
  badges,
  children,
}: {
  title: string;
  description: string;
  badges?: ReactNode;
  children: ReactNode;
}) {
  return (
    <Card>
      <CardContent className="space-y-3 py-4">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <h4 className="text-sm font-semibold">{title}</h4>
            {badges}
          </div>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
        {children}
      </CardContent>
    </Card>
  );
}

function MetricRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium text-right">{value}</span>
    </div>
  );
}

function ParameterBadgeGroup({
  parameters,
  emptyLabel,
  defaultVariant = "outline",
  limit = 8,
}: {
  parameters: ParameterChip[];
  emptyLabel: string;
  defaultVariant?: "secondary" | "warning" | "destructive" | "success" | "outline";
  limit?: number;
}) {
  if (parameters.length === 0) {
    return <p className="text-sm text-muted-foreground">{emptyLabel}</p>;
  }

  const visible = parameters.slice(0, limit);
  const remaining = parameters.length - visible.length;

  return (
    <div className="flex flex-wrap gap-2">
      {visible.map((parameter) => (
        <Badge
          key={parameter.parameter}
          variant={parameter.status ? statusToBadgeVariant(parameter.status) : defaultVariant}
          title={
            parameter.label === parameter.parameter
              ? parameter.parameter
              : `${parameter.label} (${parameter.parameter})`
          }
        >
          {parameter.label}
        </Badge>
      ))}
      {remaining > 0 && <Badge variant="secondary">+{remaining} more</Badge>}
    </div>
  );
}

function LoadingsBarChart({ loadings }: { loadings: CurvatureDirectionLoading[] }) {
  if (loadings.length === 0) return <span className="text-xs text-muted-foreground">—</span>;

  const maxAbs = Math.max(...loadings.map((loading) => Math.abs(loading.loading)));
  const data = loadings.map((loading) => ({
    name: loading.interpretable_parameter,
    loading: loading.loading,
    opacity: maxAbs > 0 ? 0.15 + 0.7 * (Math.abs(loading.loading) / maxAbs) : 0.5,
  }));

  return (
    <div className="h-14 w-40">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 2, right: 4, left: 0, bottom: 0 }}>
          <XAxis dataKey="name" tick={false} axisLine={{ stroke: "var(--border)" }} height={2} />
          <YAxis
            tick={{ fontSize: 8 }}
            tickLine={false}
            axisLine={false}
            width={32}
            tickFormatter={(value: number) => formatNumber(value, 2)}
          />
          <RechartsTooltip
            formatter={(value: number | string | undefined) => {
              const numericValue = typeof value === "number" ? value : Number(value);
              return [
                Number.isFinite(numericValue) ? formatMetric(numericValue) : "—",
                "loading",
              ] as const;
            }}
            labelFormatter={(label: unknown) => String(label)}
            contentStyle={{ fontSize: 10, padding: "2px 6px" }}
          />
          <ReferenceLine y={0} stroke="var(--border)" />
          <Bar dataKey="loading" isAnimationActive={false}>
            {data.map((entry, index) => (
              <Cell key={index} fill="var(--primary)" fillOpacity={entry.opacity} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

const weakDirectionColumns = [
  directionCol.accessor("index", {
    header: "Hessian Direction",
    cell: (info) => <span className="font-medium">d{info.getValue()}</span>,
    meta: { mono: true },
  }),
  directionCol.accessor("normalized_eigenvalue", {
    header: () => (
      <HeaderWithTooltip
        label="Normalized Eigenvalue"
        tooltip="Eigenvalue after prior-standard-deviation normalization. Values <= 1 indicate weak local curvature relative to the prior scale; values between 1 and 10 are borderline."
      />
    ),
    cell: (info) => formatMetric(info.getValue()),
    meta: {
      mono: true,
      severity: (_value: number, row: CurvatureDirection) => statusToSeverity(row.status),
    },
  }),
  directionCol.accessor("eigenvalue", {
    header: () => (
      <HeaderWithTooltip
        label="Raw Eigenvalue"
        tooltip="Local Hessian eigenvalue before prior-scale normalization. Negative values indicate a locally non-convex direction."
      />
    ),
    cell: (info) => formatMetric(info.getValue()),
    meta: { mono: true },
  }),
  directionCol.accessor("top_loadings", {
    header: () => (
      <HeaderWithTooltip
        label="Dominant Parameter Loadings"
        tooltip="Largest signed components of the normalized Hessian eigenvector for this weak local direction. These are the parameters most involved in the poorly constrained curvature direction."
      />
    ),
    cell: (info) => <LoadingsBarChart loadings={info.getValue()} />,
  }),
];

const fullParameterColumns = [
  parameterCol.accessor("parameter", {
    header: "Parameter",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
    meta: { mono: true },
  }),
  parameterCol.accessor("interpretable_parameter", {
    header: () => (
      <HeaderWithTooltip
        label="Interpretable Parameter"
        tooltip="Semantic parameter name resolved from the Stage 4 model specification when available."
      />
    ),
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
    meta: { mono: true },
  }),
  parameterCol.accessor("diagonal_curvature", {
    header: () => (
      <HeaderWithTooltip
        label="Diagonal Curvature"
        tooltip="Diagonal entry of the local Hessian for this scalar parameter at the selected MAP."
      />
    ),
    cell: (info) => formatMetric(info.getValue()),
    meta: { mono: true },
  }),
  parameterCol.accessor("effective_eigenvalue", {
    header: () => (
      <HeaderWithTooltip
        label="Effective Eigenvalue"
        tooltip="Smallest raw Hessian eigenvalue among directions where this parameter has substantial loading."
      />
    ),
    cell: (info) => formatMetric(info.getValue()),
    meta: {
      mono: true,
      severity: (_value: number, row: CurvatureParameterEntry) => statusToSeverity(row.status),
    },
  }),
  parameterCol.accessor("normalized_effective_eigenvalue", {
    header: () => (
      <HeaderWithTooltip
        label="Normalized Eigenvalue"
        tooltip="Smallest prior-normalized Hessian eigenvalue among directions where this parameter has substantial loading. Thresholds: >10 pass, >1 warn, <=1 fail."
      />
    ),
    cell: (info) => formatMetric(info.getValue()),
    meta: {
      mono: true,
      severity: (_value: number, row: CurvatureParameterEntry) =>
        statusToSeverity(row.normalized_status),
    },
  }),
  parameterCol.accessor("normalized_status", {
    header: "Status",
    cell: (info) => (
      <Badge variant={statusToBadgeVariant(info.getValue())} className="capitalize">
        {info.getValue()}
      </Badge>
    ),
    meta: { align: "center" },
  }),
];

const mapStartColumns = [
  startCol.accessor("index", {
    header: "Start",
    cell: (info) => <span className="font-medium">#{info.getValue() + 1}</span>,
    meta: { mono: true },
  }),
  startCol.accessor("start_kind", {
    header: "Initializer",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
    meta: { mono: true },
  }),
  startCol.accessor("success", {
    header: "Converged",
    cell: (info) => (
      <Badge variant={info.getValue() ? "success" : "destructive"}>
        {info.getValue() ? "yes" : "no"}
      </Badge>
    ),
    meta: { align: "center" },
  }),
  startCol.accessor("objective", {
    header: () => (
      <HeaderWithTooltip
        label="Objective"
        tooltip="Final negative log-posterior minimized by the MAP optimizer. Lower is better."
      />
    ),
    cell: (info) => formatMetric(info.getValue()),
    meta: { mono: true },
  }),
  startCol.accessor("log_posterior", {
    header: () => (
      <HeaderWithTooltip
        label="Log Posterior"
        tooltip="Final log-posterior value at the optimized point."
      />
    ),
    cell: (info) => formatMetric(info.getValue()),
    meta: { mono: true },
  }),
  startCol.accessor("grad_norm", {
    header: () => (
      <HeaderWithTooltip
        label="Grad Norm"
        tooltip="Euclidean norm of the final optimizer gradient. Smaller values indicate a better-localized MAP solution."
      />
    ),
    cell: (info) => formatMetric(info.getValue()),
    meta: { mono: true },
  }),
  startCol.accessor("distance_to_best", {
    header: () => (
      <HeaderWithTooltip
        label="Distance To Best"
        tooltip="Euclidean distance in unconstrained parameter space between this optimized point and the selected best MAP."
      />
    ),
    cell: (info) => formatMetric(info.getValue()),
    meta: { mono: true },
  }),
];

function CurvatureSection({
  title,
  description,
  result,
  accordionValue,
  accordionLabel,
}: {
  title: string;
  description: string;
  result: MAPCurvatureResult;
  accordionValue: string;
  accordionLabel: string;
}) {
  const directionCount = result.normalized_eigenvalues.length;

  return (
    <section className="space-y-3">
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <h4 className="text-sm font-semibold">{title}</h4>
          <Badge variant={result.deficiency_count > 0 ? "destructive" : "success"}>
            weak directions: {result.deficiency_count}/{directionCount}
          </Badge>
          {result.negative_direction_count > 0 ? (
            <Badge variant="destructive">
              negative curvature: {result.negative_direction_count}
            </Badge>
          ) : (
            <Badge variant="success">positive definite</Badge>
          )}
          {result.normalized_condition_number != null ? (
            <Badge variant="secondary">
              cond: {formatMetric(result.normalized_condition_number)}
            </Badge>
          ) : (
            <Badge variant="warning">cond unavailable</Badge>
          )}
        </div>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>

      {result.weak_directions.length > 0 ? (
        <InfoTable
          columns={weakDirectionColumns as ColumnDef<CurvatureDirection, unknown>[]}
          data={result.weak_directions}
          filtering={false}
          maxHeight="max-h-[20rem]"
          estimateRowHeight={68}
        />
      ) : (
        <div className="rounded-md border border-dashed px-4 py-6 text-sm text-muted-foreground">
          No weak or borderline local Hessian directions were detected under the current thresholds.
        </div>
      )}

      <Accordion multiple>
        <AccordionItem value={accordionValue}>
          <AccordionTrigger className="text-sm">
            <span className="inline-flex flex-wrap items-center gap-1.5">
              {accordionLabel}
              <Badge variant="outline">{result.per_parameter.length} rows</Badge>
            </span>
          </AccordionTrigger>
          <AccordionContent>
            <InfoTable
              columns={fullParameterColumns as ColumnDef<CurvatureParameterEntry, unknown>[]}
              data={result.per_parameter}
              maxHeight="max-h-[28rem]"
            />
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </section>
  );
}

export function MapGeometryPanel({ result }: { result: MapGeometryResult }) {
  const bestStartLabel = `#${result.best_start_index + 1}`;
  const labelMap = buildParameterLabelMap(result);

  const remainingWeakParameters = [...result.posterior_curvature.per_parameter]
    .filter((entry) => entry.normalized_status !== "pass")
    .sort((left, right) => {
      const statusOrder =
        parameterStatusRank(left.normalized_status) - parameterStatusRank(right.normalized_status);
      if (statusOrder !== 0) return statusOrder;
      return left.normalized_effective_eigenvalue - right.normalized_effective_eigenvalue;
    });
  const remainingWeakChips = remainingWeakParameters.map((entry) => ({
    parameter: entry.parameter,
    label: entry.interpretable_parameter,
    status: entry.normalized_status,
  }));

  const priorRescuedChips = result.prior_rescued_parameters
    .map((parameter) => ({
      parameter,
      label: labelMap.get(parameter) ?? parameter,
    }))
    .sort((left, right) => left.label.localeCompare(right.label));

  const boundaryChips = result.boundary_parameters
    .map((parameter) => ({
      parameter,
      label: labelMap.get(parameter) ?? parameter,
    }))
    .sort((left, right) => left.label.localeCompare(right.label));

  const likelihoodWeakCount = result.likelihood_curvature.per_parameter.filter(
    (entry) => entry.normalized_status !== "pass",
  ).length;
  const posteriorWeakCount = remainingWeakParameters.length;
  const remainingWeakFailCount = remainingWeakParameters.filter(
    (entry) => entry.normalized_status === "fail",
  ).length;
  const remainingWeakWarnCount = remainingWeakParameters.filter(
    (entry) => entry.normalized_status === "warn",
  ).length;
  const verdict = getGeometryVerdict(result, remainingWeakParameters);
  const VerdictIcon = verdict.alertVariant === "default" ? CheckCircle2 : AlertTriangle;

  return (
    <div className="space-y-4">
      <Alert variant={verdict.alertVariant} className="border-2">
        <VerdictIcon className="mt-0.5 h-5 w-5" />
        <AlertTitle className="text-base font-semibold">
          <span className="inline-flex flex-wrap items-center gap-2">
            MAP Geometry
            <Badge variant={verdict.badgeVariant}>{verdict.badgeLabel}</Badge>
            <StatTooltip explanation="Stage 4b next checks dataset-conditioned local curvature around the selected MAP. The likelihood Hessian shows what the data identify locally; the posterior Hessian shows what remains after priors are included." />
          </span>
        </AlertTitle>
        <AlertDescription className="mt-2 space-y-3">
          <p>{verdict.description}</p>
          <div className="flex flex-wrap gap-2">
            <Badge
              variant={posteriorWeakCount > 0 ? "warning" : "success"}
            >{`posterior weak: ${posteriorWeakCount}`}</Badge>
            <Badge
              variant={likelihoodWeakCount > 0 ? "warning" : "success"}
            >{`likelihood weak: ${likelihoodWeakCount}`}</Badge>
            <Badge
              variant={result.n_successful_starts === result.n_starts ? "success" : "warning"}
            >{`converged starts: ${result.n_successful_starts}/${result.n_starts}`}</Badge>
            {result.prior_rescued_parameters.length > 0 && (
              <Badge variant="success">{`prior-rescued: ${result.prior_rescued_parameters.length}`}</Badge>
            )}
            {result.boundary_parameters.length > 0 && (
              <Badge variant="warning">{`boundary: ${result.boundary_parameters.length}`}</Badge>
            )}
          </div>
        </AlertDescription>
      </Alert>

      <div className="grid gap-4 xl:grid-cols-2">
        <SummaryCard
          title="Remaining Weak Parameters"
          description="These parameters are still weak in the posterior Hessian, so they remain the main local-identification problem after priors are applied."
          badges={
            <>
              <Badge variant={remainingWeakFailCount > 0 ? "destructive" : "success"}>
                fails: {remainingWeakFailCount}
              </Badge>
              <Badge variant={remainingWeakWarnCount > 0 ? "warning" : "secondary"}>
                warns: {remainingWeakWarnCount}
              </Badge>
            </>
          }
        >
          <ParameterBadgeGroup
            parameters={remainingWeakChips}
            emptyLabel="No parameters remain weak in the posterior geometry."
          />
        </SummaryCard>

        <SummaryCard
          title="Data vs Prior"
          description="This compares what the realized dataset identifies on its own with what only becomes stable after prior regularization."
          badges={
            <>
              <Badge variant={likelihoodWeakCount > 0 ? "warning" : "success"}>
                data-weak: {likelihoodWeakCount}
              </Badge>
              <Badge variant={result.prior_rescued_parameters.length > 0 ? "success" : "secondary"}>
                rescued: {result.prior_rescued_parameters.length}
              </Badge>
            </>
          }
        >
          <div className="space-y-3">
            <MetricRow
              label="Likelihood weak directions"
              value={result.likelihood_curvature.deficiency_count}
            />
            <MetricRow
              label="Posterior weak directions"
              value={result.posterior_curvature.deficiency_count}
            />
            <div className="space-y-2">
              <p className="text-sm font-medium">Prior-rescued parameters</p>
              <ParameterBadgeGroup
                parameters={priorRescuedChips}
                emptyLabel="No parameters changed from weak-in-data to strong-after-priors."
                defaultVariant="success"
              />
            </div>
          </div>
        </SummaryCard>

        <SummaryCard
          title="Mode Stability"
          description="These checks tell you whether the selected MAP looks like a clean local solution rather than a start-dependent optimizer artifact."
          badges={
            <>
              <Badge
                variant={
                  result.posterior_curvature.negative_direction_count > 0 ? "destructive" : "success"
                }
              >
                negative posterior dirs: {result.posterior_curvature.negative_direction_count}
              </Badge>
              <Badge variant="secondary">best start: {bestStartLabel}</Badge>
            </>
          }
        >
          <div className="space-y-2">
            <MetricRow
              label="Converged starts"
              value={`${result.n_successful_starts}/${result.n_starts}`}
            />
            <MetricRow label="Final grad norm" value={formatMetric(result.final_grad_norm)} />
            <MetricRow
              label="Runner-up gap"
              value={
                result.runner_up_objective_gap != null
                  ? formatMetric(result.runner_up_objective_gap)
                  : "—"
              }
            />
          </div>
        </SummaryCard>

        <SummaryCard
          title="Boundary Warnings"
          description="Boundary-adjacent parameters often make local curvature look stronger or weaker than it really is, so these should be reviewed before trusting the mode."
          badges={
            <Badge variant={boundaryChips.length > 0 ? "warning" : "success"}>
              flagged: {boundaryChips.length}
            </Badge>
          }
        >
          <ParameterBadgeGroup
            parameters={boundaryChips}
            emptyLabel="No parameters sit near a support or prior boundary."
            defaultVariant="warning"
          />
        </SummaryCard>
      </div>

      <Accordion multiple>
        <AccordionItem value="technical-details">
          <AccordionTrigger className="text-sm">
            <span className="inline-flex flex-wrap items-center gap-1.5">
              Technical Details
              <Badge variant="outline">advanced</Badge>
            </span>
          </AccordionTrigger>
          <AccordionContent className="space-y-6">
            <section className="space-y-2">
              <div>
                <h4 className="text-sm font-semibold">Selected MAP</h4>
                <p className="text-sm text-muted-foreground">
                  Objective terms at the highest-posterior mode selected from the multi-start
                  search.
                </p>
              </div>
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm">
                <span className="inline-flex items-center gap-1.5">
                  <span className="text-muted-foreground">log posterior</span>
                  <span className="font-medium">{formatMetric(result.map_log_posterior)}</span>
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="text-muted-foreground">log likelihood</span>
                  <span className="font-medium">{formatMetric(result.map_log_likelihood)}</span>
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="text-muted-foreground">log prior</span>
                  <span className="font-medium">{formatMetric(result.map_log_prior)}</span>
                </span>
              </div>
            </section>

            <div className="grid gap-6 xl:grid-cols-2">
              <CurvatureSection
                title="Likelihood Hessian"
                description="Local curvature of H_lik at the selected MAP. Weak directions here are weakly identified by the realized dataset itself."
                result={result.likelihood_curvature}
                accordionValue="likelihood-parameters"
                accordionLabel="Likelihood Parameter Curvature"
              />
              <CurvatureSection
                title="Posterior Hessian"
                description="Local curvature of H_post at the selected MAP. Weak directions here remain weak even after prior regularization is included."
                result={result.posterior_curvature}
                accordionValue="posterior-parameters"
                accordionLabel="Posterior Parameter Curvature"
              />
            </div>

            <Accordion multiple>
              <AccordionItem value="map-starts">
                <AccordionTrigger className="text-sm">
                  <span className="inline-flex flex-wrap items-center gap-1.5">
                    MAP Optimization Starts
                    <Badge variant="outline">{result.starts.length} rows</Badge>
                  </span>
                </AccordionTrigger>
                <AccordionContent>
                  <InfoTable
                    columns={mapStartColumns as ColumnDef<MAPOptimizationRun, unknown>[]}
                    data={result.starts}
                    maxHeight="max-h-[24rem]"
                  />
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}
