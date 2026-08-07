"use client";

import {
  Activity as ActivityIcon,
  BadgeCheck,
  ChartSpline,
  Check,
  CircleDot,
  Database,
  Eye,
  EyeOff,
  FlaskConical,
  GitBranch,
  Layers3,
  Link2,
  LockKeyhole,
  type LucideIcon,
  PanelRight,
  Rows3,
  ShieldCheck,
  Sparkles,
  Table2,
  Tags,
  Waves,
} from "lucide-react";
import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  type ArtifactMaterialization,
  DATA_LAYERS,
  isMaterialized,
  MATERIALIZATION_META,
  MATERIALIZATION_ORDER,
  MODEL_LAYERS,
  type ModelNodeId,
  PROTOTYPE_INDICATORS,
  PROTOTYPE_NODE_META,
  type WorkspaceLayer,
  type WorkspaceLayerId,
  type WorkspaceLens,
} from "./artifact-workspace-fixture";
import { DataCanvas } from "./data-canvas";
import { ModelCanvas } from "./model-canvas";

export interface ArtifactWorkspacePrototypeProps {
  initialLens?: WorkspaceLens;
  materialization?: ArtifactMaterialization;
}

const ALL_LAYERS = [...MODEL_LAYERS, ...DATA_LAYERS];

const LAYER_ICONS: Record<WorkspaceLayerId, LucideIcon> = {
  "model.structure": GitBranch,
  "model.measurement": Tags,
  "model.identification": ShieldCheck,
  "model.dynamics": Waves,
  "model.posterior": ChartSpline,
  "model.simulation": FlaskConical,
  "data.source": Database,
  "data.mapping": Link2,
  "data.observations": Rows3,
  "data.quality": BadgeCheck,
  "data.fit": ChartSpline,
};

const ACTIVITY_ITEMS: Array<{
  minimum: ArtifactMaterialization;
  label: string;
  detail: string;
  artifact: string;
  version: string;
}> = [
  {
    minimum: "structure",
    label: "Theory graph proposed",
    detail: "5 constructs and 7 directed relationships",
    artifact: "latent_structure",
    version: "v2",
  },
  {
    minimum: "measurement",
    label: "Indicators mapped",
    detail: "6 indicators across 4 observed constructs",
    artifact: "measurement_structure",
    version: "v3",
  },
  {
    minimum: "identified",
    label: "Identification checked",
    detail: "Screen-time estimand supported; chronotype marginalized",
    artifact: "identification_report",
    version: "v3",
  },
  {
    minimum: "identified",
    label: "Indicator panel validated",
    detail: "One reporting gap retained as an explicit finding",
    artifact: "validation_report",
    version: "v2",
  },
  {
    minimum: "fitted",
    label: "Posterior materialized",
    detail: "Exact particle inference and predictive checks complete",
    artifact: "posterior",
    version: "v1",
  },
  {
    minimum: "simulated",
    label: "Scenario saved",
    detail: "Reduce evening screen time by 30 minutes from day 14",
    artifact: "saved_scenarios",
    version: "v1",
  },
];

function LensButton({
  lens,
  active,
  icon: Icon,
  children,
  onClick,
}: {
  lens: WorkspaceLens;
  active: boolean;
  icon: LucideIcon;
  children: React.ReactNode;
  onClick: (lens: WorkspaceLens) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onClick(lens)}
      aria-pressed={active}
      className={cn(
        "inline-flex h-8 items-center gap-1.5 rounded-md px-3 text-xs font-medium transition-all",
        active
          ? "bg-white text-slate-900 shadow-sm ring-1 ring-slate-200"
          : "text-slate-500 hover:text-slate-800",
      )}
    >
      <Icon className="size-3.5" />
      {children}
    </button>
  );
}

function LayerRow({
  layer,
  available,
  visible,
  foundation,
  onToggle,
}: {
  layer: WorkspaceLayer;
  available: boolean;
  visible: boolean;
  foundation?: boolean;
  onToggle: (id: WorkspaceLayerId) => void;
}) {
  const Icon = LAYER_ICONS[layer.id];

  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <button
            type="button"
            disabled={!available || foundation}
            onClick={() => onToggle(layer.id)}
            aria-pressed={foundation || visible}
            className={cn(
              "group flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors",
              available && visible && "bg-white text-slate-900 shadow-sm ring-1 ring-slate-200",
              available && !visible && "text-slate-500 hover:bg-white/70",
              !available && "cursor-not-allowed text-slate-300",
              foundation && "cursor-default",
            )}
          />
        }
      >
        <span
          className={cn(
            "grid size-7 shrink-0 place-items-center rounded-md",
            available && visible ? "bg-slate-900 text-white" : "bg-slate-200/70 text-slate-500",
            !available && "text-slate-300",
          )}
        >
          <Icon className="size-3.5" />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[11px] font-medium">{layer.label}</span>
          <span className="block truncate font-mono text-[8px] text-muted-foreground/75">
            {layer.artifact}
          </span>
        </span>
        {foundation ? (
          <LockKeyhole className="size-3 text-slate-400" />
        ) : available ? (
          visible ? (
            <Eye className="size-3 text-slate-500" />
          ) : (
            <EyeOff className="size-3 text-slate-400" />
          )
        ) : (
          <CircleDot className="size-3 text-slate-300" />
        )}
      </TooltipTrigger>
      <TooltipContent side="right">
        {available ? layer.description : `Available when ${layer.artifact} materializes`}
      </TooltipContent>
    </Tooltip>
  );
}

function LayerSection({
  title,
  layers,
  materialization,
  visibleLayers,
  onToggle,
}: {
  title: string;
  layers: WorkspaceLayer[];
  materialization: ArtifactMaterialization;
  visibleLayers: ReadonlySet<WorkspaceLayerId>;
  onToggle: (id: WorkspaceLayerId) => void;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between px-2.5">
        <span className="text-[9px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
          {title}
        </span>
        <span className="text-[9px] tabular-nums text-muted-foreground">
          {layers.filter((layer) => isMaterialized(materialization, layer.minimum)).length}/
          {layers.length}
        </span>
      </div>
      <div className="space-y-1">
        {layers.map((layer) => (
          <LayerRow
            key={layer.id}
            layer={layer}
            available={isMaterialized(materialization, layer.minimum)}
            visible={visibleLayers.has(layer.id)}
            foundation={layer.id === "model.structure"}
            onToggle={onToggle}
          />
        ))}
      </div>
    </div>
  );
}

function LayersRail({
  lens,
  materialization,
  visibleLayers,
  onToggle,
}: {
  lens: WorkspaceLens;
  materialization: ArtifactMaterialization;
  visibleLayers: ReadonlySet<WorkspaceLayerId>;
  onToggle: (id: WorkspaceLayerId) => void;
}) {
  return (
    <aside className="border-r bg-slate-50/90 px-3 py-4">
      <div className="mb-4 flex items-center gap-2 px-2">
        <Layers3 className="size-4 text-slate-500" />
        <div>
          <div className="text-xs font-semibold text-slate-800">Visible layers</div>
          <div className="text-[9px] text-muted-foreground">The asset grows in place</div>
        </div>
      </div>
      <div className="space-y-5">
        {lens !== "data" ? (
          <LayerSection
            title="Model"
            layers={MODEL_LAYERS}
            materialization={materialization}
            visibleLayers={visibleLayers}
            onToggle={onToggle}
          />
        ) : null}
        {lens !== "model" ? (
          <LayerSection
            title="Data"
            layers={DATA_LAYERS}
            materialization={materialization}
            visibleLayers={visibleLayers}
            onToggle={onToggle}
          />
        ) : null}
      </div>

      <div className="mt-5 rounded-lg border border-dashed bg-white/70 p-2.5">
        <div className="flex items-center gap-1.5 text-[9px] font-semibold text-slate-700">
          <Sparkles className="size-3 text-blue-600" /> Linked selection
        </div>
        <p className="mt-1 text-[9px] leading-4 text-muted-foreground">
          Selecting a construct highlights its indicators in every lens.
        </p>
      </div>
    </aside>
  );
}

function InspectorPanel({
  lens,
  selectedNode,
  visibleLayers,
}: {
  lens: WorkspaceLens;
  selectedNode: ModelNodeId;
  visibleLayers: ReadonlySet<WorkspaceLayerId>;
}) {
  const node = PROTOTYPE_NODE_META[selectedNode];
  const nodeIndicators = PROTOTYPE_INDICATORS.filter(
    (indicator) => indicator.construct_name === selectedNode,
  );

  const modelVisible = lens !== "data";
  const dataVisible = lens !== "model";
  const measurementVisible =
    (modelVisible && visibleLayers.has("model.measurement")) ||
    (dataVisible && visibleLayers.has("data.mapping"));
  const identificationVisible = modelVisible && visibleLayers.has("model.identification");
  const fittedVisible =
    (modelVisible && visibleLayers.has("model.posterior")) ||
    (dataVisible && visibleLayers.has("data.fit"));
  const simulationVisible = modelVisible && visibleLayers.has("model.simulation");

  return (
    <div className="flex h-full flex-col">
      <div className="border-b px-4 py-3">
        <div className="text-[9px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
          Selected construct
        </div>
        <div className="mt-2 flex items-start justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold tracking-tight text-slate-900">{node.label}</h3>
            <p className="mt-0.5 text-[10px] text-muted-foreground">{node.eyebrow}</p>
          </div>
          {node.kind === "outcome" ? <Badge variant="success">Outcome</Badge> : null}
          {node.kind === "treatment" ? <Badge variant="outline">Treatment</Badge> : null}
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-auto p-4">
        <p className="text-[11px] leading-5 text-slate-600">{node.description}</p>

        {measurementVisible ? (
          <section className="rounded-lg border bg-slate-50/60 p-3">
            <div className="flex items-center gap-1.5 text-[10px] font-semibold text-slate-800">
              <Tags className="size-3" /> Measurement
            </div>
            {nodeIndicators.length > 0 ? (
              <div className="mt-2 space-y-1.5">
                {nodeIndicators.map((indicator) => (
                  <div key={indicator.name} className="flex items-center justify-between gap-2">
                    <span className="truncate font-mono text-[9px] text-slate-600">
                      {indicator.name}
                    </span>
                    <span className="text-[8px] text-muted-foreground">daily</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-2 text-[9px] leading-4 text-muted-foreground">
                No direct indicators. This construct remains explicit in the theory graph.
              </p>
            )}
          </section>
        ) : null}

        {identificationVisible ? (
          <section
            className={cn(
              "rounded-lg border p-3",
              node.kind === "confounder"
                ? "border-slate-200 bg-slate-50"
                : "border-emerald-200 bg-emerald-50/50",
            )}
          >
            <div className="flex items-center gap-1.5 text-[10px] font-semibold text-slate-800">
              <ShieldCheck className="size-3" /> Identification
            </div>
            <p className="mt-1.5 text-[9px] leading-4 text-slate-600">
              {node.kind === "confounder"
                ? "Integrated out for estimation; retained here to preserve the causal explanation."
                : node.kind === "treatment"
                  ? "The intervention effect on sleep quality is identified under the current design."
                  : "Retained in the identified causal design."}
            </p>
          </section>
        ) : null}

        {fittedVisible && node.observationModel ? (
          <section className="rounded-lg border bg-white p-3">
            <div className="flex items-center justify-between gap-2">
              <span className="inline-flex items-center gap-1.5 text-[10px] font-semibold text-slate-800">
                <Waves className="size-3" /> Fitted semantics
              </span>
              <Badge variant="secondary" className="font-mono text-[8px]">
                posterior v1
              </Badge>
            </div>
            <div className="mt-2 text-[9px] text-slate-600">{node.observationModel}</div>
            {node.posterior ? (
              <div className="mt-2 rounded-md bg-slate-50 px-2 py-1.5 font-mono text-[9px] text-slate-700">
                {node.posterior}
              </div>
            ) : null}
          </section>
        ) : null}

        {simulationVisible ? (
          <section className="rounded-lg border border-blue-200 bg-blue-50/60 p-3">
            <div className="flex items-center gap-1.5 text-[10px] font-semibold text-blue-900">
              <FlaskConical className="size-3" /> Runtime role
            </div>
            <p className="mt-1.5 text-[9px] leading-4 text-blue-900/70">
              {node.kind === "confounder"
                ? "Context only: no fitted state, trajectory, or intervention control."
                : node.kind === "treatment"
                  ? "Intervened: 30-minute reduction from day 14."
                  : "Trajectory shown for the selected saved scenario."}
            </p>
          </section>
        ) : null}
      </div>

      <div className="border-t bg-slate-50/70 px-4 py-3">
        <div className="flex items-center justify-between text-[8px] text-muted-foreground">
          <span>Snapshot pins</span>
          <span className="font-mono">latent v2 · measure v3 · panel v2</span>
        </div>
      </div>
    </div>
  );
}

function ActivityPanel({ materialization }: { materialization: ArtifactMaterialization }) {
  const visibleItems = ACTIVITY_ITEMS.filter((item) =>
    isMaterialized(materialization, item.minimum),
  ).reverse();
  const currentIndex = MATERIALIZATION_ORDER.indexOf(materialization);
  const next = MATERIALIZATION_ORDER[currentIndex + 1];

  return (
    <div className="flex h-full flex-col">
      <div className="border-b px-4 py-3">
        <div className="text-[9px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
          Activity & provenance
        </div>
        <h3 className="mt-1 text-sm font-semibold tracking-tight text-slate-900">
          How this asset changed
        </h3>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-4">
        {next ? (
          <div className="mb-4 rounded-lg border border-blue-200 bg-blue-50/60 p-3">
            <div className="flex items-center gap-2 text-[10px] font-semibold text-blue-900">
              <span className="size-2 animate-pulse rounded-full bg-blue-600" />
              Next capability
            </div>
            <p className="mt-1 text-[9px] leading-4 text-blue-900/70">
              {MATERIALIZATION_META[next].summary}
            </p>
          </div>
        ) : (
          <div className="mb-4 flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50/60 p-3 text-[10px] font-semibold text-emerald-800">
            <Check className="size-3.5" /> All current artifacts are materialized
          </div>
        )}

        <div className="relative space-y-4 before:absolute before:bottom-2 before:left-[6px] before:top-2 before:w-px before:bg-slate-200">
          {visibleItems.map((item, index) => (
            <div key={`${item.artifact}-${item.version}`} className="relative flex gap-3">
              <span
                className={cn(
                  "relative z-10 mt-1 size-[13px] shrink-0 rounded-full border-2 border-white",
                  index === 0 ? "bg-emerald-500" : "bg-slate-300",
                )}
              />
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-[10px] font-semibold text-slate-800">
                    {item.label}
                  </span>
                  <span className="font-mono text-[8px] text-muted-foreground">{item.version}</span>
                </div>
                <p className="mt-0.5 text-[9px] leading-4 text-muted-foreground">{item.detail}</p>
                <span className="mt-1 block font-mono text-[8px] text-slate-400">
                  {item.artifact}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function AssetPanel({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <section className="flex min-h-[560px] min-w-0 flex-col overflow-hidden rounded-xl border bg-white shadow-sm">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold tracking-tight text-slate-900">{title}</h2>
          <p className="mt-0.5 text-[10px] text-muted-foreground">{subtitle}</p>
        </div>
        <Button variant="ghost" size="icon-sm" aria-label={`Open ${title} options`}>
          <PanelRight />
        </Button>
      </div>
      <div className="min-h-0 flex-1">{children}</div>
    </section>
  );
}

export function ArtifactWorkspacePrototype({
  initialLens = "model",
  materialization = "simulated",
}: ArtifactWorkspacePrototypeProps) {
  const [lens, setLens] = useState<WorkspaceLens>(initialLens);
  const [selectedNode, setSelectedNode] = useState<ModelNodeId>("screen_time");
  const [visibilityOverrides, setVisibilityOverrides] = useState<
    Partial<Record<WorkspaceLayerId, boolean>>
  >({});
  const [showActivity, setShowActivity] = useState(false);

  const visibleLayers = useMemo(() => {
    const visible = new Set<WorkspaceLayerId>();
    for (const layer of ALL_LAYERS) {
      if (!isMaterialized(materialization, layer.minimum)) continue;
      if (layer.id === "model.structure" || visibilityOverrides[layer.id] !== false) {
        visible.add(layer.id);
      }
    }
    return visible;
  }, [materialization, visibilityOverrides]);

  const toggleLayer = (id: WorkspaceLayerId) => {
    if (id === "model.structure") return;
    setVisibilityOverrides((current) => ({
      ...current,
      [id]: !(current[id] ?? true),
    }));
  };

  const meta = MATERIALIZATION_META[materialization];

  return (
    <div className="min-h-screen bg-[#eef1f4] text-slate-950">
      <header className="border-b bg-white">
        <div className="flex min-h-11 items-center justify-between gap-4 border-b px-4 sm:px-6">
          <div className="flex min-w-0 items-center gap-2.5">
            <span className="grid size-6 place-items-center rounded-md bg-slate-950 text-[10px] font-bold text-white">
              N1
            </span>
            <span className="truncate text-xs font-semibold tracking-tight">N-of-1 Causal Lab</span>
            <Badge variant="secondary" className="hidden text-[9px] sm:inline-flex">
              Design prototype
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <div className="hidden items-center gap-2 text-[10px] text-muted-foreground md:flex">
              <span className="size-1.5 rounded-full bg-emerald-500" />
              <span>{meta.label}</span>
              <span className="font-mono">{meta.artifacts}/14 artifacts</span>
            </div>
            <Button
              variant={showActivity ? "secondary" : "outline"}
              size="sm"
              onClick={() => setShowActivity((current) => !current)}
              aria-pressed={showActivity}
            >
              <ActivityIcon /> Activity
            </Button>
          </div>
        </div>

        <div className="flex flex-wrap items-end justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-[9px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
              <span className="font-mono">SLEEP-42</span>
              <span>·</span>
              <span>Current analysis</span>
            </div>
            <h1 className="mt-1 truncate text-base font-semibold tracking-tight sm:text-lg">
              How does evening screen time affect sleep quality over the next month?
            </h1>
          </div>

          <div className="inline-flex rounded-lg bg-slate-100 p-1">
            <LensButton lens="data" active={lens === "data"} icon={Table2} onClick={setLens}>
              Data
            </LensButton>
            <LensButton lens="model" active={lens === "model"} icon={GitBranch} onClick={setLens}>
              Model
            </LensButton>
            <LensButton lens="split" active={lens === "split"} icon={PanelRight} onClick={setLens}>
              Linked
            </LensButton>
          </div>
        </div>
      </header>

      <div className="grid min-h-[760px] xl:grid-cols-[220px_minmax(0,1fr)_300px]">
        <LayersRail
          lens={lens}
          materialization={materialization}
          visibleLayers={visibleLayers}
          onToggle={toggleLayer}
        />

        <main className="min-w-0 space-y-4 p-4">
          {lens === "model" ? (
            <AssetPanel
              title="Model asset"
              subtitle="One persistent DAG; materialized semantics arrive as layers"
            >
              <ModelCanvas
                visibleLayers={visibleLayers}
                selectedNode={selectedNode}
                onSelectNode={setSelectedNode}
              />
            </AssetPanel>
          ) : null}

          {lens === "data" ? (
            <AssetPanel
              title="Data asset"
              subtitle="Source records become a validated, model-aligned indicator panel"
            >
              <DataCanvas
                visibleLayers={visibleLayers}
                selectedNode={selectedNode}
                onSelectNode={setSelectedNode}
              />
            </AssetPanel>
          ) : null}

          {lens === "split" ? (
            <>
              <div className="flex items-center gap-2 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-[10px] text-blue-900">
                <Link2 className="size-3.5" />
                The selected construct is shared across both assets; choose a node or indicator
                track to follow the link.
              </div>
              <AssetPanel
                title="Model asset"
                subtitle="Causal entities and currently supported runtime semantics"
              >
                <ModelCanvas
                  visibleLayers={visibleLayers}
                  selectedNode={selectedNode}
                  onSelectNode={setSelectedNode}
                />
              </AssetPanel>
              <AssetPanel
                title="Data asset"
                subtitle="Observed evidence linked back to the same construct identities"
              >
                <DataCanvas
                  visibleLayers={visibleLayers}
                  selectedNode={selectedNode}
                  onSelectNode={setSelectedNode}
                />
              </AssetPanel>
            </>
          ) : null}
        </main>

        <aside className="border-l bg-white">
          {showActivity ? (
            <ActivityPanel materialization={materialization} />
          ) : (
            <InspectorPanel lens={lens} selectedNode={selectedNode} visibleLayers={visibleLayers} />
          )}
        </aside>
      </div>
    </div>
  );
}
