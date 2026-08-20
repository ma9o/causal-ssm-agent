import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = fileURLToPath(new URL("../../../../../", import.meta.url));

// The current DEMO episode stops at validation. Raw data and extracted measurements
// remain byte-identical; the deterministic presentation projection is generated in
// the same canonical fixture root from that durable source.
const copiedStoreArtifacts = {
  raw_data: ["store/raw_data/v1/profile.json", "fixture/artifacts/raw_data.json"],
  measurements: ["store/measurements/v1/measurements.json", "fixture/artifacts/measurements.json"],
} as const;

describe("promoted DEMO fixture", () => {
  it("keeps materialized artifact projections byte-identical to the store", () => {
    const drifted = Object.entries(copiedStoreArtifacts)
      .filter(([, [storePath, fixturePath]]) => {
        const store = readFileSync(join(repoRoot, "data/DEMO", storePath), "utf8");
        const fixture = readFileSync(join(repoRoot, "data/DEMO", fixturePath), "utf8");
        return store !== fixture;
      })
      .map(([artifactId]) => artifactId);

    expect(drifted, "Run `bun run fixture:promote --from <workspace-id>`.").toEqual([]);
  });

  it("has no second component-local demo fixture root", () => {
    expect(existsSync(join(repoRoot, "apps/web/src/components/__fixtures__/demo-run"))).toBe(false);
  });

  it("keeps the compact scientific DAG and backend structural dispositions aligned", () => {
    const read = (path: string) =>
      JSON.parse(readFileSync(join(repoRoot, "data/DEMO", path), "utf8")) as Record<
        string,
        unknown
      >;
    const fixtureLatent = read("fixture/artifacts/latent_structure.json") as {
      latent_structure: {
        constructs: Array<{ name: string; role: string }>;
        edges: Array<{ cause: string; effect: string }>;
      };
    };
    const storedLatent = read("store/latent_structure/v1/latent-structure.json") as {
      latent_structure: {
        edges: Array<{ cause: string; effect: string }>;
      };
    };
    const fixtureMeasurement = read("fixture/artifacts/measurement_structure.json") as {
      measurement_structure: unknown;
      known_inputs: Array<{ construct: string }>;
      scientific_only_constructs: Array<{ construct: string }>;
    };
    const fixtureCausal = read("fixture/artifacts/causal_design.json").causal_design as {
      latent: unknown;
      measurement: unknown;
      known_inputs: Array<{ construct: string }>;
      scientific_only_constructs: Array<{ construct: string }>;
      estimation?: unknown;
    };
    const fixturePlan = read("fixture/artifacts/structural_plan.json").structural_plan as {
      semantics: { constructs: Record<string, { name: string }> };
      dispositions: Array<{
        source_id: string;
        source_kind: string;
        disposition: string;
      }>;
    };

    expect(fixtureCausal.latent).toEqual(fixtureLatent.latent_structure);
    expect(fixtureCausal.measurement).toEqual(fixtureMeasurement.measurement_structure);
    expect(fixtureCausal.known_inputs).toEqual(fixtureMeasurement.known_inputs);
    expect(fixtureCausal.scientific_only_constructs).toEqual(
      fixtureMeasurement.scientific_only_constructs,
    );
    expect(fixtureCausal.known_inputs).toHaveLength(2);
    expect(fixtureCausal.scientific_only_constructs).toHaveLength(6);
    expect(fixtureLatent.latent_structure.constructs).toHaveLength(17);
    expect(fixtureLatent.latent_structure.edges).toHaveLength(32);
    expect(
      (fixtureMeasurement.measurement_structure as { indicators: unknown[] }).indicators,
    ).toHaveLength(19);

    const constructsWithIncomingEdges = new Set(
      fixtureLatent.latent_structure.edges.map(({ effect }) => effect),
    );
    expect(
      fixtureLatent.latent_structure.constructs
        .filter(({ name, role }) => role === "endogenous" && !constructsWithIncomingEdges.has(name))
        .map(({ name }) => name),
      "The reduced story must not silently turn endogenous constructs into unexplained roots.",
    ).toEqual([]);

    const selectedConstructNames = new Set(
      fixtureLatent.latent_structure.constructs.map(({ name }) => name),
    );
    const storedInducedEdgeKeys = storedLatent.latent_structure.edges
      .filter(
        ({ cause, effect }) =>
          selectedConstructNames.has(cause) && selectedConstructNames.has(effect),
      )
      .map(({ cause, effect }) => `${cause}→${effect}`);
    const contractedEdgeKeys = [
      "natural_recovery_propensity→internalizing_symptom_burden",
      "taper_speed_dose_reduction→withdrawal_symptom_burden",
      "escitalopram_dose_taken→internalizing_symptom_burden",
    ];
    expect(
      fixtureLatent.latent_structure.edges.map(({ cause, effect }) => `${cause}→${effect}`).sort(),
    ).toEqual([...storedInducedEdgeKeys, ...contractedEdgeKeys].sort());

    const dispositions = fixturePlan.dispositions
      .filter(({ source_kind }) => source_kind === "construct")
      .map((item) => ({
        name: fixturePlan.semantics.constructs[item.source_id]?.name,
        disposition: item.disposition,
      }));
    expect(dispositions).toEqual(
      expect.arrayContaining([
        { name: "withdrawal_symptom_burden", disposition: "identification_only" },
        { name: "neuroadaptation_dependence_state", disposition: "identification_only" },
        { name: "natural_recovery_propensity", disposition: "identification_only" },
        {
          name: "past_escitalopram_response_tolerability",
          disposition: "identification_only",
        },
        { name: "clinical_monitoring_rescue_care", disposition: "identification_only" },
        { name: "stable_withdrawal_susceptibility", disposition: "marginalized" },
        { name: "external_stressful_events", disposition: "known_input" },
        { name: "internalizing_symptom_burden", disposition: "retained_state" },
      ]),
    );
    expect(fixtureCausal).not.toHaveProperty("estimation");
  });

  it("keeps every artificial downstream projection on the current DEMO ontology", () => {
    const readFixture = (name: string) =>
      JSON.parse(readFileSync(join(repoRoot, "data/DEMO/fixture", name), "utf8")) as Record<
        string,
        unknown
      >;

    const causal = readFixture("artifacts/causal_design.json").causal_design as {
      latent: {
        constructs: Array<{ name: string; temporal_status: string }>;
        edges: Array<{ cause: string; effect: string }>;
      };
      measurement: { indicators: Array<{ name: string; construct_name: string }> };
      identifiability: { identifiable_treatments: Record<string, unknown> };
      known_inputs: Array<{ construct: string }>;
      scientific_only_constructs: Array<{ construct: string }>;
    };
    const plan = readFixture("artifacts/structural_plan.json").structural_plan as {
      semantics: {
        constructs: Record<string, { name: string }>;
        edges: Record<string, { cause: string; effect: string }>;
        indicators: Record<string, { name: string }>;
      };
      state_order: string[];
      edges: Array<{ source_id: string }>;
      manifest_indicator_order: string[];
    };
    const model = readFixture("artifacts/statistical_model_spec.json") as {
      statistical_model_spec: {
        likelihoods: Array<{ variable: string }>;
        parameters: Array<{ name: string; role: string }>;
      };
      authored_priors: Record<string, unknown>;
      resolved_priors: Array<{ parameter: string }>;
      prior_predictive_samples: Record<string, number[]>;
    };
    const posterior = readFixture("artifacts/posterior.json") as {
      ppc: { overlays: Array<{ variable: string }> };
      mcmc_diagnostics: { per_parameter: Array<{ parameter: string }> };
      posterior_marginals: Array<{ parameter: string }>;
    };
    const report = readFixture("artifacts/baseline_report.json") as {
      intervention_results: Array<{ treatment: string }>;
    };

    const stateNames = plan.state_order.map((sourceId) => plan.semantics.constructs[sourceId].name);
    const stateNameSet = new Set(stateNames);
    const indicatorNames = causal.measurement.indicators.map(({ name }) => name);
    const manifestIndicatorNames = plan.manifest_indicator_order.map(
      (sourceId) => plan.semantics.indicators[sourceId].name,
    );
    const executableEdges = plan.edges.map(({ source_id }) => plan.semantics.edges[source_id]);
    const edgeParameterNames = executableEdges.map(
      ({ cause, effect }) => `beta_${cause}_${effect}`,
    );
    const parameterNames = model.statistical_model_spec.parameters.map(({ name }) => name);

    expect(Object.keys(model).sort()).toEqual([
      "authored_priors",
      "prior_predictive_diagnostics",
      "prior_predictive_samples",
      "resolved_priors",
      "search_queries",
      "statistical_model_spec",
      "validation_warnings",
    ]);
    expect(Object.keys(posterior).sort()).toEqual([
      "inference_metadata",
      "loo_diagnostics",
      "mcmc_diagnostics",
      "posterior_marginals",
      "posterior_pairs",
      "ppc",
      "smc_diagnostics",
    ]);
    expect(Object.keys(report).sort()).toEqual([
      "final_summary",
      "intervention_results",
      "saved_scenarios",
    ]);
    expect(model.statistical_model_spec.likelihoods.map(({ variable }) => variable)).toEqual(
      manifestIndicatorNames,
    );
    expect(Object.keys(model.prior_predictive_samples).sort()).toEqual(
      [...manifestIndicatorNames].sort(),
    );
    expect(Object.keys(model.authored_priors).sort()).toEqual([...parameterNames].sort());
    expect(model.resolved_priors.map(({ parameter }) => parameter)).toEqual(parameterNames);
    expect(posterior.ppc.overlays.map(({ variable }) => variable)).toEqual(manifestIndicatorNames);
    expect(posterior.mcmc_diagnostics.per_parameter.map(({ parameter }) => parameter)).toEqual(
      parameterNames,
    );
    expect(posterior.posterior_marginals.map(({ parameter }) => parameter)).toEqual(parameterNames);
    expect(
      model.statistical_model_spec.parameters
        .filter(({ role }) => role === "fixed_effect")
        .map(({ name }) => name),
    ).toEqual(edgeParameterNames);
    expect(report.intervention_results.map(({ treatment }) => treatment).sort()).toEqual(
      Object.keys(causal.identifiability.identifiable_treatments)
        .filter((treatment) => stateNameSet.has(treatment))
        .sort(),
    );
    expect(stateNames).toHaveLength(7);
    expect(manifestIndicatorNames).toHaveLength(10);
    expect(executableEdges).toHaveLength(13);
    expect(parameterNames).toHaveLength(48);
    expect(indicatorNames).toHaveLength(19);
    expect(stateNames).toContain("internalizing_symptom_burden");
  });

  it("materializes comprehensive DAG layers only where their process semantics exist", () => {
    const plan = JSON.parse(
      readFileSync(join(repoRoot, "data/DEMO/fixture/artifacts/structural_plan.json"), "utf8"),
    ).structural_plan as {
      semantics: {
        constructs: Record<string, { name: string }>;
      };
      state_order: string[];
    };
    const trace = JSON.parse(
      readFileSync(join(repoRoot, "data/DEMO/fixture/traces/baseline_report.json"), "utf8"),
    ) as {
      messages: Array<{ tool_name: string | null; tool_result: string | null }>;
    };
    const simulations = trace.messages
      .filter(
        (message): message is { tool_name: string; tool_result: string } =>
          message.tool_name === "simulate" && message.tool_result != null,
      )
      .map((message) => JSON.parse(message.tool_result)) as Array<{
      outcome: string;
      visualization: {
        reference_node_trajectories: Record<string, number[]>;
        action_node_trajectories: Record<string, number[]>;
        node_effect_trajectories: Record<string, number[]>;
        start_state: Record<string, number>;
      };
      clamps: unknown[];
      effect_trajectory: Array<{ day: number; effect: number }>;
    }>;

    const stateNames = plan.state_order
      .map((sourceId) => plan.semantics.constructs[sourceId].name)
      .sort();

    expect(simulations).toHaveLength(5);
    for (const simulation of simulations) {
      expect(simulation.outcome).toBe("internalizing_symptom_burden");
      expect(simulation.clamps).toHaveLength(1);
      expect(Object.keys(simulation.visualization).sort()).toEqual([
        "action_node_trajectories",
        "node_effect_trajectories",
        "reference_node_trajectories",
        "start_state",
      ]);
      expect(Object.keys(simulation.visualization.reference_node_trajectories).sort()).toEqual(
        stateNames,
      );
      expect(Object.keys(simulation.visualization.action_node_trajectories).sort()).toEqual(
        stateNames,
      );
      expect(Object.keys(simulation.visualization.node_effect_trajectories).sort()).toEqual(
        stateNames,
      );
      expect(Object.keys(simulation.visualization.start_state).sort()).toEqual(stateNames);
      expect(simulation.effect_trajectory).toHaveLength(61);
      expect(
        Object.values(simulation.visualization.reference_node_trajectories).every(
          (trajectory) => trajectory.length === simulation.effect_trajectory.length,
        ),
      ).toBe(true);
    }
  });
});
