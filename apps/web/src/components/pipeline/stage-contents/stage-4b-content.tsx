import { RBPartitionCard } from "@/components/stages/parametric-id/rb-partition-card";
import { TRuleCard } from "@/components/stages/parametric-id/t-rule-card";
import { WeakParamsList } from "@/components/stages/parametric-id/weak-params-list";
import { HardGateAlert } from "@/components/ui/custom/hard-gate-alert";
import type { Stage4bData } from "@causal-ssm/api-types";

export default function Stage4bContent({ data }: { data: Stage4bData }) {
  const pid = data.parametric_id;

  return (
    <div className="space-y-4">
      {pid.t_rule && !pid.t_rule.satisfies && (
        <HardGateAlert
          title="T-Rule Violated — Pipeline Halted"
          explanation={`The model has ${pid.t_rule.n_free_params} free parameters but only ${pid.t_rule.n_moments} moment conditions. The model has more unknowns than equations.`}
          suggestion="Reduce model complexity by removing parameters or collect more time points to increase moment conditions."
        />
      )}
      {pid.t_rule && <TRuleCard tRule={pid.t_rule} />}
      {data.rb_partition && <RBPartitionCard partition={data.rb_partition} />}
      {pid.per_param_classification && pid.per_param_classification.length > 0 && (
        <WeakParamsList params={pid.per_param_classification} threshold={pid.threshold} />
      )}
    </div>
  );
}
