import { STAGES } from "@causal-ssm/api-types";
import type { StageId } from "@causal-ssm/api-types";
import { Loader2 } from "lucide-react";
import { motion } from "motion/react";

export function ActiveStageIndicator({ stageId }: { stageId: StageId | null }) {
  if (!stageId) return null;
  const stage = STAGES.find((s) => s.id === stageId);
  if (!stage) return null;

  return (
    <motion.div
      className="flex items-center gap-2 rounded-lg border border-dashed border-muted-foreground/30 px-4 py-3 text-sm text-muted-foreground"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      <Loader2 className="h-4 w-4 animate-spin" />
      <span>
        Running Stage {stage.number}: {stage.label}...
      </span>
    </motion.div>
  );
}
