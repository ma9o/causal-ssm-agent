import { TRANSITION_META } from "@nof1-causal-lab/api-types";
import type { ArtifactViewId } from "@nof1-causal-lab/api-types";
import { Loader2 } from "lucide-react";
import { motion } from "motion/react";

export function ActiveTransitionsIndicator({ artifactIds }: { artifactIds: ArtifactViewId[] }) {
  if (artifactIds.length === 0) return null;
  const labels = artifactIds.map((artifactId) => TRANSITION_META[artifactId].label);

  return (
    <motion.div
      className="flex items-center gap-2 rounded-lg border border-dashed border-muted-foreground/30 px-4 py-3 text-sm text-muted-foreground"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      <Loader2 className="h-4 w-4 animate-spin" />
      <span>Running {labels.join(", ")}...</span>
    </motion.div>
  );
}
