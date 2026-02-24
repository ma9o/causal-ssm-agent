export const CHAIN_COLORS = [
  "var(--primary)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
];

export const diagnosisLabel: Record<string, string> = {
  well_identified: "Well Identified",
  prior_dominated: "Prior Dominated",
  prior_data_conflict: "Prior-Data Conflict",
};

export const diagnosisColor: Record<string, string> = {
  well_identified: "var(--success)",
  prior_dominated: "var(--warning)",
  prior_data_conflict: "var(--destructive)",
};

export const diagnosisBadgeVariant: Record<string, "success" | "warning" | "destructive"> = {
  well_identified: "success",
  prior_dominated: "warning",
  prior_data_conflict: "destructive",
};
