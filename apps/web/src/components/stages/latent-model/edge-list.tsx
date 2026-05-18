import { Badge } from "@/components/ui/badge";
import { InfoTable } from "@/components/ui/info-table";
import type { CausalEdge } from "@nof1-causal-lab/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";

const col = createColumnHelper<CausalEdge>();

const columns = [
  col.accessor("cause", {
    header: "Cause",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
  }),
  col.accessor("effect", {
    header: "Effect",
    cell: (info) => <span className="font-medium">{info.getValue()}</span>,
  }),
  col.accessor("lagged", {
    header: "Timing",
    cell: (info) => (
      <Badge variant={info.getValue() ? "default" : "secondary"}>
        {info.getValue() ? "Lagged" : "Contemporaneous"}
      </Badge>
    ),
  }),
  col.accessor("description", {
    header: "Description",
    cell: (info) => <span className="max-w-xs text-muted-foreground">{info.getValue()}</span>,
  }),
];

export function EdgeList({ edges }: { edges: CausalEdge[] }) {
  return <InfoTable columns={columns as ColumnDef<CausalEdge, unknown>[]} data={edges} />;
}
