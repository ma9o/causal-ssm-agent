import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { CausalEdge } from "@nof1-causal-lab/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import { ExternalLink } from "lucide-react";

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
    cell: (info) => (
      <span className="max-w-xs whitespace-normal text-muted-foreground">{info.getValue()}</span>
    ),
  }),
  col.display({
    id: "sources",
    header: () => (
      <HeaderWithTooltip
        label="Sources"
        tooltip="Literature sources supporting this causal link. Click to open."
      />
    ),
    cell: ({ row }) => {
      const sources = row.original.sources;
      if (sources.length === 0) {
        return <span className="text-xs text-muted-foreground">--</span>;
      }
      return (
        <div className="flex items-center gap-1.5">
          {sources.map((source, i) => (
            <Tooltip
              key={`source-${
                // biome-ignore lint/suspicious/noArrayIndexKey: stable ordered list
                i
              }`}
            >
              <TooltipTrigger>
                {source.url ? (
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-0.5 text-primary hover:underline"
                  >
                    <Badge variant="secondary" className="cursor-pointer text-[10px] px-1.5">
                      {i + 1}
                      <ExternalLink className="ml-0.5 h-2.5 w-2.5" />
                    </Badge>
                  </a>
                ) : (
                  <Badge variant="secondary" className="text-[10px] px-1.5">
                    {i + 1}
                  </Badge>
                )}
              </TooltipTrigger>
              <TooltipContent>
                <div className="max-w-xs text-xs">
                  <p className="font-medium">{source.title}</p>
                  <p className="text-muted-foreground">{source.snippet}</p>
                </div>
              </TooltipContent>
            </Tooltip>
          ))}
        </div>
      );
    },
    meta: { align: "center" },
  }),
];

export function EdgeList({ edges }: { edges: CausalEdge[] }) {
  return <InfoTable columns={columns as ColumnDef<CausalEdge, unknown>[]} data={edges} />;
}
