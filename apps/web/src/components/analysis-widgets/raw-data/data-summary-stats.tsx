import { formatDate } from "@/lib/utils/format";
import { Calendar, Columns3, Database } from "lucide-react";

interface DataSummaryStatsProps {
  nRecords: number;
  nColumns?: number;
  dateRange: { start: string; end: string };
}

export function DataSummaryStats({ nRecords, nColumns, dateRange }: DataSummaryStatsProps) {
  return (
    <div className="flex items-center gap-4 text-sm text-muted-foreground px-1">
      <span className="font-medium text-foreground">Data Summary</span>
      <div className="flex items-center gap-1.5">
        <Database className="h-3.5 w-3.5" />
        <span className="font-medium text-foreground">{nRecords.toLocaleString()}</span>
        <span>rows</span>
      </div>
      {nColumns !== undefined && (
        <div className="flex items-center gap-1.5">
          <Columns3 className="h-3.5 w-3.5" />
          <span className="font-medium text-foreground">{nColumns}</span>
          <span>columns</span>
        </div>
      )}
      {dateRange.start && (
        <div className="flex items-center gap-1.5">
          <Calendar className="h-3.5 w-3.5" />
          <span>
            {formatDate(dateRange.start)} &ndash; {formatDate(dateRange.end)}
          </span>
        </div>
      )}
    </div>
  );
}
