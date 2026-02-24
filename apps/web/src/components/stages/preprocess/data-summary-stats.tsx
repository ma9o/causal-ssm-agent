import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatDate } from "@/lib/utils/format";
import { Calendar, Database } from "lucide-react";

interface DataSummaryStatsProps {
  sourceLabel: string;
  nRecords: number;
  dateRange: { start: string; end: string };
}

export function DataSummaryStats({ sourceLabel, nRecords, dateRange }: DataSummaryStatsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Data Summary</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-4">
          <Badge variant="outline">{sourceLabel}</Badge>
          <div className="flex items-center gap-2 text-sm">
            <Database className="h-4 w-4 text-muted-foreground" />
            <span className="font-medium">{nRecords.toLocaleString()}</span>
            <span className="text-muted-foreground">records</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Calendar className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">
              {formatDate(dateRange.start)} &ndash; {formatDate(dateRange.end)}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
