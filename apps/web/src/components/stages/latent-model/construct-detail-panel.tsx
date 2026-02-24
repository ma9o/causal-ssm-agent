import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Construct } from "@causal-ssm/api-types";
import { Star } from "lucide-react";

interface ConstructDetailPanelProps {
  construct: Construct;
}

export function ConstructDetailPanel({ construct }: ConstructDetailPanelProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardTitle className="text-base">{construct.name}</CardTitle>
          {construct.is_outcome && <Star className="h-4 w-4 fill-foreground/75 text-foreground/75" />}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-muted-foreground">{construct.description}</p>

        <div className="flex flex-wrap gap-2">
          <Badge variant={construct.role === "endogenous" ? "default" : "secondary"}>
            {construct.role}
          </Badge>
          <Badge variant="outline">{construct.temporal_status.replace("_", " ")}</Badge>
          {construct.temporal_scale && (
            <Badge variant="outline">{construct.temporal_scale}</Badge>
          )}
          {construct.is_outcome && <Badge variant="warning">outcome</Badge>}
        </div>
      </CardContent>
    </Card>
  );
}
