import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ExternalLink } from "lucide-react";

interface LiteratureSource {
  title: string;
  snippet: string;
  url?: string | null;
  effect_size?: string | null;
}

export function SourceBadges({ sources }: { sources?: readonly LiteratureSource[] }) {
  if (!sources || sources.length === 0) {
    return <span className="text-xs text-muted-foreground">--</span>;
  }

  return (
    <div className="flex items-center gap-1.5">
      {sources.map((source, index) => (
        <Tooltip
          key={`source-${
            // biome-ignore lint/suspicious/noArrayIndexKey: duplicate citations are valid
            index
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
                  {index + 1}
                  <ExternalLink className="ml-0.5 h-2.5 w-2.5" />
                </Badge>
              </a>
            ) : (
              <Badge variant="secondary" className="text-[10px] px-1.5">
                {index + 1}
              </Badge>
            )}
          </TooltipTrigger>
          <TooltipContent>
            <div className="max-w-xs text-xs">
              <p className="font-medium">{source.title}</p>
              <p className="text-muted-foreground">{source.snippet}</p>
              {source.effect_size ? (
                <span className="text-muted-foreground">Effect: {source.effect_size}</span>
              ) : null}
            </div>
          </TooltipContent>
        </Tooltip>
      ))}
    </div>
  );
}
