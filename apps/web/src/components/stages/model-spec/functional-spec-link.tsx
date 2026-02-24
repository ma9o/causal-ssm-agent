import { FUNCTIONAL_SPEC_URL } from "@/lib/constants/stages";
import { BookOpen } from "lucide-react";

export function FunctionalSpecLink() {
  return (
    <a
      href={FUNCTIONAL_SPEC_URL}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
    >
      <BookOpen className="h-3.5 w-3.5" />
      Parameter roles &amp; constraints
    </a>
  );
}
