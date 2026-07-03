import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { linkifyDocRefs } from "@/lib/utils/linkify-docs";
import { ArrowRight, Eye, FileText, Loader2, Upload, X } from "lucide-react";
import { motion } from "motion/react";
import prettyBytes from "pretty-bytes";
import { useRef, useState } from "react";

export const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100 MB

const fadeIn = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  transition: { duration: 0.3, ease: "easeOut" as const },
};

const fadeInUp = (delay = 0) => ({
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.4, ease: "easeOut" as const, delay },
});

export type LandingPageViewProps = {
  /** null while capabilities are loading; false on a read-only (hosted viewer) deployment. */
  movesEnabled: boolean | null;

  question: string;
  onQuestionChange: (q: string) => void;

  file: { name: string; size: number } | null;
  onFileSelect: (f: File) => void;
  onFileRemove: () => void;

  isSubmitting: boolean;
  submitDisabled: boolean;
  onSubmit: () => void;

  error: string | null;
};

export function LandingPageView({
  movesEnabled,
  question,
  onQuestionChange,
  file,
  onFileSelect,
  onFileRemove,
  isSubmitting,
  submitDisabled,
  onSubmit,
  error,
}: LandingPageViewProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) onFileSelect(dropped);
  };

  return (
    <div className="w-full max-w-2xl mx-auto space-y-4 sm:space-y-6">
      <motion.div className="text-center space-y-2" {...fadeIn}>
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight">nof1-causal-lab</h1>
      </motion.div>

      {movesEnabled === false && (
        <motion.div {...fadeInUp()}>
          <Card className="shadow-sm">
            <CardContent className="flex items-center gap-3 py-4">
              <Eye className="h-4 w-4 text-muted-foreground" />
              <div>
                <p className="text-sm font-medium">Read-only viewer</p>
                <p className="text-xs text-muted-foreground">
                  This deployment serves published workspaces. Runs are driven against a local
                  service (via its MCP/HTTP facade) and published here.
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {movesEnabled !== false && (
        <>
          <motion.div {...fadeInUp(0.05)}>
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle>Research Question</CardTitle>
                <CardDescription>
                  What causal relationship do you want to investigate?
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <textarea
                  aria-label="Research question"
                  className="w-full rounded-md border bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground min-h-[100px] resize-y"
                  placeholder="e.g., How does my daily screen time affect my sleep quality and mood?"
                  value={question}
                  onChange={(e) => onQuestionChange(e.target.value)}
                />
              </CardContent>
            </Card>
          </motion.div>

          <motion.div {...fadeInUp(0.15)}>
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle>Data Upload</CardTitle>
                <CardDescription>
                  {linkifyDocRefs(
                    "Upload a ZIP or text file containing your observational data, without worrying about heterogeneity or sparsity. See docs/pipeline/02-indicator-extraction.md",
                  )}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  className={`relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors ${
                    dragOver
                      ? "border-primary bg-primary/5"
                      : "border-muted-foreground/25 hover:border-muted-foreground/50"
                  }`}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={handleDrop}
                >
                  {file ? (
                    <div className="flex items-center gap-3">
                      <FileText className="h-6 w-6 text-primary" />
                      <div>
                        <p className="text-sm font-medium">{file.name}</p>
                        <p className="text-xs text-muted-foreground">{prettyBytes(file.size)}</p>
                      </div>
                      <button
                        type="button"
                        className="ml-2 rounded-full p-1 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                        onClick={onFileRemove}
                        aria-label="Remove file"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                  ) : (
                    <>
                      <Upload className="h-8 w-8 text-muted-foreground mb-2" />
                      <p className="text-sm text-muted-foreground">
                        Drag and drop or{" "}
                        <button
                          type="button"
                          className="text-primary underline underline-offset-2 hover:no-underline"
                          onClick={() => fileInputRef.current?.click()}
                        >
                          browse
                        </button>
                      </p>
                      <p className="mt-1 text-xs text-muted-foreground/60">
                        ZIP or text file, up to {prettyBytes(MAX_FILE_SIZE)}
                      </p>
                    </>
                  )}
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) onFileSelect(f);
                    }}
                  />
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {error && (
            <motion.p className="text-sm text-destructive text-center" {...fadeIn}>
              {error}
            </motion.p>
          )}

          <motion.div className="space-y-2" {...fadeInUp(0.25)}>
            <Button className="w-full" size="lg" onClick={onSubmit} disabled={submitDisabled}>
              {isSubmitting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Starting Analysis...
                </>
              ) : (
                <>
                  Start Analysis
                  <ArrowRight className="h-4 w-4 ml-2" />
                </>
              )}
            </Button>
          </motion.div>
        </>
      )}
    </div>
  );
}
