import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { AccessStatus } from "@/lib/auth-status";
import { linkifyDocRefs } from "@/lib/utils/linkify-docs";
import {
  ArrowRight,
  FileText,
  KeyRound,
  Loader2,
  Upload,
  X,
} from "lucide-react";
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

function AccessIndicator({ access }: { access: AccessStatus | null }) {
  if (!access) {
    return <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />;
  }
  if (access.mode === "user" || access.mode === "trial") {
    return <div className="h-2 w-2 rounded-full bg-success" />;
  }
  return <div className="h-2 w-2 rounded-full bg-destructive" />;
}

function AccessMessage({ access }: { access: AccessStatus | null }) {
  if (!access) {
    return <p className="text-sm text-muted-foreground">Checking access...</p>;
  }
  if (access.mode === "user") {
    return <p className="text-sm font-medium">Using your OpenRouter account</p>;
  }
  if (access.mode === "trial" && access.creditStatus === "available") {
    return (
      <>
        <p className="text-sm font-medium">Free credits available</p>
        <p className="text-xs text-muted-foreground">
          Or sign in with OpenRouter to use your own key
        </p>
      </>
    );
  }
  if (access.mode === "trial" && access.creditStatus === "unknown") {
    return (
      <>
        <p className="text-sm font-medium">Trial access available</p>
        <p className="text-xs text-muted-foreground">
          Credit status is unavailable, but the server can still run requests
        </p>
      </>
    );
  }
  if (access.mode === "none" && access.reason === "trial_exhausted") {
    return (
      <>
        <p className="text-sm font-medium text-destructive">
          Free credits exhausted
        </p>
        <p className="text-xs text-muted-foreground">
          Sign in with OpenRouter to continue with your own key
        </p>
      </>
    );
  }
  return (
    <>
      <p className="text-sm font-medium text-destructive">
        Trial access unavailable
      </p>
      <p className="text-xs text-muted-foreground">
        Sign in with OpenRouter to continue
      </p>
    </>
  );
}

export type LandingPageViewProps = {
  access: AccessStatus | null;
  noAccess: boolean;
  onSignOut: () => void;
  onOpenRouterAuth: () => void;

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
  access,
  noAccess,
  onSignOut,
  onOpenRouterAuth,
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
    <div className="flex min-h-screen flex-col items-center justify-center px-4 py-6 sm:px-6">
      <div className="w-full max-w-2xl space-y-4 sm:space-y-6">
        <motion.div className="text-center space-y-2" {...fadeIn}>
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight">
            causal-ssm-agent
          </h1>
        </motion.div>

        <motion.div {...fadeInUp()}>
          <Card
            className={`shadow-sm ${noAccess ? "border-destructive/50" : ""}`}
          >
            <CardContent className="flex items-center justify-between py-4">
              <div className="flex items-center gap-3">
                <AccessIndicator access={access} />
                <div>
                  <AccessMessage access={access} />
                </div>
              </div>
              {access?.mode === "user" ? (
                <Button variant="ghost" size="sm" onClick={onSignOut}>
                  Sign out
                </Button>
              ) : (
                <Button
                  variant={noAccess ? "default" : "outline"}
                  size="sm"
                  onClick={onOpenRouterAuth}
                >
                  <KeyRound className="h-3.5 w-3.5 mr-1.5" />
                  Sign in with OpenRouter
                </Button>
              )}
            </CardContent>
          </Card>
        </motion.div>

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
                      <p className="text-xs text-muted-foreground">
                        {prettyBytes(file.size)}
                      </p>
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
          <motion.p
            className="text-sm text-destructive text-center"
            {...fadeIn}
          >
            {error}
          </motion.p>
        )}

        <motion.div className="space-y-2" {...fadeInUp(0.25)}>
          <Button
            className="w-full"
            size="lg"
            onClick={onSubmit}
            disabled={submitDisabled}
          >
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
      </div>
    </div>
  );
}
