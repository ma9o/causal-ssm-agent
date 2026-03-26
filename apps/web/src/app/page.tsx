"use client";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { AccessStatus } from "@/lib/auth-status";
import { apiFetch } from "@/lib/api/client";
import { unlockWorkspace } from "@/lib/api/analysis";
import { uploadFile } from "@/lib/api/endpoints";
import { getMockFixture, isMockMode } from "@/lib/api/mock-provider";
import { initiateOpenRouterAuth } from "@/lib/auth";
import { getIdentity, setIdentity } from "@/lib/identity";
import { useAuth } from "@/lib/hooks/use-auth";
import { ANONYMOUS_WORKSPACE_ID_LENGTH } from "@/lib/workspace-id";
import { getSharedWorkspaceAccessCode, parseResumeKey } from "@/lib/resume-key";
import {
  ArrowRight,
  FileText,
  KeyRound,
  Loader2,
  RotateCcw,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import { motion } from "motion/react";
import { useRouter } from "next/navigation";
import prettyBytes from "pretty-bytes";
import { useCallback, useEffect, useRef, useState } from "react";

const EXAMPLE_QUESTIONS = [
  "How does my daily screen time affect my sleep quality and mood?",
  "Does exercise frequency causally influence my productivity at work?",
  "What is the effect of social media usage on my anxiety levels?",
];

const MAX_FILE_SIZE = 500 * 1024 * 1024; // 500 MB

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

function renderAccessIndicator(access: AccessStatus | null) {
  if (!access) {
    return <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />;
  }
  if (access.mode === "user" || access.mode === "trial") {
    return <div className="h-2 w-2 rounded-full bg-success" />;
  }
  return <div className="h-2 w-2 rounded-full bg-destructive" />;
}

function renderAccessMessage(access: AccessStatus | null) {
  if (!access) {
    return <p className="text-sm text-muted-foreground">Checking access...</p>;
  }
  if (access.mode === "user") {
    return <p className="text-sm font-medium">Using your OpenRouter session</p>;
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

export default function LandingPage() {
  const router = useRouter();
  const [question, setQuestion] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isMac, setIsMac] = useState(false);
  useEffect(() => {
    setIsMac(/Mac/.test(navigator.userAgent));
  }, []);
  const auth = useAuth();
  const launchIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (isMockMode() && !sessionStorage.getItem("mock-landed")) {
      sessionStorage.setItem("mock-landed", "true");
      const workspaceId = getMockFixture();
      const sharedAccessCode = getSharedWorkspaceAccessCode(workspaceId);
      if (!sharedAccessCode) {
        router.push(`/analysis/${workspaceId}`);
        return;
      }
      void unlockWorkspace(workspaceId, sharedAccessCode)
        .catch(() => {
          // The analysis page retries with the stored identity if the cookie was not set here.
        })
        .finally(() => {
          setIdentity({
            workspaceId,
            accessCode: sharedAccessCode,
            kind: "anonymous",
          });
          router.push(`/analysis/${workspaceId}`);
        });
    }
  }, [router]);
  useEffect(() => {
    launchIdRef.current = null;
  }, [question, file]);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [resumeKeyInput, setResumeKeyInput] = useState("");
  const [resumeError, setResumeError] = useState<string | null>(null);
  const [isResuming, setIsResuming] = useState(false);

  const validateFile = useCallback((f: File): string | null => {
    if (!f.name.trim()) {
      return "Please choose a file to upload.";
    }
    if (f.size > MAX_FILE_SIZE) {
      return `File too large (${prettyBytes(f.size)}). Maximum size is ${prettyBytes(MAX_FILE_SIZE)}.`;
    }
    return null;
  }, []);

  const handleFileSelect = useCallback(
    (f: File) => {
      const validationError = validateFile(f);
      if (validationError) {
        setError(validationError);
        return;
      }
      setError(null);
      setFile(f);
    },
    [validateFile],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const dropped = e.dataTransfer.files[0];
      if (dropped) handleFileSelect(dropped);
    },
    [handleFileSelect],
  );

  const handleSubmit = async () => {
    if (!question.trim()) {
      setError("Please enter a research question.");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      if (!file) {
        setError("Please upload your data file.");
        setIsSubmitting(false);
        return;
      }

      const identity = auth.ensureIdentity();
      await uploadFile(file, identity.workspaceId, identity.accessCode);
      const launchId = launchIdRef.current ?? crypto.randomUUID();
      launchIdRef.current = launchId;

      const { rootFlowRunId } = await apiFetch<{ rootFlowRunId: string }>(
        "/api/runs",
        {
          method: "POST",
          body: JSON.stringify({
            workspaceId: identity.workspaceId,
            accessCode: identity.accessCode,
            launchId,
            query: question,
          }),
        },
      );

      router.push(
        `/analysis/${identity.workspaceId}?${new URLSearchParams({ rootFlowRunId }).toString()}`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start analysis");
      setIsSubmitting(false);
    }
  };

  const handleResume = async () => {
    const parsedResume = parseResumeKey(resumeKeyInput);
    if (!parsedResume) {
      setResumeError("Enter a resume key or fixture ID.");
      return;
    }

    const rawWorkspaceId = parsedResume.workspaceId.trim();
    const workspaceId =
      rawWorkspaceId.length === ANONYMOUS_WORKSPACE_ID_LENGTH &&
      /^[A-Za-z0-9]+$/.test(rawWorkspaceId)
        ? rawWorkspaceId.toUpperCase()
        : rawWorkspaceId;
    const storedIdentity = getIdentity();
    const accessCode =
      parsedResume.accessCode ??
      getSharedWorkspaceAccessCode(workspaceId) ??
      (storedIdentity?.workspaceId === workspaceId
        ? storedIdentity.accessCode
        : null);

    setIsResuming(true);
    setResumeError(null);

    try {
      if (!accessCode) {
        setResumeError("Use the full resume key for this workspace.");
        setIsResuming(false);
        return;
      }

      await unlockWorkspace(workspaceId, accessCode);
      setIdentity({
        workspaceId,
        accessCode,
        kind: storedIdentity?.kind ?? "anonymous",
      });
      router.push(`/analysis/${workspaceId}`);
    } catch (err) {
      setResumeError(
        err instanceof Error ? err.message : "Failed to unlock workspace.",
      );
      setIsResuming(false);
    }
  };

  const handleOpenRouterAuth = async () => {
    await initiateOpenRouterAuth(`${window.location.origin}/auth/callback`);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (
      e.key === "Enter" &&
      (e.metaKey || e.ctrlKey) &&
      question.trim() &&
      file &&
      !isSubmitting
    ) {
      handleSubmit();
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-4 sm:p-8">
      <div className="w-full max-w-2xl space-y-8">
        <motion.div className="text-center space-y-3" {...fadeIn}>
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight">
            Causal Inference Pipeline
          </h1>
          <p className="text-base sm:text-lg text-muted-foreground max-w-lg mx-auto">
            From research question to quantified treatment effects — powered by
            LLMs, state-space models, and Bayesian inference
          </p>
        </motion.div>

        <motion.div {...fadeInUp()}>
          <Card className={auth.noAccess ? "border-destructive/50" : ""}>
            <CardContent className="flex items-center justify-between py-4">
              <div className="flex items-center gap-3">
                {renderAccessIndicator(auth.access)}
                <div>{renderAccessMessage(auth.access)}</div>
              </div>
              {auth.access?.mode === "user" ? (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => void auth.signOut()}
                >
                  Sign out
                </Button>
              ) : (
                <Button
                  variant={auth.noAccess ? "default" : "outline"}
                  size="sm"
                  onClick={handleOpenRouterAuth}
                >
                  <KeyRound className="h-3.5 w-3.5 mr-1.5" />
                  Sign in with OpenRouter
                </Button>
              )}
            </CardContent>
          </Card>
        </motion.div>

        <motion.div {...fadeInUp(0.05)}>
          <Card>
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
                onChange={(e) => {
                  setQuestion(e.target.value);
                  if (error) setError(null);
                }}
                onKeyDown={handleKeyDown}
              />
              <div className="space-y-2">
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <Sparkles className="h-3 w-3" />
                  Try an example:
                </p>
                <div className="flex flex-wrap gap-2">
                  {EXAMPLE_QUESTIONS.map((q) => (
                    <button
                      key={q}
                      type="button"
                      className="rounded-full border px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-primary hover:text-foreground"
                      onClick={() => setQuestion(q)}
                    >
                      {q.length > 50 ? `${q.slice(0, 50)}...` : q}
                    </button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div {...fadeInUp(0.15)}>
          <Card>
            <CardHeader>
              <CardTitle>Data Upload</CardTitle>
              <CardDescription>
                Upload your Google Takeout export or any text data file
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
                      onClick={() => setFile(null)}
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
                    if (f) handleFileSelect(f);
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
            onClick={handleSubmit}
            disabled={
              isSubmitting || !question.trim() || !file || auth.noAccess
            }
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
          <p className="text-center text-xs text-muted-foreground/60">
            Press{" "}
            <kbd className="rounded border bg-secondary px-1 py-0.5 text-[10px] font-mono">
              {isMac ? "\u2318" : "Ctrl"}
            </kbd>
            +
            <kbd className="rounded border bg-secondary px-1 py-0.5 text-[10px] font-mono">
              Enter
            </kbd>{" "}
            to submit
          </p>
        </motion.div>

        <div className="flex items-center gap-3 opacity-40">
          <div className="flex-1 border-t" />
          <span className="text-xs text-muted-foreground">or</span>
          <div className="flex-1 border-t" />
        </div>

        <motion.div className="space-y-3" {...fadeInUp(0.3)}>
          <p className="text-center text-sm text-muted-foreground">
            Resume a previous analysis
          </p>
          <div className="flex items-center gap-2 max-w-xs mx-auto">
            <input
              type="text"
              aria-label="Resume key or fixture ID"
              placeholder="Resume key or fixture ID"
              value={resumeKeyInput}
              onChange={(e) => {
                setResumeKeyInput(e.target.value);
                if (resumeError) setResumeError(null);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && resumeKeyInput.trim()) handleResume();
              }}
              className="flex-1 rounded-md border bg-background px-3 py-2 font-mono text-sm placeholder:text-muted-foreground/40"
            />
            <Button
              variant="outline"
              onClick={handleResume}
              disabled={isResuming || !resumeKeyInput.trim()}
            >
              {isResuming ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RotateCcw className="h-4 w-4" />
              )}
              Resume
            </Button>
          </div>
          <p className="text-center text-xs text-muted-foreground/70">
            Paste your resume key, or enter a shared fixture ID like{" "}
            <span className="font-mono">DEFAULT</span>.
          </p>
          {resumeError && (
            <p className="text-center text-sm text-destructive">
              {resumeError}
            </p>
          )}
        </motion.div>
      </div>
    </div>
  );
}
