"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { uploadFile } from "@/lib/api/endpoints";
import { isMockMode, MOCK_RUN_ID } from "@/lib/api/mock-provider";
import { getDeploymentId, triggerRun } from "@/lib/api/prefect";
import { generateSessionCode } from "@/lib/session-code";
import { Switch } from "@/components/ui/switch";
import { ArrowRight, FileText, Loader2, RotateCcw, ShieldAlert, Sparkles, Upload, X } from "lucide-react";
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

export default function LandingPage() {
  const router = useRouter();
  const [question, setQuestion] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [overrideGates, setOverrideGates] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isMac, setIsMac] = useState(false);

  useEffect(() => {
    if (isMockMode() && !sessionStorage.getItem("mock-landed")) {
      sessionStorage.setItem("mock-landed", "true");
      router.push(`/analysis/${MOCK_RUN_ID}`);
    }
  }, [router]);

  useEffect(() => {
    setIsMac(/Mac/.test(navigator.userAgent));
  }, []);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [resumeCode, setResumeCode] = useState("");
  const [resumeError, setResumeError] = useState<string | null>(null);
  const [isResuming, setIsResuming] = useState(false);

  const validateFile = useCallback((f: File): string | null => {
    const validTypes = [".zip", ".json"];
    const ext = f.name.toLowerCase().slice(f.name.lastIndexOf("."));
    if (!validTypes.includes(ext)) {
      return `Invalid file type "${ext}". Please upload a ZIP or JSON file.`;
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

      const code = generateSessionCode();
      await uploadFile(file, code);

      const deploymentId = await getDeploymentId();
      const runId = await triggerRun(deploymentId, {
        query: question,
        user_id: code,
        override_gates: overrideGates || undefined,
      });

      await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code, runId, question }),
      });

      router.push(`/analysis/${runId}?code=${code}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start analysis");
      setIsSubmitting(false);
    }
  };

  const handleResume = async () => {
    const code = resumeCode.trim().toUpperCase();
    if (code.length !== 6) {
      setResumeError("Session code must be 6 characters.");
      return;
    }

    setIsResuming(true);
    setResumeError(null);

    try {
      const res = await fetch(`/api/sessions/${code}`);
      if (!res.ok) {
        setResumeError("Session not found. Check the code and try again.");
        setIsResuming(false);
        return;
      }
      const { runId } = await res.json();
      router.push(`/analysis/${runId}?code=${code}`);
    } catch {
      setResumeError("Failed to look up session.");
      setIsResuming(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && question.trim() && file && !isSubmitting) {
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
            From research question to quantified treatment effects — powered by LLMs, state-space
            models, and Bayesian inference
          </p>
        </motion.div>

        <motion.div {...fadeInUp()}>
          <Card>
            <CardHeader>
              <CardTitle>Research Question</CardTitle>
              <CardDescription>What causal relationship do you want to investigate?</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <textarea
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

        <motion.div {...fadeInUp(0.1)}>
          <Card>
            <CardHeader>
              <CardTitle>Data Upload</CardTitle>
              <CardDescription>
                Upload your Google Takeout export (ZIP or JSON)
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
                      ZIP or JSON, up to {prettyBytes(MAX_FILE_SIZE)}
                    </p>
                  </>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  accept=".zip,.json"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleFileSelect(f);
                  }}
                />
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div {...fadeInUp(0.15)}>
          <Card>
            <CardContent className="flex items-center justify-between py-4">
              <div className="flex items-center gap-2">
                <ShieldAlert className="h-4 w-4 text-warning-foreground" />
                <div>
                  <p className="text-sm font-medium">Override stage gates</p>
                  <p className="text-xs text-muted-foreground">
                    Continue past stage failures instead of halting. Results may be unreliable.
                  </p>
                </div>
              </div>
              <Switch
                checked={overrideGates}
                onCheckedChange={setOverrideGates}
              />
            </CardContent>
          </Card>
        </motion.div>

        {error && (
          <motion.p className="text-sm text-destructive text-center" {...fadeIn}>
            {error}
          </motion.p>
        )}

        <motion.div className="space-y-2" {...fadeInUp(0.2)}>
          <Button
            className="w-full"
            size="lg"
            onClick={handleSubmit}
            disabled={isSubmitting || !question.trim() || !file}
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

        <motion.div className="space-y-3" {...fadeInUp(0.25)}>
          <p className="text-center text-sm text-muted-foreground">
            Resume a previous session
          </p>
          <div className="flex items-center gap-2 max-w-xs mx-auto">
            <input
              type="text"
              maxLength={6}
              placeholder="ABC123"
              value={resumeCode}
              onChange={(e) => {
                setResumeCode(e.target.value.toUpperCase());
                if (resumeError) setResumeError(null);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && resumeCode.trim().length === 6) handleResume();
              }}
              className="flex-1 rounded-md border bg-background px-3 py-2 text-center font-mono text-lg tracking-[0.3em] uppercase placeholder:text-muted-foreground/40 placeholder:tracking-[0.3em]"
            />
            <Button
              variant="outline"
              onClick={handleResume}
              disabled={isResuming || resumeCode.trim().length !== 6}
            >
              {isResuming ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RotateCcw className="h-4 w-4" />
              )}
              Resume
            </Button>
          </div>
          {resumeError && (
            <p className="text-center text-sm text-destructive">{resumeError}</p>
          )}
        </motion.div>
      </div>
    </div>
  );
}
