"use client";

import {
  LandingPageView,
  MAX_FILE_SIZE,
} from "@/components/landing/landing-page-view";
import { apiFetch } from "@/lib/api/client";
import { uploadFile } from "@/lib/api/endpoints";
import { getMockFixture, isMockMode } from "@/lib/api/mock-provider";
import { initiateOpenRouterAuth } from "@/lib/auth";
import { useAuth } from "@/lib/hooks/use-auth";
import { generateAnonymousWorkspaceId } from "@/lib/workspace-id";
import { useRouter } from "next/navigation";
import prettyBytes from "pretty-bytes";
import { useCallback, useEffect, useRef, useState } from "react";

export default function LandingPage() {
  const router = useRouter();
  const [question, setQuestion] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const auth = useAuth();
  const launchIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (isMockMode() && !sessionStorage.getItem("mock-landed")) {
      sessionStorage.setItem("mock-landed", "true");
      router.push(`/analysis/${getMockFixture()}`);
    }
  }, [router]);

  useEffect(() => {
    launchIdRef.current = null;
  }, [question, file]);

  const [error, setError] = useState<string | null>(null);

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

      const workspaceId = generateAnonymousWorkspaceId();
      await uploadFile(file, workspaceId);
      const launchId = launchIdRef.current ?? crypto.randomUUID();
      launchIdRef.current = launchId;

      const { rootFlowRunId } = await apiFetch<{ rootFlowRunId: string }>(
        "/api/runs",
        {
          method: "POST",
          body: JSON.stringify({
            workspaceId,
            launchId,
            query: question,
          }),
        },
      );

      router.push(
        `/analysis/${workspaceId}?${new URLSearchParams({ rootFlowRunId }).toString()}`,
      );
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to start analysis",
      );
      setIsSubmitting(false);
    }
  };

  const handleOpenRouterAuth = async () => {
    await initiateOpenRouterAuth(`${window.location.origin}/auth/callback`);
  };


  return (
    <LandingPageView
      access={auth.access}
      noAccess={auth.noAccess}
      onSignOut={() => void auth.signOut()}
      onOpenRouterAuth={handleOpenRouterAuth}
      question={question}
      onQuestionChange={(q) => {
        setQuestion(q);
        if (error) setError(null);
      }}
      file={file}
      onFileSelect={handleFileSelect}
      onFileRemove={() => setFile(null)}
      isSubmitting={isSubmitting}
      submitDisabled={isSubmitting || !question.trim() || !file || auth.noAccess}
      onSubmit={handleSubmit}
      error={error}
    />
  );
}
