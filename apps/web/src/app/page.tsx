"use client";

import { LandingPageView, MAX_FILE_SIZE } from "@/components/landing/landing-page-view";
import { WorkspacesRail } from "@/components/pipeline/workspaces-rail";
import { apiFetch } from "@/lib/api/client";
import { getCapabilities, getCapabilitiesQueryKey } from "@/lib/api/capabilities";
import { getWorkspaces, getWorkspacesQueryKey } from "@/lib/api/workspaces";
import { uploadFile } from "@/lib/api/endpoints";
import { getMockFixture, isMockMode } from "@/lib/api/mock-provider";
import { generateAnonymousWorkspaceId } from "@/lib/workspace-id";
import { useQuery } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import prettyBytes from "pretty-bytes";
import { useCallback, useEffect, useState } from "react";

export default function LandingPage() {
  const router = useRouter();
  const [question, setQuestion] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const capabilitiesQuery = useQuery({
    queryKey: getCapabilitiesQueryKey(),
    queryFn: getCapabilities,
    staleTime: Infinity,
    retry: false,
  });
  const movesEnabled = capabilitiesQuery.data?.moves_enabled ?? null;

  const workspacesQuery = useQuery({
    queryKey: getWorkspacesQueryKey(),
    queryFn: getWorkspaces,
    staleTime: 30_000,
    retry: false,
  });

  useEffect(() => {
    if (isMockMode() && !sessionStorage.getItem("mock-landed")) {
      sessionStorage.setItem("mock-landed", "true");
      router.push(`/analysis/${getMockFixture()}`);
    }
  }, [router]);

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

      await apiFetch<{ workspaceId: string }>("/api/runs", {
        method: "POST",
        body: JSON.stringify({
          workspaceId,
          query: question,
        }),
      });

      router.push(`/analysis/${workspaceId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start analysis");
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4 py-6 sm:px-6 xl:grid xl:grid-cols-[1fr_auto_1fr] xl:items-center xl:gap-6">
      <LandingPageView
        movesEnabled={movesEnabled}
        question={question}
        onQuestionChange={(q) => {
          setQuestion(q);
          if (error) setError(null);
        }}
        file={file}
        onFileSelect={handleFileSelect}
        onFileRemove={() => setFile(null)}
        isSubmitting={isSubmitting}
        submitDisabled={isSubmitting || !question.trim() || !file || movesEnabled === false}
        onSubmit={handleSubmit}
        error={error}
      />
      <div className="hidden xl:block w-px h-2/3 bg-border" />
      <WorkspacesRail
        data={workspacesQuery.data}
        error={workspacesQuery.error?.message ?? null}
        isLoading={workspacesQuery.isLoading}
      />
    </div>
  );
}
