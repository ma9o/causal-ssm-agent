"use client";

import { createContext, type ReactNode, useCallback, useContext, useMemo, useState } from "react";

interface WorkspaceViewState {
  /** Move affordances hidden: the backing facade is read-only (hosted viewer). */
  readOnly: boolean;
  /** Currently focused analysis scenario key (simulate tool-call id or `baseline:<treatment>`). */
  selectedScenarioKey: string | null;
  /** Focus a analysis scenario in the viewer (null clears the selection). */
  selectScenario: (key: string | null) => void;
}

const WorkspaceViewContext = createContext<WorkspaceViewState | null>(null);

export function useWorkspaceView() {
  const ctx = useContext(WorkspaceViewContext);
  if (!ctx) throw new Error("useWorkspaceView must be used within WorkspaceViewProvider");
  return ctx;
}

export function WorkspaceViewProvider({
  children,
  readOnly = false,
}: {
  children: ReactNode;
  readOnly?: boolean;
}) {
  const [selectedScenarioKey, setSelectedScenarioKey] = useState<string | null>(null);
  const selectScenario = useCallback((key: string | null) => setSelectedScenarioKey(key), []);

  const value = useMemo<WorkspaceViewState>(
    () => ({ readOnly, selectedScenarioKey, selectScenario }),
    [readOnly, selectedScenarioKey, selectScenario],
  );

  return <WorkspaceViewContext.Provider value={value}>{children}</WorkspaceViewContext.Provider>;
}
