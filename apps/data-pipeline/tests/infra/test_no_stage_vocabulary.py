from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCAN_ROOTS = (
    REPO_ROOT / "apps/data-pipeline/src",
    REPO_ROOT / "apps/data-pipeline/scripts",
    REPO_ROOT / "apps/data-pipeline/evaluation",
    REPO_ROOT / "apps/web/src",
    REPO_ROOT / "packages/api-types/src",
    REPO_ROOT / "packages/api-types/scripts",
)
TEXT_SUFFIXES = {
    ".css",
    ".json",
    ".md",
    ".py",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
FORBIDDEN = re.compile(
    r"stage-\d|"
    r"\bStageId\b|"
    r"\bSTAGE_IDS\b|"
    r"\bStage\d|"
    r"\bstage\d|"
    r"\bstage_id\b|"
    r"Pipeline/Stages|"
    r"Stages/[0-9]|"
    r"stage-contents|"
    r"components/stages|"
    r"stage-section|"
    r"stage-header|"
    r"stage-presentation|"
    r"stage-story|"
    r"lazy-stage|"
    r"new-stages|"
    r"active-stage|"
    r"stage-with-trace|"
    r"\bStageSection\b|"
    r"\bStagePresentation\b|"
    r"\bStageContent\b|"
    r"\bStageHeader\b|"
    r"\bStageStory\b|"
    r"\bStageWithTrace\b|"
    r"\bLazyStage\b|"
    r"\bNewStages\b|"
    r"\bActiveStage\b|"
    r"\bstageRun\b|"
    r"\bstageData\b|"
    r"\bvisibleStages\b|"
    r"\bstageStory\b|"
    r"\bcreateStage"
)


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if "__pycache__" in path.parts or "node_modules" in path.parts or ".next" in path.parts:
                continue
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                files.append(path)
    return files


def test_stage_vocabulary_is_not_reintroduced() -> None:
    hits: list[str] = []
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN.search(line):
                rel = path.relative_to(REPO_ROOT)
                hits.append(f"{rel}:{line_no}: {line.strip()}")

    assert hits == []
