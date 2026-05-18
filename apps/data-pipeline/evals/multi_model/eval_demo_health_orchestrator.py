"""Inspect AI evaluation for orchestrator models on the DEMO_HEALTH fixture.

Ranks competing orchestrator models on how well they reproduce the fixed
DEMO_HEALTH Stage 2 fixture after running the real Stage 1a -> 1b -> 2
domain components against a fixed Stage 0 parquet and fixed worker model.

Usage:
    inspect eval evals/multi_model/eval_demo_health_orchestrator.py
    inspect eval evals/multi_model/eval_demo_health_orchestrator.py --model openrouter/anthropic/claude-opus-4.6
    inspect eval evals/multi_model/eval_demo_health_orchestrator.py -T models=openrouter/anthropic/claude-opus-4.6,openrouter/openai/gpt-5.1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import logging
from dataclasses import dataclass

from evals.common import (
    format_labeled_candidates,
    get_generate_config,
    load_eval_config,
    make_anonymous_label_mapping,
    make_generate_fn,
    parse_csv_task_arg,
    score_judge_ranking_response,
)
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from nof1_causal_lab.flows.pipeline_helpers import format_schema_for_llm
from nof1_causal_lab.flows.stages.stage2.flow import run_stage2_extraction_core
from nof1_causal_lab.flows.stages.stage2.materialization import materialize_stage2_outputs
from nof1_causal_lab.orchestrator.stage1a import Stage1aResult, run_stage1a
from nof1_causal_lab.orchestrator.stage1b import Stage1bResult, run_stage1b
from nof1_causal_lab.utils.config import get_config
from nof1_causal_lab.utils.demo_health_fixture import (
    FIXTURE_USER_ID,
    DemoHealthComparison,
    compare_demo_health_outputs,
    load_demo_health_fixture,
)
from nof1_causal_lab.utils.litellm_client import GenerateConfig as PipelineGenerateConfig
from nof1_causal_lab.utils.llm import make_generate_fn as make_pipeline_generate_fn
from nof1_causal_lab.workers.core import run_worker_extraction

_CONFIG = load_eval_config()
ORCHESTRATOR_MODELS = {m["id"]: m["alias"] for m in _CONFIG["orchestrator_models"]}
LOGGER = logging.getLogger(__name__)

JUDGE_SYSTEM = """\
You are an expert evaluator assessing which orchestrator model best reproduces a
fixed Stage 2 fixture.

Each candidate used:
- the same fixed DEMO_HEALTH Stage 0 parquet and question
- the same fixed Stage 2 worker model
- the same production Stage 1a, Stage 1b, and Stage 2 domain components

Each candidate report uses the same ordered multi-level comparison primitive:
1. `stage1b_surface`: indicator identity
2. `stage2_structure`: row shape and per-indicator counts
3. `stage2_values`: row-for-row raw/model value agreement

Rank candidates lexicographically by these levels:
- A candidate that is better on an earlier level must outrank one that is only
  better on a later level.
- If candidates fail the same earliest level, prefer fewer issues in that level.
- Use later levels only as tie-breakers after earlier levels are matched.
- Ignore prose quality or verbosity. Judge only fixture reproduction quality.

Return JSON:
```json
{
  "ranking": ["A", "B", "C"],
  "rationale": {
    "A": "Brief explanation",
    "B": "Brief explanation",
    "C": "Brief explanation"
  },
  "winner": "A"
}
```
"""

JUDGE_USER = """\
## Fixed Fixture

- fixture: `{fixture_user_id}`
- question: `{question}`
- fixed worker model: `{worker_model}`

## Candidate Reports

{candidates}

Rank the candidates from best to worst using the ordered comparison levels above.
"""


@dataclass
class CandidateRun:
    """End-to-end candidate result for one orchestrator model."""

    model_id: str
    stage1a_result: Stage1aResult
    stage1b_result: Stage1bResult
    comparison: DemoHealthComparison
    row_count: int


def create_eval_dataset() -> MemoryDataset:
    """Create the fixed DEMO_HEALTH eval dataset."""
    fixture = load_demo_health_fixture()
    return MemoryDataset(
        [
            Sample(
                input=fixture.question,
                id="demo_health",
                metadata={
                    "fixture_user_id": FIXTURE_USER_ID,
                    "question": fixture.question,
                },
            )
        ]
    )


def _dataset_summary(stage0_df) -> str:
    return f"{stage0_df.shape[0]} rows x {stage0_df.shape[1]} columns"


def _make_eval_semantic_chunk_runner(worker_model_id: str, chunk_timeout_seconds: float):
    """Create a non-Prefect semantic chunk runner for Inspect evals."""
    worker_model = get_model(worker_model_id)
    worker_generate = make_generate_fn(worker_model)

    async def _runner(
        *,
        chunk_texts: list[str],
        chunk_window_starts: list[list[str]],
        chunk_contexts: list[dict],
        question: str,
        root_run_id: str | None,
        max_concurrent_workers: int,
        max_rpm: int,
    ) -> tuple[list[dict], list[dict], int, dict | None]:
        del root_run_id, max_rpm
        semaphore = asyncio.Semaphore(max(1, max_concurrent_workers))

        async def _run_chunk(idx: int) -> tuple[int, dict]:
            async with semaphore:
                try:
                    result = await asyncio.wait_for(
                        run_worker_extraction(
                            window_text=chunk_texts[idx],
                            window_starts=chunk_window_starts[idx],
                            question=question,
                            causal_spec=chunk_contexts[idx],
                            generate=worker_generate,
                            logger=LOGGER,
                            call_label=f"eval stage2 chunk={idx}",
                        ),
                        timeout=chunk_timeout_seconds,
                    )
                except TimeoutError:
                    return idx, {
                        "dataframe": [],
                        "n_extractions": 0,
                        "status": "failed",
                        "n_windows": len(chunk_window_starts[idx]),
                        "error": f"chunk timed out after {chunk_timeout_seconds}s",
                    }
                except Exception as exc:  # noqa: BLE001
                    return idx, {
                        "dataframe": [],
                        "n_extractions": 0,
                        "status": "failed",
                        "n_windows": len(chunk_window_starts[idx]),
                        "error": str(exc),
                    }

                return idx, {
                    "dataframe": result.dataframe.to_dicts(),
                    "n_extractions": len(result.output.extractions),
                    "status": "completed",
                    "n_windows": len(chunk_window_starts[idx]),
                }

        completed = await asyncio.gather(*[_run_chunk(idx) for idx in range(len(chunk_texts))])
        completed.sort(key=lambda item: item[0])

        semantic_rows: list[dict] = []
        worker_statuses: list[dict] = []
        n_total = 0

        for idx, result in completed:
            n_total += int(result.get("n_extractions", 0))
            semantic_rows.extend(result.get("dataframe", []))
            status = {
                "worker_id": idx,
                "status": result.get("status", "completed"),
                "n_extractions": int(result.get("n_extractions", 0)),
                "n_windows": int(result.get("n_windows", len(chunk_window_starts[idx]))),
            }
            if result.get("error"):
                status["error"] = result["error"]
            worker_statuses.append(status)

        return semantic_rows, worker_statuses, n_total, None

    return _runner


async def _run_orchestrator_candidate(
    model_id: str,
    worker_model_id: str,
    chunk_timeout_seconds: float,
) -> CandidateRun:
    fixture = load_demo_health_fixture()
    config = get_config()
    stage2_workers = config.stage2_workers
    candidate_generate_config = PipelineGenerateConfig(
        max_tokens=config.llm.max_tokens,
        timeout=config.llm.timeout,
        verbosity="max",
        effort="max",
        reasoning_effort=config.llm.reasoning_effort,
        reasoning_history="all",
    )
    generate = make_pipeline_generate_fn(model_id, config=candidate_generate_config)

    stage1a_result = await run_stage1a(
        question=fixture.question,
        generate=generate,
    )

    dataset_schema = format_schema_for_llm(fixture.stage0, fixture.column_descriptions)
    stage1b_result = await run_stage1b(
        question=fixture.question,
        latent_model=stage1a_result.latent_model,
        chunks=[dataset_schema],
        generate=generate,
        dataset_summary=_dataset_summary(fixture.stage0),
    )

    stage2_result = await run_stage2_extraction_core(
        raw_df=fixture.stage0,
        question=fixture.question,
        causal_spec=stage1b_result.causal_spec,
        stage2_workers=stage2_workers,
        semantic_chunk_runner=_make_eval_semantic_chunk_runner(
            worker_model_id,
            chunk_timeout_seconds,
        ),
    )
    materialized = materialize_stage2_outputs(stage2_result, stage1b_result.causal_spec)

    data_for_model = materialized["data_for_model"]
    comparison = compare_demo_health_outputs(
        causal_spec=stage1b_result.causal_spec,
        stage0=fixture.stage0,
        data_for_model=data_for_model,
        expected_model=fixture.expected_model,
    )

    return CandidateRun(
        model_id=model_id,
        stage1a_result=stage1a_result,
        stage1b_result=stage1b_result,
        comparison=comparison,
        row_count=data_for_model.height,
    )


def _format_candidate_report(candidate: CandidateRun) -> str:
    outcome_name = candidate.stage1a_result.outcome_name or "[missing outcome]"
    indicators = ", ".join(candidate.comparison.stage1b_indicators)
    return "\n".join(
        [
            f"- stage1a constructs: {candidate.stage1a_result.n_constructs}",
            f"- stage1a edges: {candidate.stage1a_result.n_edges}",
            f"- stage1a outcome: {outcome_name}",
            f"- stage1b indicators ({len(candidate.comparison.stage1b_indicators)}): {indicators}",
            f"- stage2 rows: {candidate.row_count}",
            "",
            candidate.comparison.format_report(),
        ]
    )


def judge_solver(
    models: list[str] | None = None,
    worker_model: str | None = None,
    candidate_timeout_seconds: float | None = None,
    chunk_timeout_seconds: float | None = None,
):
    """Run candidates and ask the judge model to rank them."""
    if models is None:
        models = list(ORCHESTRATOR_MODELS.keys())
    if worker_model is None:
        worker_model = get_config().stage2_workers.model
    if candidate_timeout_seconds is None:
        candidate_timeout_seconds = _CONFIG.get("orchestrator_candidate_timeout_seconds", 1800)
    if chunk_timeout_seconds is None:
        chunk_timeout_seconds = _CONFIG.get("orchestrator_chunk_timeout_seconds", 300)

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: ARG001
            question = state.metadata["question"]

            async def _safe_run(model_id: str) -> tuple[str, CandidateRun | str]:
                try:
                    return model_id, await asyncio.wait_for(
                        _run_orchestrator_candidate(
                            model_id,
                            worker_model,
                            chunk_timeout_seconds,
                        ),
                        timeout=candidate_timeout_seconds,
                    )
                except TimeoutError:
                    return (
                        model_id,
                        f"[TIMEOUT: Candidate did not finish within {candidate_timeout_seconds}s]",
                    )
                except Exception as exc:  # noqa: BLE001
                    return model_id, f"[ERROR: {exc}]"

            results = await asyncio.gather(*[_safe_run(model_id) for model_id in models])

            candidates: dict[str, CandidateRun] = {}
            failure_reports: dict[str, str] = {}
            for model_id, result in results:
                if isinstance(result, CandidateRun):
                    candidates[model_id] = result
                else:
                    failure_reports[model_id] = result

            label_mapping = make_anonymous_label_mapping(
                sample_id=state.sample_id,
                candidate_ids=models,
            )

            def _render_candidate(model_id: str) -> str:
                if model_id in candidates:
                    return _format_candidate_report(candidates[model_id])
                return "- run failed before a comparison report was produced."

            candidate_sections = format_labeled_candidates(label_mapping, _render_candidate)

            judge_prompt = JUDGE_USER.format(
                fixture_user_id=FIXTURE_USER_ID,
                question=question,
                worker_model=worker_model,
                candidates=candidate_sections,
            )

            state.metadata["label_map"] = label_mapping.label_map
            state.metadata["reverse_label_map"] = label_mapping.reverse_label_map
            state.metadata["candidate_reports"] = {
                model_id: (
                    candidates[model_id].comparison.format_report()
                    if model_id in candidates
                    else failure_reports[model_id]
                )
                for model_id in models
            }

            state.messages = [
                ChatMessageSystem(content=JUDGE_SYSTEM),
                ChatMessageUser(content=judge_prompt),
            ]

            judge_model = get_model()
            response = await judge_model.generate(state.messages, config=get_generate_config())
            state.output.completion = response.completion
            return state

        return solve

    return _solver()


@scorer(metrics=[mean(), stderr()])
def demo_health_ranking_scorer():
    """Score by parsing the judge ranking response."""

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        return score_judge_ranking_response(
            completion=state.output.completion,
            reverse_label_map=state.metadata.get("reverse_label_map", {}),
            alias_lookup=ORCHESTRATOR_MODELS,
            extra_metadata={
                "candidate_reports": state.metadata.get("candidate_reports", {}),
            },
        )

    return score


@task
def demo_health_orchestrator_eval(
    models: str | list[str] | None = None,
    worker_model: str | None = None,
    candidate_timeout_seconds: float | None = None,
    chunk_timeout_seconds: float | None = None,
):
    """Judge-rank orchestrator models on DEMO_HEALTH fixture reproduction."""
    model_ids = parse_csv_task_arg(models)
    return Task(
        dataset=create_eval_dataset(),
        solver=[
            judge_solver(
                models=model_ids,
                worker_model=worker_model,
                candidate_timeout_seconds=candidate_timeout_seconds,
                chunk_timeout_seconds=chunk_timeout_seconds,
            )
        ],
        scorer=demo_health_ranking_scorer(),
    )
