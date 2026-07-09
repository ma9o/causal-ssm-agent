"""Inspect AI evaluation for orchestrator models on the DEMO fixture.

Ranks competing orchestrator models on how well they reproduce the fixed
DEMO Target 2 fixture after running the real Target 1a -> 1b -> 2
domain components against a fixed Target 0 parquet and fixed worker model.

Usage:
    inspect eval evals/multi_model/eval_demo_health_orchestrator.py
    inspect eval evals/multi_model/eval_demo_health_orchestrator.py --model openrouter/anthropic/claude-opus-4.6
    inspect eval evals/multi_model/eval_demo_health_orchestrator.py -T models=openrouter/anthropic/claude-opus-4.6,openrouter/openai/gpt-5.1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import asyncio
import logging
from dataclasses import dataclass

from evaluation.inspect_evals.common import (
    format_labeled_candidates,
    get_generate_config,
    load_eval_config,
    make_anonymous_label_mapping,
    make_eval_session_factory,
    parse_csv_task_arg,
    score_judge_ranking_response,
)
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from nof1_causal_lab.flows.pipeline_helpers import format_schema_for_llm
from nof1_causal_lab.flows.transitions.extraction.flow import run_extraction_core
from nof1_causal_lab.flows.transitions.extraction.materialization import (
    materialize_extraction_outputs,
)
from nof1_causal_lab.flows.transitions.latent_structure.run import (
    LatentStructureResult,
    run_latent_structure,
)
from nof1_causal_lab.flows.transitions.measurement_structure.assemble import build_causal_design
from nof1_causal_lab.flows.transitions.measurement_structure.run import (
    MeasurementStructureResult,
    run_measurement_structure,
)
from nof1_causal_lab.utils.causal_design import get_outcome_name
from nof1_causal_lab.utils.config import get_config
from nof1_causal_lab.utils.demo_health_fixture import (
    FIXTURE_USER_ID,
    DemoHealthComparison,
    compare_demo_health_outputs,
    load_demo_health_fixture,
)
from nof1_causal_lab.workers.core import run_worker_extraction

_CONFIG = load_eval_config()
ORCHESTRATOR_MODELS = {m["id"]: m["alias"] for m in _CONFIG["orchestrator_models"]}
LOGGER = logging.getLogger(__name__)

JUDGE_SYSTEM = """\
You are an expert evaluator assessing which orchestrator model best reproduces a
fixed Target 2 fixture.

Each candidate used:
- the same fixed DEMO Target 0 parquet and question
- the same fixed Target 2 worker model
- the same production Target 1a, Target 1b, and Target 2 domain components

Each candidate report uses the same ordered multi-level comparison primitive:
1. `measurement_surface`: indicator identity
2. `extraction_structure`: row shape and per-indicator counts
3. `extraction_values`: row-for-row raw/model value agreement

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
    latent_result: LatentStructureResult
    measurement_result: MeasurementStructureResult
    comparison: DemoHealthComparison
    row_count: int


def create_eval_dataset() -> MemoryDataset:
    """Create the fixed DEMO eval dataset."""
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


def _dataset_summary(raw_data_df) -> str:
    return f"{raw_data_df.shape[0]} rows x {raw_data_df.shape[1]} columns"


def _make_eval_semantic_chunk_runner(worker_model_id: str, chunk_timeout_seconds: float):
    """Create a non-Prefect semantic chunk runner for Inspect evals."""

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

        async with make_eval_session_factory("target-2", worker_model_id) as factory:

            async def _run_chunk(idx: int) -> tuple[int, dict]:
                async with semaphore:
                    try:
                        result = await asyncio.wait_for(
                            run_worker_extraction(
                                window_text=chunk_texts[idx],
                                window_starts=chunk_window_starts[idx],
                                question=question,
                                causal_design=chunk_contexts[idx],
                                session_factory=factory,
                                logger=LOGGER,
                                call_label=f"eval extraction chunk={idx}",
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
    extraction_workers = config.extraction_workers

    async with make_eval_session_factory("target-1a", model_id) as factory_1a:
        latent_result = await run_latent_structure(
            question=fixture.question,
            session_factory=factory_1a,
        )

    dataset_schema = format_schema_for_llm(fixture.raw_data_df, fixture.column_descriptions)
    async with make_eval_session_factory("target-1b", model_id) as factory_1b:
        measurement_result = await run_measurement_structure(
            question=fixture.question,
            latent_structure=latent_result.latent_structure,
            chunks=[dataset_schema],
            session_factory=factory_1b,
            dataset_summary=_dataset_summary(fixture.raw_data_df),
        )

    causal_design = build_causal_design(
        latent_result.latent_structure,
        measurement_result.measurement_structure,
    )

    extraction_result = await run_extraction_core(
        raw_df=fixture.raw_data_df,
        question=fixture.question,
        measurement_structure=measurement_result.measurement_structure,
        extraction_workers=extraction_workers,
        semantic_chunk_runner=_make_eval_semantic_chunk_runner(
            worker_model_id,
            chunk_timeout_seconds,
        ),
    )
    materialized = materialize_extraction_outputs(
        extraction_result,
        measurement_result.measurement_structure,
    )

    data_for_model = materialized["data_for_model"]
    comparison = compare_demo_health_outputs(
        causal_design=causal_design,
        raw_data_df=fixture.raw_data_df,
        data_for_model=data_for_model,
        expected_model=fixture.expected_model,
    )

    return CandidateRun(
        model_id=model_id,
        latent_result=latent_result,
        measurement_result=measurement_result,
        comparison=comparison,
        row_count=data_for_model.height,
    )


def _format_candidate_report(candidate: CandidateRun) -> str:
    outcome_name = get_outcome_name(candidate.latent_result.latent_structure) or "[missing outcome]"
    indicators = ", ".join(candidate.comparison.measurement_indicators)
    return "\n".join(
        [
            f"- latent_structure constructs: {candidate.latent_result.n_constructs}",
            f"- latent_structure edges: {candidate.latent_result.n_edges}",
            f"- latent_structure outcome: {outcome_name}",
            f"- measurement_structure indicators ({len(candidate.comparison.measurement_indicators)}): {indicators}",
            f"- extraction rows: {candidate.row_count}",
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
        worker_model = get_config().extraction_workers.model
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
    """Judge-rank orchestrator models on DEMO fixture reproduction."""
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
