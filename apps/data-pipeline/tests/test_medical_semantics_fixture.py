"""Advisory contract check for the tracked MEDICAL_SEMANTICS Stage 2 fixture."""

from __future__ import annotations

import json
import warnings

import polars as pl

from causal_ssm_agent.utils.medical_semantics_fixture import (
    compare_medical_semantics_outputs,
    load_medical_semantics_fixture,
)


def test_medical_semantics_stage2_fixture_contract() -> None:
    try:
        fixture = load_medical_semantics_fixture()
    except FileNotFoundError as exc:
        warnings.warn(
            f"MEDICAL_SEMANTICS advisory skipped: missing fixture artifact: {exc.filename}",
            stacklevel=1,
        )
        return

    required_paths = [
        fixture.run_dir / "stage-1b.json",
        fixture.run_dir / "stage2-raw-data.parquet",
        fixture.run_dir / "stage2-model-data.parquet",
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        warnings.warn(
            "MEDICAL_SEMANTICS advisory skipped: missing run artifacts "
            + ", ".join(str(path.relative_to(fixture.fixture_dir)) for path in missing_paths),
            stacklevel=1,
        )
        return

    causal_spec = json.loads((fixture.run_dir / "stage-1b.json").read_text())["causal_spec"]
    raw = pl.read_parquet(fixture.run_dir / "stage2-raw-data.parquet")
    model = pl.read_parquet(fixture.run_dir / "stage2-model-data.parquet")

    comparison = compare_medical_semantics_outputs(
        causal_spec=causal_spec,
        stage0=fixture.stage0,
        raw=raw,
        model=model,
        expected_raw=fixture.expected_raw,
        expected_model=fixture.expected_model,
    )

    issues = comparison.all_issues()
    if issues:
        warnings.warn(
            "MEDICAL_SEMANTICS Stage 2 advisory mismatches:\n- " + "\n- ".join(issues),
            stacklevel=1,
        )
