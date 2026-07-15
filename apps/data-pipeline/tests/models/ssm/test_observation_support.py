from dataclasses import replace

import polars as pl
import pytest

from nof1_causal_lab.models.ssm.observation_support import hydrate_discrete_manifest_metadata
from tests.models.ssm._support import complex_mixed_runtime_spec


def _single_row_panel(**overrides: float) -> pl.DataFrame:
    values = {
        "stress_cont": 0.0,
        "adherence_flag": 1.0,
        "steps_count": 10.0,
        "fatigue_t": 0.0,
        "screen_gap": 1.0,
        "sleep_efficiency": 0.8,
        "symptom_severity": 0.0,
        "coping_style": 0.0,
        "rumination_count": 1.0,
        "focus_cont": 0.0,
    }
    values.update(overrides)
    return pl.DataFrame(values)


def test_declared_discrete_levels_allow_one_observed_level():
    spec = complex_mixed_runtime_spec()

    hydrated = hydrate_discrete_manifest_metadata(spec, _single_row_panel())

    assert hydrated.manifest_level_counts == [0, 0, 0, 0, 0, 0, 4, 4, 0, 0]


def test_declared_discrete_levels_reject_out_of_range_code():
    spec = complex_mixed_runtime_spec()

    with pytest.raises(ValueError, match=r"outside declared range 0\.\.3"):
        hydrate_discrete_manifest_metadata(spec, _single_row_panel(symptom_severity=4.0))


def test_undeclared_discrete_levels_still_require_observed_support():
    spec = replace(complex_mixed_runtime_spec(), manifest_level_counts=None)

    with pytest.raises(ValueError, match=r"only 1 level\(s\) are present"):
        hydrate_discrete_manifest_metadata(spec, _single_row_panel())
