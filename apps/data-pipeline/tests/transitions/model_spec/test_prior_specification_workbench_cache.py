from __future__ import annotations

import polars as pl
from notebooks import prior_specification_support as support

from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
from nof1_causal_lab.models.ssm.construct_admission import AdmissionReport
from nof1_causal_lab.models.ssm.reachability import CheckResult
from tests.helpers import make_structural_plan


class _AdmittedState:
    def __init__(self, current_construct: str | None, submit_calls: list[str]) -> None:
        self.current_construct = current_construct
        self.submit_calls = submit_calls
        self.admitted_contributions = {}
        self.last_report = None
        self.last_coupled_results = ()
        self.n_draws = 200
        self.seed = 0
        self.attempt = 0
        self.submission_made = False

    def submit_construct(self, *, construct: str, **_kwargs) -> str:
        self.submit_calls.append(construct)
        self.current_construct = None
        self.last_report = AdmissionReport(
            name=construct,
            results=(
                CheckResult(
                    check="C2 latent scale",
                    target=construct,
                    value="median scale 1.0",
                    band="[0.33, 3.0]",
                    passed=True,
                    note="ok",
                ),
            ),
            timings=(),
            outcome="ADMITTED",
            annotations=(),
            admitted=True,
        )
        return "accepted"


def test_workbench_replay_reuses_cached_evaluation_and_invalidates_semantic_inputs(
    monkeypatch,
    tmp_path,
):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    submit_calls: list[str] = []
    monkeypatch.setattr(support, "build_construct_order", lambda _plan: ["sleep"])
    monkeypatch.setattr(
        support,
        "restore_construct_state",
        lambda _checkpoint, *, target_construct, **_kwargs: _AdmittedState(
            target_construct,
            submit_calls,
        ),
    )
    proposal = {
        "construct": "sleep",
        "indicators": [],
        "priors": {
            "rho_sleep": {
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
            }
        },
    }
    plan = StructuralPlan.model_validate(make_structural_plan(["sleep"], []))
    panel = pl.DataFrame({"value": [1.0]})

    first = support.run_authored_proposals(
        cache_workspace_id="workbench-test",
        structural_plan=plan,
        data_for_model=panel,
        proposals=[proposal],
        n_draws=16,
        seed=7,
    )
    second = support.run_authored_proposals(
        cache_workspace_id="workbench-test",
        structural_plan=plan,
        data_for_model=panel,
        proposals=[proposal],
        n_draws=16,
        seed=7,
    )

    assert submit_calls == ["sleep"]
    assert first.attempts[0].cache_hit is False
    assert second.attempts[0].cache_hit is True
    assert second.attempts[0].report is not None
    assert second.complete

    changed_proposal = {
        **proposal,
        "priors": {
            "rho_sleep": {
                "distribution": "Beta",
                "params": {"alpha": 5.0, "beta": 2.0},
            }
        },
    }
    support.run_authored_proposals(
        cache_workspace_id="workbench-test",
        structural_plan=plan,
        data_for_model=panel,
        proposals=[changed_proposal],
        n_draws=16,
        seed=7,
    )
    support.run_authored_proposals(
        cache_workspace_id="workbench-test",
        structural_plan=plan,
        data_for_model=pl.DataFrame({"value": [2.0]}),
        proposals=[proposal],
        n_draws=16,
        seed=7,
    )

    assert submit_calls == ["sleep", "sleep", "sleep"]
