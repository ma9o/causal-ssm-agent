"""Tests for the executable architecture-boundary checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_checker() -> Any:
    module_name = "check_architecture_boundaries_under_test"
    path = Path(__file__).resolve().parents[2] / "scripts" / "check_architecture_boundaries.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_module(source_root: Path, relative_path: str, source: str) -> None:
    path = source_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_promoted_boundaries_reject_upward_runtime_imports(tmp_path: Path) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    _write_module(
        source_root,
        "models/ssm/compile/example.py",
        """
from nof1_causal_lab.models.structural import build_structural_plan
from nof1_causal_lab.models.ssm.runtime import hydrate_compiled_model
from nof1_causal_lab.workers.schemas_prior import PriorProposal
""",
    )
    _write_module(
        source_root,
        "models/ssm/model.py",
        "from nof1_causal_lab.models.ssm.inference import fit\n",
    )
    _write_module(
        source_root,
        "models/ssm/runtime.py",
        "from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact\n",
    )
    _write_module(
        source_root,
        "models/ssm/inference/example.py",
        "from nof1_causal_lab.models.ssm.runtime import hydrate_compiled_model\n",
    )

    violations = checker.find_violations(source_root)

    assert {violation.code for violation in violations} == {
        "ARCH001",
        "ARCH002",
        "ARCH003",
        "ARCH004",
        "ARCH005",
        "ARCH006",
    }


def test_type_only_dependencies_do_not_initialize_forbidden_layers(tmp_path: Path) -> None:
    checker = _load_checker()
    source_root = tmp_path / "nof1_causal_lab"
    _write_module(
        source_root,
        "models/ssm/runtime.py",
        """
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.compile.contracts import CompiledSSMArtifact
""",
    )
    _write_module(
        source_root,
        "models/ssm/model.py",
        """
import typing

if typing.TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference.types import ParticleMCMCPosterior
""",
    )

    assert checker.find_violations(source_root) == ()
