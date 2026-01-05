"""Run configuration.

A single frozen ``Config`` is threaded through the harness so that everything a
run depends on -- models, sampling budget, schema set, seed -- is recorded in
one place and can be stamped into the run summary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


@dataclass(frozen=True)
class ModelConfig:
    agent: str = "claude-sonnet-5"
    user_sim: str = "claude-sonnet-5"
    judge: str = "claude-opus-5"
    optimizer: str = "claude-opus-5"
    max_tokens: int = 4096
    effort: str = "medium"


@dataclass(frozen=True)
class EvalConfig:
    k: int = 8
    max_turns: int = 24
    max_tool_calls: int = 32
    user_mode: str = "adversarial"
    bootstrap_resamples: int = 10000
    ci_level: float = 0.95


@dataclass(frozen=True)
class GateConfig:
    # A revision must lift its target tasks by at least this much, in absolute
    # per-task success rate, before the collateral check is even considered.
    min_target_delta: float = 0.05
    # Collateral regression is judged on the non-target tasks: the revision is
    # rejected if the lower bound of the paired CI on that subset falls below
    # this threshold.
    max_collateral_regression: float = -0.03
    block_permission_expansion: bool = True


@dataclass(frozen=True)
class Config:
    name: str
    schema_set: str
    models: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    seed: int = 20260105
    tasks_path: str = "data/tasks/tasks.jsonl"
    domain_path: str = "data/domain"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable short hash of the configuration, stamped into run summaries."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def resolve(self, relative: str) -> Path:
        return REPO_ROOT / relative


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    models = ModelConfig(**raw.pop("models", {}))
    evaluation = EvalConfig(**raw.pop("eval", {}))
    gate = GateConfig(**raw.pop("gate", {}))
    return Config(models=models, eval=evaluation, gate=gate, **raw)
