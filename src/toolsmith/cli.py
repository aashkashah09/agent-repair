"""Command line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classifier.calibration import evaluate as evaluate_calibration
from .classifier.calibration import load_labelled
from .classifier.judge import class_distribution
from .config import REPO_ROOT, load_config
from .eval.bootstrap import compare as compare_outcomes
from .eval.harness import Harness, load_outcomes, load_runs
from .eval.metrics import summarise
from .optimizer.gate import tally
from .optimizer.loop import RepairLoop
from .server.mcp_server import serve_stdio
from .server.schemas import SchemaSet
from .server.seeding import apply_defects, load_defects

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_DEFECTS = "data/defects/seeded_defects.json"


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def cmd_seed(args: argparse.Namespace) -> None:
    clean = SchemaSet.load(args.clean)
    defects = load_defects(args.defects)
    seeded = apply_defects(clean, defects)
    seeded.save(args.out)
    print(f"seeded {len(defects)} defects into {args.out}")


def cmd_eval(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    harness = Harness(
        config,
        Path(args.schemas),
        Path(args.defects) if args.defects else None,
    )
    summary = harness.run_suite(args.out, run_name=args.name)
    print(json.dumps(summary, indent=2))


def cmd_summarise(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    outcomes = load_outcomes(Path(args.run) / "runs.jsonl")
    print(json.dumps(summarise(outcomes, config.eval.k), indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    before = load_outcomes(Path(args.before) / "runs.jsonl")
    after = load_outcomes(Path(args.after) / "runs.jsonl")
    result = compare_outcomes(
        before,
        after,
        resamples=config.eval.bootstrap_resamples,
        ci_level=config.eval.ci_level,
        seed=config.seed,
    )
    payload = {
        "before": Path(args.before).name,
        "after": Path(args.after).name,
        **result.to_dict(),
    }
    if args.out:
        _write(Path(args.out), payload)
    else:
        print(json.dumps(payload, indent=2))


def cmd_classify(args: argparse.Namespace) -> None:
    verdicts = [
        json.loads(line)
        for line in Path(args.verdicts).read_text().splitlines()
        if line.strip()
    ]
    payload = class_distribution(verdicts)
    if args.out:
        _write(Path(args.out), payload)
    else:
        print(json.dumps(payload, indent=2))


def cmd_calibrate(args: argparse.Namespace) -> None:
    calibration = evaluate_calibration(load_labelled(args.labels))
    if args.out:
        _write(Path(args.out), calibration.to_dict())
    else:
        print(json.dumps(calibration.to_dict(), indent=2))


def cmd_gate_report(args: argparse.Namespace) -> None:
    decisions = [
        json.loads(line)
        for line in Path(args.decisions).read_text().splitlines()
        if line.strip()
    ]
    payload = tally(decisions)
    if args.out:
        _write(Path(args.out), payload)
    else:
        print(json.dumps(payload, indent=2))


def cmd_repair(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    loop = RepairLoop(
        config,
        Path(args.defects) if args.defects else None,
        Path(args.results),
    )
    schema_dir = Path(args.schemas)
    for round_index in range(1, args.rounds + 1):
        out_dir = REPO_ROOT / "data" / "schemas" / f"round{round_index}"
        result = loop.run_round(round_index, schema_dir, out_dir)
        _write(Path(args.results) / f"round{round_index}" / "round.json", result.to_dict())
        schema_dir = out_dir


def cmd_serve(args: argparse.Namespace) -> None:
    serve_stdio(args.schemas, args.domain, args.defects)


def cmd_validate(args: argparse.Namespace) -> None:
    from .eval.tasks import load_tasks

    for directory in sorted((REPO_ROOT / "data" / "schemas").iterdir()):
        if not directory.is_dir():
            continue
        SchemaSet.load(directory).validate()
        print(f"ok  {directory.relative_to(REPO_ROOT)}")
    tasks = load_tasks(REPO_ROOT / "data" / "tasks" / "tasks.jsonl")
    print(f"ok  data/tasks/tasks.jsonl ({len(tasks)} tasks)")
    defects = load_defects(REPO_ROOT / DEFAULT_DEFECTS)
    print(f"ok  {DEFAULT_DEFECTS} ({len(defects)} defects)")
    for run_dir in sorted((REPO_ROOT / "results").glob("*/runs.jsonl")):
        rows = load_runs(run_dir)
        print(f"ok  {run_dir.relative_to(REPO_ROOT)} ({len(rows)} runs)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toolsmith")
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="apply the defect catalogue to the clean schema set")
    seed.add_argument("--clean", default="data/schemas/clean")
    seed.add_argument("--defects", default=DEFAULT_DEFECTS)
    seed.add_argument("--out", default="data/schemas/seeded")
    seed.set_defaults(func=cmd_seed)

    ev = sub.add_parser("eval", help="run the task suite against a schema set")
    ev.add_argument("--config", default=DEFAULT_CONFIG)
    ev.add_argument("--schemas", required=True)
    ev.add_argument("--defects", default=DEFAULT_DEFECTS)
    ev.add_argument("--out", required=True)
    ev.add_argument("--name", required=True)
    ev.set_defaults(func=cmd_eval)

    summary = sub.add_parser("summarise", help="recompute a run's summary from its rows")
    summary.add_argument("--config", default=DEFAULT_CONFIG)
    summary.add_argument("--run", required=True)
    summary.set_defaults(func=cmd_summarise)

    comparison = sub.add_parser("compare", help="paired bootstrap between two runs")
    comparison.add_argument("--config", default=DEFAULT_CONFIG)
    comparison.add_argument("--before", required=True)
    comparison.add_argument("--after", required=True)
    comparison.add_argument("--out")
    comparison.set_defaults(func=cmd_compare)

    classify = sub.add_parser("classify", help="summarise a round's judge verdicts")
    classify.add_argument("--verdicts", required=True)
    classify.add_argument("--out")
    classify.set_defaults(func=cmd_classify)

    calibrate = sub.add_parser("calibrate", help="score the judge against hand labels")
    calibrate.add_argument("--labels", default="data/calibration/judge_calibration.jsonl")
    calibrate.add_argument("--out")
    calibrate.set_defaults(func=cmd_calibrate)

    gate = sub.add_parser("gate-report", help="tally gate decisions")
    gate.add_argument("--decisions", default="results/gate/decisions.jsonl")
    gate.add_argument("--out")
    gate.set_defaults(func=cmd_gate_report)

    repair = sub.add_parser("repair", help="run the repair loop")
    repair.add_argument("--config", default=DEFAULT_CONFIG)
    repair.add_argument("--schemas", default="data/schemas/seeded")
    repair.add_argument("--defects", default=DEFAULT_DEFECTS)
    repair.add_argument("--results", default="results")
    repair.add_argument("--rounds", type=int, default=4)
    repair.set_defaults(func=cmd_repair)

    serve = sub.add_parser("serve", help="expose a schema set over MCP on stdio")
    serve.add_argument("--schemas", default="data/schemas/clean")
    serve.add_argument("--domain", default="data/domain")
    serve.add_argument("--defects")
    serve.set_defaults(func=cmd_serve)

    validate = sub.add_parser("validate", help="check every committed artefact loads")
    validate.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
