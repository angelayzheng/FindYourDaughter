"""Reproducible parameter search with immutable trial directories and a fixed rank."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path

from backend.configuration import configuration
from evaluation.draft_set import SCOPE, _sha256, _write_json, evaluate_dataset
from scripts.evaluate_synthetic_detection import evaluate as evaluate_synthetic


RANKING = "draft F1, then recall, then fewer changed settings, then configuration ID"
GUARD = "development F1 and recall must not fall below the detector default"


def trial_id(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return config["detector"] + "-" + hashlib.sha256(encoded).hexdigest()[:12]


def changed_parameters(config: dict) -> dict:
    defaults = configuration(config["detector"])["parameters"]
    return {f"{group}.{key}": value for group, settings in config["parameters"].items()
            for key, value in settings.items() if value != defaults[group][key]}


def expand_plan(plan: dict) -> list[dict]:
    if not isinstance(plan, dict) or set(plan) != {"matching_tolerance_mm", "shortlist_per_detector", "ranking", "synthetic_guard", "families"}:
        raise ValueError("Unexpected tuning plan fields")
    if plan["ranking"] != RANKING or plan["synthetic_guard"] != GUARD:
        raise ValueError("This runner supports the documented F1 ranking and synthetic guard")
    if type(plan["shortlist_per_detector"]) is not int or plan["shortlist_per_detector"] < 1:
        raise ValueError("shortlist_per_detector must be a positive integer")
    tolerance = plan["matching_tolerance_mm"]
    if type(tolerance) not in (float, int) or not 0 < tolerance < float("inf"):
        raise ValueError("Matching tolerance must be finite and positive")
    if not isinstance(plan["families"], list) or not plan["families"]:
        raise ValueError("Provide at least one parameter family")
    configs = {}
    for family in plan["families"]:
        if not isinstance(family, dict) or set(family) != {"detector", "grid"}:
            raise ValueError("Each family requires detector and grid")
        detector, grid = family["detector"], family["grid"]
        default = configuration(detector)
        configs[trial_id(default)] = default
        if not isinstance(grid, dict) or any(not isinstance(v, list) or not v for v in grid.values()):
            raise ValueError("Each grid dimension must be a nonempty list")
        keys = sorted(grid)
        count = 1
        for key in keys:
            if not isinstance(key, str) or len(key.split(".")) != 2:
                raise ValueError("Use group.parameter grid keys")
            count *= len(grid[key])
        if count > 10000:
            raise ValueError("A grid family may contain at most 10000 combinations")
        for values in product(*(grid[key] for key in keys)):
            parameters = {}
            for key, value in zip(keys, values):
                group, name = key.split(".")
                parameters.setdefault(group, {})[name] = value
            config = configuration(detector, parameters)
            configs[trial_id(config)] = config
    return sorted(configs.values(), key=lambda c: (c["detector"], bool(changed_parameters(c)), trial_id(c)))


def rank_key(trial: dict) -> tuple:
    summary = trial["summary"]
    return (-(summary["reference_f1"] if summary["reference_f1"] is not None else -1),
            -(summary["reference_recall"] if summary["reference_recall"] is not None else -1),
            len(trial["changed_parameters"]), trial["trial_id"])


def ranked_trials(trials: list[dict], detector: str) -> list[dict]:
    return sorted((t for t in trials if t["detector"] == detector and t["status"] == "complete"
                   and t["summary"]["failed_cases"] == 0), key=rank_key)


def synthetic_metrics(report: dict) -> dict:
    tp, fp, fn = (report[k] for k in ("true_positive", "false_positive", "false_negative"))
    return {"matched": tp, "unmatched_predictions": fp, "unmatched_references": fn,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "recall": tp / (tp + fn) if tp + fn else None, "mean_errors": report["mean_errors"]}


def _synthetic_inventory(dataset: Path) -> dict:
    result = {}
    for truth in sorted(dataset.glob("subject*/truth*.json")):
        suffix = truth.stem.removeprefix("truth")
        for path in (truth, truth.with_name(f"orig{suffix}.nii"), truth.with_name(f"mask{suffix}.nii")):
            result[path.relative_to(dataset).as_posix()] = _sha256(path)
    if not result:
        raise ValueError("Synthetic dataset contains no truth cases")
    return result


def _save_index(output: Path, index: dict) -> None:
    _write_json(output / "index.json", index)
    lines = ["# Parameter tuning record", "", SCOPE, "", f"Status: {index['status']}",
             f"Ranking: {RANKING}.", "", "Defaults are retained; winners are optional configurations.", "",
             "| Trial | Detector | Changed parameters | Matches | Extras | Precision | Recall | F1 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for trial in index["trials"]:
        if trial["status"] != "complete":
            lines.append(f"| {trial['trial_id']} | {trial['detector']} | ERROR | | | | | |")
            continue
        s = trial["summary"]
        values = [f"{100*s[k]:.1f}%" if s[k] is not None else "n/a"
                  for k in ("reference_precision", "reference_recall", "reference_f1")]
        changes = "; ".join(f"{k}={v}" for k, v in trial["changed_parameters"].items()) or "defaults"
        lines.append(f"| {trial['trial_id']} | {trial['detector']} | {changes} | {s['matched']}/{s['reference_count']} | "
                     f"{s['unmatched_predictions']} | {' | '.join(values)} |")
    if index.get("selection"):
        lines += ["", "## Selected configurations", "",
                  "Selection uses draft F1 among the synthetic-checked shortlist; this is not a global optimum.", ""]
        for name, selection in index["selection"].items():
            lines.append(f"- {name}: draft leader `{selection['best_draft_trial']}`; "
                         f"selected `{selection['selected_trial']}` after the development guard.")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(plan: dict, dataset: Path, output: Path, *, synthetic_development: Path) -> dict:
    configs = expand_plan(plan)
    dataset, output, synthetic_development = (Path(p).resolve() for p in (dataset, output, synthetic_development))
    if output.is_relative_to(dataset) or output.is_relative_to(synthetic_development):
        raise ValueError("Tuning output must be outside the input datasets")
    inventory = _synthetic_inventory(synthetic_development)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "plan.json", plan)
    _write_json(output / "configurations.json", configs)
    index = {"scope": SCOPE, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
             "dataset": str(dataset), "synthetic_development": str(synthetic_development),
             "synthetic_input_sha256": inventory, "plan": plan, "trials": [], "selection": {},
             "note": "Best means best observed in this search. Real draft cases are development data."}
    _save_index(output, index)
    for number, config in enumerate(configs, 1):
        identifier = trial_id(config)
        folder = output / "trials" / identifier
        folder.mkdir(parents=True, exist_ok=False)
        _write_json(folder / "config.json", config)
        trial = {"trial_id": identifier, "detector": config["detector"], "configuration": config,
                 "changed_parameters": changed_parameters(config), "status": "running"}
        try:
            report = evaluate_dataset(dataset, folder, detectors=(config["detector"],),
                                      tolerance_mm=plan["matching_tolerance_mm"],
                                      parameters={config["detector"]: config["parameters"]})
            hashes = {k: v for case in report["cases"] for k, v in case.get("input_sha256", {}).items()}
            if "draft_input_sha256" not in index:
                index["draft_input_sha256"] = hashes
                index["backend_sha256"] = report["backend_sha256"]
            elif hashes != index["draft_input_sha256"] or report["backend_sha256"] != index["backend_sha256"]:
                raise ValueError("Inputs or detector implementation changed during the search")
            trial.update(status=report["status"], summary=report["summary"][config["detector"]],
                         cases=[{"case_id": c["case_id"], "status": c["status"],
                                 "summary": c["detectors"].get(config["detector"], {}).get("scores", {}).get("summary")}
                                for c in report["cases"]])
        except (ValueError, OSError) as error:
            trial.update(status="error", error=str(error))
        index["trials"].append(trial)
        _save_index(output, index)
        s = trial.get("summary", {})
        print(f"[{number}/{len(configs)}] {identifier}: {s.get('matched')}/{s.get('reference_count')} matches, "
              f"{s.get('unmatched_predictions')} extras; {trial['status']}", flush=True)
    # Only the predeclared shortlist and each default see synthetic development
    # data. A separate validation command evaluates frozen winners on fresh data.
    for name in sorted({c["detector"] for c in configs}):
        ranking = ranked_trials(index["trials"], name)
        default_id = trial_id(configuration(name))
        default = next((t for t in ranking if t["trial_id"] == default_id), None)
        if default is None:
            continue
        shortlist = ranking[:plan["shortlist_per_detector"]]
        if default not in shortlist:
            shortlist.append(default)
        for trial in shortlist:
            cfg = trial["configuration"]
            try:
                report = evaluate_synthetic(synthetic_development, plan["matching_tolerance_mm"],
                                            detector=name, parameters=cfg["parameters"])
                _write_json(output / "trials" / trial["trial_id"] / "synthetic_development.json", report)
                trial["synthetic_development"] = synthetic_metrics(report)
            except Exception as error:
                trial["synthetic_error"] = f"{type(error).__name__}: {error}"
            _save_index(output, index)
            print(f"Synthetic development: {trial['trial_id']}: {trial.get('synthetic_development', trial.get('synthetic_error'))}", flush=True)
        baseline = default.get("synthetic_development")
        if baseline is None or baseline["f1"] is None or baseline["recall"] is None:
            continue
        eligible = [t for t in shortlist if t.get("synthetic_development", {}).get("f1") is not None
                    and t["synthetic_development"]["f1"] >= baseline["f1"]
                    and t["synthetic_development"]["recall"] >= baseline["recall"]]
        selected = min(eligible, key=rank_key)
        index["selection"][name] = {"best_draft_trial": ranking[0]["trial_id"],
                                    "selected_trial": selected["trial_id"],
                                    "checked_trials": [t["trial_id"] for t in shortlist],
                                    "eligible_trials": [t["trial_id"] for t in eligible]}
        _write_json(output / "selected" / f"{name}.json", selected["configuration"])
    index["status"] = "complete" if (all(t["status"] == "complete" and "synthetic_error" not in t for t in index["trials"])
                                          and len(index["selection"]) == len({c["detector"] for c in configs})) else "incomplete"
    _save_index(output, index)
    return index


def validate_selection(run: Path, dataset: Path) -> dict:
    """Evaluate frozen defaults/winners once; never rank or select using holdout."""
    run, dataset = Path(run).resolve(), Path(dataset).resolve()
    index = json.loads((run / "index.json").read_text(encoding="utf-8"))
    if index["status"] != "complete":
        raise ValueError("Only a complete search with frozen selections can be validated")
    inventory = _synthetic_inventory(dataset)
    evidence = lambda files: {v for k, v in files.items() if Path(k).name.startswith(("orig", "truth"))}
    if evidence(inventory) & evidence(index["synthetic_input_sha256"]):
        raise ValueError("Holdout files overlap the synthetic development inputs")
    output = run / "holdout"
    if output.is_relative_to(dataset):
        raise ValueError("Validation output must be outside the dataset")
    backend = Path(__file__).resolve().parents[1] / "backend"
    for selected in index["selection"].values():
        report = json.loads((run / "trials" / selected["selected_trial"] / "report.json").read_text())
        if any(_sha256(backend / name) != digest for name, digest in report["backend_sha256"].items()):
            raise ValueError("Detector code changed after selection; start a new experiment")
    output.mkdir(exist_ok=False)
    result = {"scope": "Fresh synthetic validation of frozen defaults/winners; not clinical accuracy.",
              "selection_index_sha256": _sha256(run / "index.json"), "dataset": str(dataset),
              "input_sha256": inventory, "status": "running", "results": {}}
    _write_json(output / "validation.json", result)
    for name, selection in index["selection"].items():
        default = configuration(name)
        selected = next(t["configuration"] for t in index["trials"] if t["trial_id"] == selection["selected_trial"])
        for config in {trial_id(default): default, trial_id(selected): selected}.values():
            identifier = trial_id(config)
            try:
                report = evaluate_synthetic(dataset, index["plan"]["matching_tolerance_mm"],
                                            detector=name, parameters=config["parameters"])
                _write_json(output / f"{identifier}.json", report)
                result["results"][identifier] = synthetic_metrics(report)
            except Exception as error:
                result["results"][identifier] = {"error": f"{type(error).__name__}: {error}"}
            _write_json(output / "validation.json", result)
            print(f"Holdout: {identifier}: {result['results'][identifier]}", flush=True)
    result["status"] = "complete" if all("error" not in r for r in result["results"].values()) else "incomplete"
    _write_json(output / "validation.json", result)
    return result
