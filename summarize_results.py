"""Summarize POEM results with model comparison.

Reads settings and metrics recursively from results/.

Generates:
  results/comparison.csv     side-by-side comparison (oldest on left, newest on right)
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")
DATASETS = [
    ("ETTh1", "ETTh1"),
    ("ETTh2", "ETTh2"),
    ("ETTm1", "ETTm1"),
    ("ETTm2", "ETTm2"),
    ("weather", "weather"),
    ("electricity", "Electricity"),
    ("traffic", "traffic"),
]
PRED_LENS = [96, 192, 336, 720]
MODEL_RUNS = {
    "POEM": (
        "POEM",
        "POEM",
        "poem",
    ),
    "PhaseFormer": (
        "PhaseFormer",
        "PhaseFormer",
        "phaseformer_original",
    ),
    "TimeBase": (
        "TimeBase",
        "TimeBase",
        "timebase_original",
    ),
}

ROW_SPECS = [
    (dataset, pred_len, f"{display_name} {pred_len}")
    for dataset, display_name in DATASETS
    for pred_len in PRED_LENS
]


def _infer_pred_len(pred_dir_name):
    match = re.match(r"pred_(\d+)$", pred_dir_name)
    return int(match.group(1)) if match else None


def collect_results(output_family, model_id=None, run_tag=None, seed=2021):
    """Collect one result per dataset and horizon for a given model/run_tag."""
    selected = {}
    if not RESULTS_DIR.is_dir():
        return selected

    for metrics_path in sorted(RESULTS_DIR.rglob("metrics.npy")):
        result_dir = metrics_path.parent
        settings_path = result_dir / "settings.json"
        if not settings_path.is_file():
            raise RuntimeError(f"Missing settings.json for {metrics_path}")
        settings = json.loads(settings_path.read_text())
        stored_family = metrics_path.relative_to(RESULTS_DIR).parts[0]
        if stored_family != output_family:
            continue
        if model_id is not None and settings.get("model_id") != model_id:
            continue
        if run_tag is not None:
            accepted_tags = run_tag if isinstance(run_tag, (tuple, list, set)) else (run_tag,)
            if settings.get("run_tag") not in accepted_tags:
                continue
        if seed is not None and settings.get("seed") != seed:
            continue

        dataset = settings["dataset"]
        pred_len = settings["pred_len"]

        key = (dataset, pred_len)
        if key in selected:
            previous_path = selected[key]["PATH"]
            raise RuntimeError(
                "Multiple configurations match "
                f"output_family={output_family!r}, model_id={model_id!r}, "
                f"run_tag={run_tag!r}, seed={seed!r}, "
                f"dataset={dataset!r}, pred_len={pred_len}: "
                f"{previous_path} and {metrics_path}"
            )
        metrics = np.load(metrics_path)  # [mae, mse, rmse, mape, mspe, rse, corr]
        selected[key] = {
            "MAE": float(metrics[0]),
            "MSE": float(metrics[1]),
            "PATH": str(metrics_path),
        }
    return selected


def write_comparison_csv(all_results, output_path):
    """Write comparison CSV with two header rows.

    Row 1: model names (each spanning MSE+MAE)
    Row 2: Dataset, Pred Len, MSE, MAE, MSE, MAE, ...
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = list(all_results.keys())

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)

        # Header row 1: model names, each spanning 2 columns
        row1 = ["", ""]
        for m in models:
            row1 += [m, ""]
        writer.writerow(row1)

        # Header row 2: Dataset, Pred Len, MSE, MAE per model
        row2 = ["Dataset", "Pred Len"]
        for _ in models:
            row2 += ["MSE", "MAE"]
        writer.writerow(row2)

        # Data rows
        for dataset, pred_len, _label in ROW_SPECS:
            row = [dataset, pred_len]
            for model_name in models:
                entry = all_results.get(model_name, {}).get((dataset, pred_len), {})
                row.append(f"{entry['MSE']:.6f}" if entry else "")
                row.append(f"{entry['MAE']:.6f}" if entry else "")
            writer.writerow(row)

    print(f"  -> {output_path}")


def main():
    all_results = {}
    for model_name, (output_family, model_id, run_tag) in MODEL_RUNS.items():
        results = collect_results(
            output_family=output_family, model_id=model_id, run_tag=run_tag
        )
        all_results[model_name] = results
        tag_label = run_tag if run_tag is not None else "all retained runs"
        print(f"  [{model_name}] {len(results)} entries  ({tag_label})")

    total = sum(len(results) for results in all_results.values())
    if total == 0:
        print("No experiment results found under results/")
        return

    write_comparison_csv(all_results, RESULTS_DIR / "comparison.csv")
    print("Done.")


if __name__ == "__main__":
    main()
