"""Summarize POEM ablation results.

Part 1 — Full ablation (8 variants × 7 datasets):
  visualization/data/ablation_detail.csv   – per dataset × pred_len
  visualization/data/ablation_average.csv  – per dataset (averaged across pred_lens)

Part 2 — Geometry-focused ablation (4 variants × 4 datasets, with Δ% vs POEM):
  results/geometry_ablation_detail.csv     – per dataset × pred_len
  results/geometry_ablation_average.csv    – averaged across pred_lens
  results/geometry_ablation_all_average.csv – long-format, averaged across pred_lens
"""

import csv
import json
from pathlib import Path

from summarize_results import collect_results

SEED = 2021
PRED_LENS = (96, 192, 336, 720)

# ============================================================================
# Part 1 — Full ablation (8 variants, all 7 datasets)
# ============================================================================

ALL_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic")

ALL_VARIANTS = {
    "POEM":                  ("POEM",     "POEM",                  "poem"),
    "NoPhaseInteraction":    ("ablation", "NoPhaseInteraction",    "nophaseinteraction"),
    "NoHarmonicModulation":  ("ablation", "NoHarmonicModulation",  "noharmonicmodulation"),
    "LinearPhaseBackbone":   ("ablation", "LinearPhaseBackbone",   "linearphasebackbone"),
    "VanillaMLPMixer":       ("ablation", "VanillaMLPMixer",       "vanillamlpmixer"),
    "NoGlobalForecast":      ("ablation", "NoGlobalForecast",      "noglobalforecast"),
    "NoGeometry":            ("ablation", "NoGeometry",            "nogeometry"),
    "LearnableGeometry":     ("ablation", "LearnableGeometry",     "learnablegeometry"),
}

FULL_ORDER = list(ALL_VARIANTS.keys())


def run_full_ablation():
    """8-variant ablation across all 7 datasets."""
    Path("visualization/data").mkdir(parents=True, exist_ok=True)

    all_results = {}
    for name, (family, model_id, tag) in ALL_VARIANTS.items():
        results = collect_results(family, model_id, tag, seed=SEED)
        all_results[name] = results
        print(f"  [full] {name}: {len(results)} entries")

    # -- detail CSV ----------------------------------------------------------
    detail_path = Path("visualization/data/ablation_detail.csv")
    with detail_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["Dataset", "Pred Len"]
        for v in FULL_ORDER:
            header += [f"{v} MSE", f"{v} MAE"]
        w.writerow(header)

        for dataset in ALL_DATASETS:
            for pred_len in PRED_LENS:
                row = [dataset, pred_len]
                key = (dataset, pred_len)
                for v in FULL_ORDER:
                    entry = all_results[v].get(key, {})
                    row.append(f"{entry['MSE']:.6f}" if entry else "")
                    row.append(f"{entry['MAE']:.6f}" if entry else "")
                w.writerow(row)
    print(f"  -> {detail_path}")

    # -- average CSV ---------------------------------------------------------
    avg_path = Path("visualization/data/ablation_average.csv")
    with avg_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["Dataset"]
        for v in FULL_ORDER:
            header += [f"{v} MSE", f"{v} MAE"]
        w.writerow(header)

        for dataset in ALL_DATASETS:
            row = [dataset]
            for v in FULL_ORDER:
                keys = [(dataset, p) for p in PRED_LENS]
                values = [all_results[v].get(k) for k in keys]
                values = [x for x in values if x is not None]
                if values:
                    mse = sum(x["MSE"] for x in values) / len(values)
                    mae = sum(x["MAE"] for x in values) / len(values)
                    row.append(f"{mse:.6f}")
                    row.append(f"{mae:.6f}")
                else:
                    row.append("")
                    row.append("")
            w.writerow(row)
    print(f"  -> {avg_path}")


# ============================================================================
# Part 2 — Geometry-focused ablation (4 variants, 4 datasets)
# ============================================================================

GEO_DATASETS = ("ETTh1", "ETTm1", "weather", "traffic")

GEO_VARIANTS = {
    "POEM":              ("POEM",     "POEM",              "poem"),
    "NoGlobalForecast":  ("ablation", "NoGlobalForecast",  "noglobalforecast"),
    "NoGeometry":        ("ablation", "NoGeometry",        "nogeometry"),
    "LearnableGeometry": ("ablation", "LearnableGeometry", "learnablegeometry"),
}

EXPECTED_VARIANTS = {
    "POEM":              (1, 1, 0, 1, "fixed"),
    "NoGlobalForecast":  (1, 1, 0, 0, "fixed"),
    "NoGeometry":        (1, 1, 0, 1, "none"),
    "LearnableGeometry": (1, 1, 0, 1, "learnable"),
}

CONTROL_KEYS = (
    "data", "data_path", "features", "seq_len", "pred_len", "enc_in",
    "period_len", "mixer_layers", "mixer_dropout", "d_model", "revin",
    "affine", "learning_rate", "weight_decay", "gradient_clip", "loss",
    "lradj", "train_epochs", "patience", "batch_size", "seed",
)

GEO_ORDER = ("POEM", "NoGlobalForecast", "NoGeometry", "LearnableGeometry")


def _load_settings(result):
    return json.loads(Path(result["PATH"]).with_name("settings.json").read_text())


def _safe_div(a, b):
    return a / b if b != 0 else float("nan")


def collect_geo_results():
    tasks = {(d, p) for d in GEO_DATASETS for p in PRED_LENS}
    all_results = {}
    for name, (output_family, model_id, run_tag) in GEO_VARIANTS.items():
        results = collect_results(output_family, model_id, run_tag, seed=SEED)
        missing = tasks - results.keys()
        if missing:
            missing_text = ", ".join(f"{d}-{p}" for d, p in sorted(missing))
            raise RuntimeError(f"{name} is incomplete: {missing_text}")
        all_results[name] = results
    return all_results


def validate_geo_settings(all_results):
    baseline = all_results["POEM"]
    for name, results in all_results.items():
        expected = EXPECTED_VARIANTS[name]
        for key, result in results.items():
            settings = _load_settings(result)
            actual = (
                settings.get("use_phase_interaction"),
                settings.get("use_harmonic_modulation"),
                settings.get("use_vanilla_mixer"),
                settings.get("use_global_forecast", 1),
                settings.get("geometry_type", "fixed"),
            )
            if actual != expected:
                raise RuntimeError(
                    f"{name} {key} has settings {actual}, expected {expected}"
                )
            if name == "POEM":
                continue
            base_settings = _load_settings(baseline[key])
            diffs = {
                k: (base_settings.get(k), settings.get(k))
                for k in CONTROL_KEYS
                if base_settings.get(k) != settings.get(k)
            }
            if diffs:
                raise RuntimeError(
                    f"{name} {key} control mismatch vs POEM: {diffs}"
                )


def write_geo_detail_csv(all_results, path):
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Dataset", "Pred Len",
            *(c for v in GEO_ORDER for c in (f"{v} MSE", f"{v} MAE")),
            "NoGlobalForecast Δ MSE %", "NoGeometry Δ MSE %", "LearnableGeometry Δ MSE %",
        ])
        for dataset in GEO_DATASETS:
            for pred_len in PRED_LENS:
                row = [dataset, pred_len]
                metrics = {}
                for name in GEO_ORDER:
                    r = all_results[name][(dataset, pred_len)]
                    mse, mae = r["MSE"], r["MAE"]
                    metrics[name] = (mse, mae)
                    row.extend((f"{mse:.6f}", f"{mae:.6f}"))

                poem_mse = metrics["POEM"][0]
                for name in ("NoGlobalForecast", "NoGeometry", "LearnableGeometry"):
                    delta = _safe_div(
                        (metrics[name][0] - poem_mse) * 100, poem_mse
                    )
                    row.append(f"{delta:+.2f}%")
                w.writerow(row)
    print(f"  -> {path}")


def write_geo_average_csv(all_results, path):
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Dataset",
            *(c for v in GEO_ORDER for c in (f"{v} MSE", f"{v} MAE")),
            "NoGlobalForecast Δ MSE %", "NoGeometry Δ MSE %", "LearnableGeometry Δ MSE %",
        ])
        for dataset in GEO_DATASETS:
            row = [dataset]
            metrics = {}
            for name in GEO_ORDER:
                mse = sum(
                    all_results[name][(dataset, p)]["MSE"] for p in PRED_LENS
                ) / len(PRED_LENS)
                mae = sum(
                    all_results[name][(dataset, p)]["MAE"] for p in PRED_LENS
                ) / len(PRED_LENS)
                metrics[name] = (mse, mae)
                row.extend((f"{mse:.6f}", f"{mae:.6f}"))

            poem_mse = metrics["POEM"][0]
            for name in ("NoGlobalForecast", "NoGeometry", "LearnableGeometry"):
                delta = _safe_div(
                    (metrics[name][0] - poem_mse) * 100, poem_mse
                )
                row.append(f"{delta:+.2f}%")
            w.writerow(row)
    print(f"  -> {path}")


def write_geo_all_average_csv(all_results, path):
    """Long-format: one row per variant × dataset (averaged across pred_lens)."""
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Variant", "Dataset", "MSE", "MAE", "Δ MSE % vs POEM"])
        poem_avg = {}
        for dataset in GEO_DATASETS:
            poem_avg[dataset] = sum(
                all_results["POEM"][(dataset, p)]["MSE"] for p in PRED_LENS
            ) / len(PRED_LENS)

        for name in GEO_ORDER:
            for dataset in GEO_DATASETS:
                keys = [(dataset, p) for p in PRED_LENS]
                mse = sum(all_results[name][k]["MSE"] for k in keys) / len(keys)
                mae = sum(all_results[name][k]["MAE"] for k in keys) / len(keys)
                delta = _safe_div((mse - poem_avg[dataset]) * 100, poem_avg[dataset])
                w.writerow([name, dataset, f"{mse:.6f}", f"{mae:.6f}", f"{delta:+.2f}%"])
    print(f"  -> {path}")


def run_geometry_ablation():
    """Geometry-focused ablation with settings validation and delta calculations."""
    all_results = collect_geo_results()
    validate_geo_settings(all_results)

    Path("results").mkdir(parents=True, exist_ok=True)
    write_geo_detail_csv(all_results, "results/geometry_ablation_detail.csv")
    write_geo_average_csv(all_results, "results/geometry_ablation_average.csv")
    write_geo_all_average_csv(all_results, "results/geometry_ablation_all_average.csv")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=== Full ablation (8 variants) ===")
    run_full_ablation()
    print()
    print("=== Geometry-focused ablation (4 variants) ===")
    run_geometry_ablation()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
