"""Summarize the seed-2021 POEM rank and harmonic sensitivity runs."""

import csv
import json
from pathlib import Path

from summarize_results import collect_results


SEED = 2021
DATASETS = ("ETTh1", "ETTm1", "weather")
PRED_LENS = (96, 192, 336, 720)
AXES = {
    "phase_rank": (
        (1, "Rank1", "sensitivity_rank_1"),
        (2, "Rank2", "sensitivity_rank_2"),
        (4, "Baseline", "poem"),
        (8, "Rank8", "sensitivity_rank_8"),
        (16, "Rank16", "sensitivity_rank_16"),
    ),
    "harmonics": (
        (1, "Harmonics1", "sensitivity_harmonics_1"),
        (2, "Baseline", "poem"),
        (3, "Harmonics3", "sensitivity_harmonics_3"),
        (4, "Harmonics4", "sensitivity_harmonics_4"),
        (6, "Harmonics6", "sensitivity_harmonics_6"),
    ),
}
VARIANT_VALUES = {
    "Baseline": (4, 2),
    "Rank1": (1, 2),
    "Rank2": (2, 2),
    "Rank8": (8, 2),
    "Rank16": (16, 2),
    "Harmonics1": (4, 1),
    "Harmonics3": (4, 3),
    "Harmonics4": (4, 4),
    "Harmonics6": (4, 6),
}
CONTROL_KEYS = (
    "data",
    "data_path",
    "features",
    "seq_len",
    "pred_len",
    "enc_in",
    "period_len",
    "mixer_layers",
    "mixer_dropout",
    "d_model",
    "revin",
    "affine",
    "learning_rate",
    "weight_decay",
    "gradient_clip",
    "loss",
    "lradj",
    "train_epochs",
    "patience",
    "batch_size",
    "seed",
)


def load_settings(result):
    settings_path = Path(result["PATH"]).with_name("settings.json")
    return json.loads(settings_path.read_text())


def validate_settings(runs, expected):
    baseline = runs["Baseline"]
    for model_id, results in runs.items():
        phase_rank, harmonics = VARIANT_VALUES[model_id]
        for key in expected:
            settings = load_settings(results[key])
            actual_values = (
                settings.get("phase_rank", 4),
                settings.get("harmonics", 2),
            )
            if actual_values != (phase_rank, harmonics):
                raise RuntimeError(
                    f"{model_id} {key} has r/K={actual_values}, "
                    f"expected {(phase_rank, harmonics)}"
                )
            if model_id == "Baseline":
                continue
            baseline_settings = load_settings(baseline[key])
            differences = {
                name: (baseline_settings.get(name), settings.get(name))
                for name in CONTROL_KEYS
                if baseline_settings.get(name) != settings.get(name)
            }
            if differences:
                raise RuntimeError(
                    f"{model_id} {key} differs from the POEM baseline: {differences}"
                )


def main():
    runs = {}
    for variants in AXES.values():
        for _value, model_id, run_tag in variants:
            if model_id not in runs:
                output_family = "POEM" if model_id == "Baseline" else "sensitivity"
                stored_model_id = "POEM" if model_id == "Baseline" else model_id
                runs[model_id] = collect_results(
                    output_family, stored_model_id, run_tag, seed=SEED
                )

    expected = {(dataset, pred_len) for dataset in DATASETS for pred_len in PRED_LENS}
    for model_id, results in runs.items():
        missing = expected - results.keys()
        if missing:
            missing_text = ", ".join(
                f"{dataset}-{pred_len}" for dataset, pred_len in sorted(missing)
            )
            raise RuntimeError(f"{model_id} is incomplete: {missing_text}")
    validate_settings(runs, expected)

    output_path = Path("visualization/data/sensitivity.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "Pred Len", "Parameter", "Value", "MSE", "MAE"])
        for dataset in DATASETS:
            for pred_len in PRED_LENS:
                for parameter, variants in AXES.items():
                    for value, model_id, _run_tag in variants:
                        result = runs[model_id][(dataset, pred_len)]
                        writer.writerow([
                            dataset,
                            pred_len,
                            parameter,
                            value,
                            f"{result['MSE']:.6f}",
                            f"{result['MAE']:.6f}",
                        ])

    print(f"  -> {output_path}")

    average_path = Path("visualization/data/sensitivity_average.csv")
    with average_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "Parameter", "Value", "MSE", "MAE"])
        for dataset in DATASETS:
            for parameter, variants in AXES.items():
                for value, model_id, _run_tag in variants:
                    mse = sum(
                        runs[model_id][(dataset, pred_len)]["MSE"]
                        for pred_len in PRED_LENS
                    ) / len(PRED_LENS)
                    mae = sum(
                        runs[model_id][(dataset, pred_len)]["MAE"]
                        for pred_len in PRED_LENS
                    ) / len(PRED_LENS)
                    writer.writerow([
                        dataset,
                        parameter,
                        value,
                        f"{mse:.6f}",
                        f"{mae:.6f}",
                    ])

    print(f"  -> {average_path}")


if __name__ == "__main__":
    main()
