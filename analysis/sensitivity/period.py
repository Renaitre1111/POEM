"""Summarize POEM period-length robustness at prediction length 96."""

import csv
import json
from pathlib import Path

from summarize_results import collect_results


SEED = 2021
PRED_LEN = 96
PERIODS = {
    "ETTh1": (12, 18, 24, 36, 48),
    "ETTm1": (48, 72, 96, 144, 168),
    "weather": (12, 18, 24, 36, 42),
}
DEFAULT_PERIODS = {"ETTh1": 24, "ETTm1": 96, "weather": 24}
CONTROL_KEYS = (
    "data",
    "data_path",
    "features",
    "seq_len",
    "pred_len",
    "enc_in",
    "mixer_layers",
    "mixer_dropout",
    "d_model",
    "phase_rank",
    "harmonics",
    "geometry_type",
    "use_phase_interaction",
    "use_harmonic_modulation",
    "use_vanilla_mixer",
    "use_global_forecast",
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
SETTING_DEFAULTS = {
    "phase_rank": 4,
    "harmonics": 2,
    "geometry_type": "fixed",
}


def load_settings(result):
    return json.loads(Path(result["PATH"]).with_name("settings.json").read_text())


def setting_value(settings, key):
    return settings.get(key, SETTING_DEFAULTS.get(key))


def collect_period_results():
    baseline = collect_results("POEM", "POEM", "poem", seed=SEED)
    selected = {}
    for dataset, periods in PERIODS.items():
        baseline_key = (dataset, PRED_LEN)
        if baseline_key not in baseline:
            raise RuntimeError(f"Missing formal POEM baseline for {dataset}-{PRED_LEN}")
        selected[(dataset, DEFAULT_PERIODS[dataset])] = baseline[baseline_key]

        for period in periods:
            if period == DEFAULT_PERIODS[dataset]:
                continue
            model_id = f"PeriodLen{period}"
            run_tag = f"period_sensitivity_{period}"
            results = collect_results(
                "period_sensitivity", model_id, run_tag, seed=SEED
            )
            if baseline_key not in results:
                raise RuntimeError(f"Missing period sensitivity result: {dataset}-{period}")
            selected[(dataset, period)] = results[baseline_key]
    return baseline, selected


def validate_settings(baseline, selected):
    for dataset, periods in PERIODS.items():
        baseline_settings = load_settings(baseline[(dataset, PRED_LEN)])
        for period in periods:
            settings = load_settings(selected[(dataset, period)])
            if settings.get("period_len") != period:
                raise RuntimeError(
                    f"{dataset}-{period} stored period_len={settings.get('period_len')}"
                )
            differences = {
                key: (setting_value(baseline_settings, key), setting_value(settings, key))
                for key in CONTROL_KEYS
                if setting_value(baseline_settings, key) != setting_value(settings, key)
            }
            if differences:
                raise RuntimeError(
                    f"{dataset}-{period} differs from its POEM baseline: {differences}"
                )


def main():
    baseline, selected = collect_period_results()
    validate_settings(baseline, selected)

    output = Path("visualization/data/period_sensitivity.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Dataset", "Pred Len", "Period Len", "Period Ratio", "MSE", "MAE", "Source"
        ])
        for dataset, periods in PERIODS.items():
            default_period = DEFAULT_PERIODS[dataset]
            for period in periods:
                result = selected[(dataset, period)]
                source = "formal_baseline" if period == default_period else "period_sensitivity"
                writer.writerow([
                    dataset,
                    PRED_LEN,
                    period,
                    f"{period / default_period:.2f}",
                    f"{result['MSE']:.6f}",
                    f"{result['MAE']:.6f}",
                    source,
                ])
    print(f"  -> {output}")


if __name__ == "__main__":
    main()
