"""Average complete 2021/2022/2023 results into a separate comparison CSV."""

from pathlib import Path

import numpy as np

from summarize_results import MODEL_RUNS, PRED_LENS, DATASETS, collect_results, write_comparison_csv


SEEDS = (2021, 2022, 2023)
MULTISEED_RUNS = {
    "POEM": MODEL_RUNS["POEM"],
    "PhaseFormer": MODEL_RUNS["PhaseFormer"],
    "TimeBase": MODEL_RUNS["TimeBase"],
}
EXPECTED_KEYS = {
    (dataset, pred_len)
    for dataset, _display_name in DATASETS
    for pred_len in PRED_LENS
}


def main():
    averaged_results = {}
    for display_name, (output_family, model_id, run_tag) in MULTISEED_RUNS.items():
        seed_results = {
            seed: collect_results(output_family, model_id, run_tag, seed=seed)
            for seed in SEEDS
        }
        for seed, results in seed_results.items():
            missing = EXPECTED_KEYS - results.keys()
            if missing:
                missing_text = ", ".join(
                    f"{dataset}-{pred_len}" for dataset, pred_len in sorted(missing)
                )
                raise RuntimeError(
                    f"{display_name} seed {seed} is incomplete: {missing_text}"
                )

        averaged_results[display_name] = {
            key: {
                metric: float(np.mean([
                    seed_results[seed][key][metric] for seed in SEEDS
                ]))
                for metric in ("MSE", "MAE")
            }
            for key in EXPECTED_KEYS
        }
        print(f"  [{display_name}] 28 entries averaged over seeds {SEEDS}")

    output_path = Path("results/comparison_multiseed_mean.csv")
    write_comparison_csv(averaged_results, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
