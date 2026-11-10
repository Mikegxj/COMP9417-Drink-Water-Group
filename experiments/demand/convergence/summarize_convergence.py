"""Aggregate convergence runs, keeping all budgets and seeds explicit."""
import json
from pathlib import Path

import pandas as pd

from experiments.demand import ROOT


def main():
    output = ROOT / "results/demand"
    records = []
    for directory in ("convergence", "convergence_repeats", "convergence_extended", "convergence_lstm_extended"):
        for path in (output / directory).glob("*/seed_*/summary.json"):
            row = json.loads(path.read_text())
            row["run_directory"] = str(path.parent.relative_to(output))
            records.append(row)
    runs = pd.DataFrame(records)
    runs.to_csv(output / "convergence_runs.csv", index=False)
    # The larger recorded budget supersedes its shorter diagnostic run.
    latest = runs.sort_values("max_epochs").drop_duplicates(["case", "seed"], keep="last")
    latest.to_csv(output / "convergence_latest.csv", index=False)
    summary = latest.groupby("case").agg(seeds=("seed", "size"), mean_mape=("val_mape", "mean"),
                                         std_mape=("val_mape", "std"),
                                         min_mape=("val_mape", "min"), max_mape=("val_mape", "max"))
    summary.to_csv(output / "convergence_summary.csv")
    print(latest[["case", "seed", "best_epoch", "epochs_run", "epoch16_mape", "val_mape",
                  "stop_reason", "elapsed_seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
