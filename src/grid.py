import csv
import random
from itertools import product
import pandas as pd
from pathlib import Path

# Shared paths and column constants
EXP = Path("experiments/experiments.csv")

param_cols = ["asset", "freq", "spread_mult", "fill_prob", "max_inventory", "inventory_penalty"]
metric_cols = ["train_sharpe", "val_sharpe", "oos_sharpe", "train_pnl", "val_pnl", "oos_pnl"]


def enqueue_ada_grid():
    """
    Creates a 40-point grid of ADA parameter combinations and appends new rows to experiments.csv.
    Avoids adding duplicates.
    """
    grid = {
        "spread_mult"      : [0.3, 0.5, 0.8, 1.2],          # 4
        "fill_prob"        : [0.6, 0.75, 0.9],              # 3
        "max_inventory"    : [20, 50, 100],                 # 3
        "inventory_penalty": [0.0001, 0.001, 0.002, 0.005], # 4
    }

    full = list(product(*grid.values()))    # 4×3×3×4 = 144
    random.shuffle(full)                    # mix the order

    # stratified down-sample: 10 per spread_mult
    target = 40
    bucket = {sm: [] for sm in grid["spread_mult"]}
    for combo in full:
        sm = combo[0]
        if len(bucket[sm]) < target // len(grid["spread_mult"]):
            bucket[sm].append(combo)
        if sum(len(v) for v in bucket.values()) == target:
            break

    sample40 = list(v for combos in bucket.values() for v in combos)

    # De-dupe against existing rows
    df = pd.read_csv(EXP)
    already = set(zip(df.asset, df.freq,
                      df.spread_mult, df.fill_prob,
                      df.max_inventory, df.inventory_penalty))
    new_rows = [("ADA", "1min", *c) for c in sample40
                if ("ADA", "1min", *c) not in already]

    with open(EXP, "a", newline="") as f:
        writer = csv.writer(f)
        for r in new_rows:
            writer.writerow(list(r) + [0.0002, 0.00005] + [""] * len(metric_cols))

    print(f"Added {len(new_rows)} balanced ADA grid rows")


def deduplicate_experiments():
    """
    Drops duplicate parameter rows from experiments.csv based on param_cols.
    """
    df = pd.read_csv(EXP)
    dedup = df.drop_duplicates(subset=param_cols, keep='first')
    dedup.to_csv(EXP, index=False)
    print("Deduplicated: now", len(dedup), "rows")


def clone_best_to_other_assets():
    """
    Copies the best-performing ADA row to BTC_1sec and ETH_1min rows with the same parameters.
    """
    exp_df = pd.read_csv(EXP)

    ada_rows = exp_df[(exp_df["asset"] == "ADA") & (exp_df["freq"] == "1min")]
    if ada_rows.empty:
        raise ValueError("No ADA rows found – run ADA sweep first.")

    ada_best = ada_rows.sort_values("oos_sharpe", ascending=False).iloc[0]
    param_values = ada_best[param_cols[2:]].to_list()  # skip asset & freq

    new_rows = [
        ["BTC", "1sec"] + param_values,
        ["ETH", "1min"] + param_values,
    ]

    with open(EXP, "a", newline="") as f:
        writer = csv.writer(f)
        for r in new_rows:
            writer.writerow(r + [""] * len(metric_cols))

    print("Added BTC_1sec and ETH_1min rows with ADA-tuned parameters")
