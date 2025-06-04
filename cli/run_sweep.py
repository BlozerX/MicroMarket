
"""
This script will run the whole notebook pipeline when you call it:

1. If experiments/experiments.csv doesn't exist or is empty, we enqueue a 40-point ADA grid.
2. Score any ADA rows that are missing metrics.
3. Copy the best ADA params over to BTC_1sec and ETH_1min rows.
4. Score any remaining blank BTC/ETH rows.
5. Generate the five main diagnostic plots into results/.

Just run:
    python cli/run_sweep.py
"""

import pathlib
import pandas as pd
from tqdm import tqdm

# our own modules from src/
from src import grid, loader, backtest, plots

# paths
EXP = pathlib.Path("experiments/experiments.csv")
DATA_DIR = pathlib.Path("data")


def evaluate_row(row: pd.Series) -> dict:
    """
    load the order book for this row’s asset and freq,
    turn the row into a cfg, run the backtest, and return metrics.
    """
    data_file = DATA_DIR / f"{row.asset}_{row.freq}.csv"
    book = loader.load_lob(str(data_file))            # load LOB from CSV
    cfg = backtest.row_to_cfg(row, book)              # make a cfg from this row
    return backtest.run_backtest_split(book, cfg)      # run train/val/OOS backtest


def main():
    # 1) enqueue ADA grid if experiments.csv doesn’t exist or is empty
    if not EXP.exists() or pd.read_csv(EXP).empty:
        grid.enqueue_ada_grid()  # adds 40 ADA rows
        print("no experiments.csv or it's empty -> enqueued ADA grid")

    # 2) read experiments.csv and score ADA rows missing metrics
    df = pd.read_csv(EXP)
    missing = df["train_sharpe"].isna()
    if missing.any():
        print(f"scoring {missing.sum()} ADA rows")
        for idx in tqdm(df[missing].index):
            df.loc[idx, backtest.METRIC_COLS] = evaluate_row(df.loc[idx])
        df.to_csv(EXP, index=False)
        print("done scoring ADA rows")

    # 3) clone best ADA params over to BTC and ETH rows (if not already there)
    grid.clone_best_to_other_assets()
    df = pd.read_csv(EXP)  # re-read since clone may have added rows

    # 4) score any remaining BTC/ETH rows that are blank
    missing = df["train_sharpe"].isna()
    if missing.any():
        print(f"scoring {missing.sum()} BTC/ETH rows")
        for idx in tqdm(df[missing].index):
            try:
                df.loc[idx, backtest.METRIC_COLS] = evaluate_row(df.loc[idx])
            except Exception as e:
                print(f"  row {idx} skipped: {e}")
        df.to_csv(EXP, index=False)
        print("done scoring BTC/ETH rows")

    # 5) make the five core plots into results/
    print("making core plots …")
    plots.make_core_plots(df, out_dir="results")
    print("all done - check results/ for plots and updated experiments.csv")


if __name__ == "__main__":
    main()
