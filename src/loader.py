import pandas as pd
import numpy as np
import pathlib
import kagglehub
from kagglehub import KaggleDatasetAdapter

def fetch_btc_1sec(local_dir="/content") -> pd.DataFrame:
    """
    Downloads BTC_1sec.csv using kagglehub and formats it for LOB backtest.
    Saves a copy to local_dir for user visibility.
    """
    # Pull from Kaggle (first call downloads, later hits cache)
    df_raw = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "martinsn/high-frequency-crypto-limit-order-book-data",
        "BTC_1sec.csv",
    )

    df_raw["best_bid"] = df_raw["midpoint"] - df_raw["spread"] / 2
    df_raw["best_ask"] = df_raw["midpoint"] + df_raw["spread"] / 2
    df_raw["mid"]      = df_raw["midpoint"]
    df_raw["t"]        = range(len(df_raw))

    vis_dir = pathlib.Path(local_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_csv = vis_dir / "BTC_1sec.csv"
    df_raw.to_csv(out_csv, index=False)
    print(f"Saved : {out_csv.resolve()}")

    return df_raw[["t", "best_bid", "best_ask", "mid"]]


def load_lob(path: str) -> pd.DataFrame:
    """
    Loads a LOB CSV file and ensures the expected schema [t, best_bid, best_ask, mid].
    Handles both raw and preprocessed files.
    """
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        print(f"ParserError in {path} – skipping malformed rows")
        df = pd.read_csv(
            path,
            engine="python",
            on_bad_lines="skip",  # skip malformed lines
        )

    if {"midpoint", "spread"}.issubset(df.columns):
        df["best_bid"] = df["midpoint"] - df["spread"] / 2
        df["best_ask"] = df["midpoint"] + df["spread"] / 2
        df["mid"] = df["midpoint"]
    else:
        df["mid"] = (df["best_bid"] + df["best_ask"]) / 2

    df["t"] = np.arange(len(df))
    return df[["t", "best_bid", "best_ask", "mid"]]
