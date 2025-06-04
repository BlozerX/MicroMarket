import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from dataclasses import dataclass, field
from src.loader import load_lob
from src.config import Config, State, Trade

# Path to the experiments CSV
EXP = "experiments/experiments.csv"

# Parameter column names
param_cols = [
    "asset",
    "freq",
    "spread_mult",
    "fill_prob",
    "max_inventory",
    "inventory_penalty"
]

# Metric column names (for writing back to EXP)
metric_cols = [
    "train_sharpe",
    "val_sharpe",
    "oos_sharpe",
    "max_dd",
    "turnover"
]

@dataclass
class Trade:
    side:  str
    price: float
    qty:   int
    t:     int

@dataclass
class State:
    cash:       float = 0.0
    inventory:  int   = 0
    trades:     list  = field(default_factory=list)

    def mtm(self, px: float) -> float:
        return self.cash + self.inventory * px


# SECTION – MarketMaker class and core backtest logic

class MarketMaker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = State()

    def _quote_prices(self, mid):
        inv_adj = self.cfg.inventory_penalty * self.state.inventory
        return (
            mid - self.cfg.fixed_spread / 2 - inv_adj,
            mid + self.cfg.fixed_spread / 2 - inv_adj
        )

    def step(self, row):
        bid_px, ask_px = self._quote_prices(row.mid)
        if bid_px >= row.best_bid and np.random.random() < self.cfg.fill_prob:
            self._exec("buy", row.best_bid, 1, row.t)
        if ask_px <= row.best_ask and np.random.random() < self.cfg.fill_prob:
            self._exec("sell", row.best_ask, 1, row.t)

    def _exec(self, side, price, qty, t):
        fee = self.cfg.fee_per_share * qty
        slip = self.cfg.slip_ppv * qty
        px = price + slip if side == "buy" else price - slip
        self.state.cash += (-px * qty if side == "buy" else px * qty) - fee
        self.state.inventory += qty if side == "buy" else -qty
        self.state.trades.append(Trade(side, px, qty, t))


def split_indices(n, train=0.6, val=0.2):
    """
    Return three slices for train/validation/OOS given total length n.
    """
    i_tr = int(n * train)
    i_va = int(n * (train + val))
    return slice(0, i_tr), slice(i_tr, i_va), slice(i_va, n)


def backtest(book: pd.DataFrame, cfg: Config):
    """
    Run a full backtest over the entire book (no splits),
    returning the MarketMaker instance and equity time series.
    """
    mm = MarketMaker(cfg)
    mm.state.cash = 1.0            # initialize cash to avoid divide-by-zero
    equity = [1.0]                 # starting equity
    for _, row in tqdm(book.iterrows(), total=len(book), leave=False):
        mm.step(row)
        equity.append(mm.state.mtm(row.mid))
    return mm, pd.Series(equity, name="equity")


def compute_metrics(eq: pd.Series) -> dict:
    """
    Given an equity time series, compute PnL, Sharpe ratio, and MaxDD.
    """
    pnl = eq.iloc[-1] - eq.iloc[0]
    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = (np.sqrt(252) * rets.mean() / rets.std()) if len(rets) and rets.std() > 0 else 0.0
    cummax = eq.cummax().replace(0, np.nan)  # avoid division by zero
    maxdd = ((cummax - eq) / cummax).fillna(0).max()
    return {
        "PnL": round(pnl, 3),
        "Sharpe": round(sharpe, 3),
        "MaxDD": round(maxdd, 3)
    }


def run_backtest_split(df: pd.DataFrame, cfg: Config) -> dict:
    """
    Split the DataFrame into train/val/OOS, run backtest on each slice,
    and return a dict with train_sharpe, val_sharpe, oos_sharpe, max_dd, turnover.
    """
    sli_tr, sli_va, sli_os = split_indices(len(df))
    results = {}
    for tag, sl in zip(["train", "val", "oos"], [sli_tr, sli_va, sli_os]):
        mm, eq = backtest(df.iloc[sl].reset_index(drop=True), cfg)
        metrics = compute_metrics(eq)
        results[f"{tag}_sharpe"] = round(metrics["Sharpe"], 3)
        if tag == "oos":
            results["max_dd"] = round(metrics["MaxDD"], 3)
            results["turnover"] = len(mm.state.trades)
    return results


# SECTION – Helper to build a Config object from a row and its LOB DataFrame

def build_config_from_row(row: pd.Series, lob: pd.DataFrame) -> Config:
    """
    Given a row (with fields: spread_mult, fill_prob, max_inventory, inventory_penalty,
    fee_per_share, slip_ppv) and a loaded LOB DataFrame, build and return a Config.
    """
    median_spread = (lob.best_ask - lob.best_bid).median()
    return Config(
        fixed_spread=median_spread * float(row["spread_mult"]),
        max_inventory=int(row["max_inventory"]),
        inventory_penalty=float(row["inventory_penalty"]),
        fee_per_share=float(row["fee_per_share"]),
        slip_ppv=float(row["slip_ppv"]),
        fill_prob=float(row["fill_prob"])
    )


# SECTION – Score only the first unscored row (for quick testing)

def score_first_unscored_row():
    """
    Find the first row in experiments.csv where train_sharpe is missing,
    run a backtest on that row, and write metrics back to experiments.csv.
    """
    exp_df = pd.read_csv(EXP)
    row_idx = exp_df[exp_df["train_sharpe"].isna()].index.min()

    if pd.isna(row_idx):
        print("All experiments scored!")
        return

    # Extract parameters and cast numeric fields
    row = exp_df.loc[row_idx, param_cols]
    for k in ["spread_mult", "fill_prob", "inventory_penalty", "fee_per_share", "slip_ppv"]:
        row[k] = float(row[k])
    row["max_inventory"] = int(row["max_inventory"])

    # Load LOB and build Config
    data_path = f"{row['asset']}_{row['freq']}.csv"
    lob = load_lob(data_path)
    cfg = build_config_from_row(row, lob)

    # Run backtest on full LOB (split inside run_backtest_split)
    metrics = run_backtest_split(lob, cfg)
    print(f"Row {row_idx} metrics:", metrics)

    # Write metrics back to experiments.csv
    for k, v in metrics.items():
        exp_df.at[row_idx, k] = v
    exp_df.to_csv(EXP, index=False)
    print(f"Updated experiments.csv (row {row_idx})")


# SECTION – Evaluate a single row’s parameters (used by full sweep)

def evaluate_params(row: pd.Series) -> dict:
    """
    Given a single row from experiments.csv, load its LOB,
    build a Config, run the backtest split, and return metrics dict.
    """
    asset, freq = row["asset"], row["freq"]
    data_path = f"{asset}_{freq}.csv"
    lob = load_lob(data_path)
    cfg = build_config_from_row(row, lob)
    return run_backtest_split(lob, cfg)


# SECTION – Main sweep over all unscored rows

def run_full_sweep():
    """
    Loop over every row in experiments.csv where train_sharpe is missing or blank,
    run evaluate_params for each, and write all results back to experiments.csv.
    """
    exp_df = pd.read_csv(EXP)
    pending_mask = exp_df["train_sharpe"].isna() | (exp_df["train_sharpe"].astype(str).str.strip() == "")
    pending_indices = exp_df[pending_mask].index.tolist()

    if not pending_indices:
        print("All experiments scored!")
        return

    print(f"Scoring {len(pending_indices)} pending rows…")
    for idx in tqdm(pending_indices, desc="Sweeping grid"):
        row = exp_df.loc[idx]
        metrics = evaluate_params(row)

        # Replace possible NaNs with 0.0
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                metrics[k] = 0.0

        for k, v in metrics.items():
            exp_df.at[idx, k] = v

    exp_df.to_csv(EXP, index=False)
    print("Sweep complete — experiments.csv updated")


# If this module is run directly, perform a full sweep
if __name__ == "__main__":
    run_full_sweep()
