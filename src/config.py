from dataclasses import dataclass

EXP = "experiments/experiments.csv"
METRIC_COLS = ["train_sharpe", "val_sharpe", "oos_sharpe", "max_dd", "turnover"]
PARAM_COLS = ["asset", "freq", "spread_mult", "fill_prob", "max_inventory", "inventory_penalty"]

@dataclass
class Config:
    fixed_spread:      float
    max_inventory:     int
    inventory_penalty: float
    fee_per_share:     float
    slip_ppv:          float
    fill_prob:         float
