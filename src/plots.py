import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

EXP = Path("experiments/experiments.csv")

def make_core_plots(df=None, out_dir="results"):
    if df is None:
        df = pd.read_csv(EXP)

    if df["oos_sharpe"].dropna().empty:
        raise RuntimeError("No scored rows in experiments.csv.")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 2) Scatter: OOS Sharpe vs Inventory_Penalty (hue=fill_prob)
    plt.figure(figsize=(7, 4))
    sns.scatterplot(
        x="inventory_penalty",
        y="oos_sharpe",
        hue="fill_prob",
        palette="viridis",
        data=df,
        edgecolor="k",
        linewidth=0.5,
        alpha=0.8
    )
    plt.title("OOS Sharpe vs Inventory Penalty")
    plt.xlabel("inventory_penalty"); plt.ylabel("oos_sharpe")
    plt.grid(ls="--", alpha=0.5); plt.tight_layout()
    plt.savefig(f"{out_dir}/scatter_sharpe_vs_penalty.png")
    plt.close()

    # 3) Box: OOS Sharpe by Fill_Prob
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="fill_prob", y="oos_sharpe", data=df)
    plt.title("OOS Sharpe by Fill Probability")
    plt.xlabel("fill_prob"); plt.ylabel("oos_sharpe")
    plt.grid(ls="--", alpha=0.5); plt.tight_layout()
    plt.savefig(f"{out_dir}/box_sharpe_by_fillprob.png")
    plt.close()

    # 4) Box: OOS Sharpe by Max_Inventory
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="max_inventory", y="oos_sharpe", data=df)
    plt.title("OOS Sharpe by Max Inventory")
    plt.xlabel("max_inventory"); plt.ylabel("oos_sharpe")
    plt.grid(ls="--", alpha=0.5); plt.tight_layout()
    plt.savefig(f"{out_dir}/box_sharpe_by_inventory.png")
    plt.close()

    # 5) Heatmap: Avg OOS Sharpe (spread_mult vs inv_penalty)
    if df["spread_mult"].nunique() > 6:
        df["spread_bin"] = pd.cut(df["spread_mult"].astype(float), bins=5, duplicates="drop")
    else:
        df["spread_bin"] = df["spread_mult"].astype(float)

    if df["inventory_penalty"].nunique() > 6:
        df["inv_bin"] = pd.cut(df["inventory_penalty"].astype(float), bins=5, duplicates="drop")
    else:
        df["inv_bin"] = df["inventory_penalty"].astype(float)

    heatmap_data = df.pivot_table(
        values="oos_sharpe",
        index="inv_bin",
        columns="spread_bin",
        aggfunc=np.mean
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_data,
        annot=True, fmt=".3f",
        cmap="viridis", linewidths=0.5,
        cbar_kws={"label": "Avg OOS Sharpe"}
    )
    plt.title("Heatmap: Avg OOS Sharpe\\n(inv_penalty vs spread_mult)")
    plt.xlabel("spread_mult (binned)"); plt.ylabel("inventory_penalty (binned)")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/heatmap_sharpe_grid.png")
    plt.close()

    # 6) Sharpe vs Max Drawdown by Asset (bar + line)
    summary = df[df["asset"].isin(["ADA", "BTC", "ETH"])].copy()
    summary = summary.groupby(["asset", "freq"]).agg({
        "oos_sharpe": "max",
        "max_dd": "min"
    }).reset_index()

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    x = range(len(summary))

    ax1.bar(x, summary["oos_sharpe"], width=0.4, color="skyblue", label="OOS Sharpe")
    ax2.plot(x, summary["max_dd"], color="crimson", marker="o", label="Max DD")

    ax1.set_ylabel("OOS Sharpe"); ax2.set_ylabel("Max Drawdown")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{a}_{f}" for a, f in zip(summary["asset"], summary["freq"])])
    ax1.set_title("Sharpe vs Max Drawdown by Asset")
    fig.tight_layout()
    plt.savefig(f"{out_dir}/sharpe_vs_drawdown.png")
    plt.close()
