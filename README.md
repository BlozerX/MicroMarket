
# Micromarket 
## Inventory-Aware Market Making Parameter Optimization in a Simulated Exchange.  
*(a brutally transparent repo-log of my market-making experiments on crypto LOB data)*  

> **TL;DR** – I grid-search four knobs (`spread_mult × fill_prob × max_inventory × inventory_penalty`) on **ADA-1 min**, clone the best config to **BTC-1 sec** & **ETH-1 min**, back-test each slice (train / val / OOS), and auto-generate diagnostic plots + metrics. 

---

## 0. Directory Layout

```
.
├── cli/                 # command-line entry points
│   └── run_sweep.py
├── data/                # raw LOB CSVs (NOT tracked – >100 MB)
├── experiments/         # experiments.csv lives here
├── notebooks/           # `micromarket.ipynb` – scratchpad & sanity checks
├── results/             # auto-generated plots
├── final-results/       # hand-picked plots for the write-up
├── src/                 # core library code
│   ├── backtest.py
│   ├── config.py
│   ├── grid.py
│   ├── loader.py
│   └── plots.py
├── tests.py             # two smoke tests
├── requirements.txt
└── README.md            # ← you are here
```

---

## 1. Quick-start

```bash
git clone https://github.com/bloxerx/micromarket.git
cd micromarket
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # pandas, numpy, matplotlib, seaborn, tqdm, kagglehub …
```

### Grab the data (100 + MB, so not in Git)

```bash
python - <<'PY'
from src.loader import fetch_btc_1sec
fetch_btc_1sec(local_dir="data")       #  first call downloads & caches
PY
# manually drop ADA_1min.csv and ETH_1min.csv into ./data/ (links in notebook header)
```

---

## 2. One-liner to reproduce every result

```bash
python cli/run_sweep.py
```

What it does under the hood:

| Step | What actually happens (file → function) |
|------|-----------------------------------------|
| 1 | If `experiments/experiments.csv` is empty → `src.grid.enqueue_ada_grid()` appends **40 ADA rows** (balanced stratified sample out of 144 combos). |
| 2 | Each blank ADA row is back-tested (`src.backtest.run_backtest_split`) and metrics written back. |
| 3 | **Best** ADA config (by `oos_sharpe`) is cloned to BTC-1 sec & ETH-1 min (`src.grid.clone_best_to_other_assets`). |
| 4 | Remaining blanks (BTC / ETH) are scored the same way. |
| 5 | `src.plots.make_core_plots()` pumps six PNGs into `./results/`. |

Ran on Google Colab.

---

## 3. Hyper-parameter space

| knob | values |
|------|----------------------------------|
| `spread_mult`      | 0.3, 0.5, 0.8, 1.2 |
| `fill_prob`        | 0.6, 0.75, 0.9 |
| `max_inventory`    | 20, 50, 100 |
| `inventory_penalty`| 1e-4, 1e-3, 2e-3, 5e-3 |
| _fixed_ | `fee_per_share` = 2e-4, `slip_ppv` = 5e-5 |

40 rows = 10 combos per `spread_mult`, fully deduped against previous runs.

---

## 4. Core logic (napkin, but code is 1-to-1)

```
loader.load_lob → standardise cols → backtest.row_to_cfg
MarketMaker.step → quote (bid,ask) ± inventory skew → probabilistic fills
State.mtm → mark-to-mid every tick
run_backtest_split → walk-forward 60/20/20
metrics → Sharpe, PnL, MaxDD, turnover (# trades)
```

Edge-cases I hit & patched:

* **ParserError** on malformed CSV rows → `on_bad_lines="skip"` in `loader.py`.
* Huge BTC file (≈1 GB) choking GitHub → download script, ignore in `.gitignore`.
* Duplicate param rows when I re-ran the grid → `grid.deduplicate_experiments()`.

---

## 5. Plots & What They Mean

| Plot | tl;dr (+ what I looked at) |
|------|----------------------------|
| `scatter_oos_sharpe_vs_inventory_penalty.png` | Clear upward tilt: as `inventory_penalty` increases, inventory risk drops and Sharpe tightens. Low penalties (≈0.0001) scatter below 2 Sharpe; 0.005 often > 3.0. |
| `boxplot_oos_sharpe_vs_fillprob.png` | Median Sharpe climbs from 0.6 fill-prob (~0.8) → 0.75 (~1.6) → 0.9 (~3.2). High fill-prob obviously wins; tail risk grows but is offset by spread. |
| `boxplot_oos_sharpe_vs_max_inventory.png` | 20 & 50 beat 100. Allowing 100-coin inventory drags Sharpe (wider PnL tails, more drawdown). |
| `heatmap_oos_sharpe_spread_invpenalty.png` | Sweet-spot at `(spread_mult=0.3 | 0.5, inv_penalty=0.005)` hitting ~3 Sharpe. Too tight a spread (0.3) + low penalty burns; too wide a spread (1.2) starves fills. |
| `sharpe_vs_drawdown.png` | **ADA-1 min** wins (Sharpe ≈ 4.3, MaxDD ≈ 0.3). BTC-1 sec barely breaks even (Sharpe ≈ 0.05, nasty 7 % drawdown – microstructure & min-tick mismatch). ETH sits in the middle. |

Full PNGs live in [`/results`](./results).

---

## 6. What Went Wrong (and how I fixed it)

| Bug / Roadblock | Fix |
|-----------------|-----|
| *“ParserError: Error tokenizing data”* on BTC file | Skip malformed lines, then assert schema. |
| NaN Sharpe due to 0 std returns | Guard in `compute_metrics`: if `rets.std()==0` → Sharpe=0. |
| Massive duplicate rows after hot-reloading notebook | `grid.deduplicate_experiments()` trims by param key. |
| Accidentally committed a 1 GB file | Rewrote history, added `data/` to `.gitignore`, replaced with download helper. |

---

## 7. Reproducing My Exact Numbers

```bash
# 1. fresh clone + install (see §1)
# 2. download ADA, ETH, BTC CSVs into ./data/
python cli/run_sweep.py
python tests.py                  # optional sanity check
open results/*png                # or embed in Jupyter for a report
```

Environment I ran:

```
Python 3.11.3  |  pandas 2.2.2  |  numpy 1.26.4
matplotlib 3.9 |  seaborn 0.13  |  tqdm 4.66
```

---

## 8. Next Steps

* **Latency model** – current fill logic is static Bernoulli; swap in exponential decay vs queue depth.  
* **Re-inventory-risk** – test dynamic penalty = k * |inventory| not constant.  
* **RL fine-tune** – seed an A2C agent with best grid config as prior.
* **Segment bull and bear regimes** – Label directional regimes manually on HFT data and re-run the grid to evaluate robustness across market phases.


---

## 9. Credits & Licence

Everything here is MIT. Kaggle dataset © Martin Søgaard Nielsen (license inside dataset).

> Built & debugged by **Sudershan Sarraf** (IIIT-Hyderabad, ECE ’27).
