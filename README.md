# Quantitative Risk & Liquidity Stress Testing Engine (`stress_engine`)

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-15%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/typing-PEP%20585%20%7C%20604%20%7C%20646-informational.svg)]()

A high-performance quantitative risk and liquidity stress-testing engine developed to institutional risk standards (Basel Committee on Banking Supervision / FRM Part 1).

The platform evaluates historical asset and portfolio drawdowns, calculates dynamic conditional Value-at-Risk (VaR) and Expected Shortfall (ES), runs formal econometric regulatory backtests, and conducts vectorized Monte Carlo macroeconomic stress simulations across a 4D parametric state space ($36.45\text{M}$ path coordinates in $<3.0\text{s}$).

---

## Key Econometric & Engineering Features

* **Anti-Loop Mandate & Tensor Vectorization:** Zero iterative loops for mathematical transformations, portfolio combinations, simulation paths, or rolling windows. Employs SIMD-aligned C-contiguous memory structures via NumPy and SciPy.
* **Rolling Dynamic Strided Windows ($O(1)$ Allocation):** Leverages `numpy.lib.stride_tricks.sliding_window_view` to construct out-of-sample training/evaluation matrices without memory duplication.
* **Dynamic Conditional Volatility Engines:**
  * **RiskMetrics EWMA ($\lambda=0.94$):** Exponentially decaying memory footprint.
  * **GJR-GARCH(1,1):** Asymmetric leverage estimation via constrained Sequential Least Squares Programming (SLSQP) Maximum Likelihood Estimation (MLE).
  * **Filtered Historical Simulation (FHS):** Non-parametric standardization using empirical standardized innovations $z_t = r_t / \sigma_t$.
* **Regulatory Backtesting Suite:**
  * **Kupiec Proportion of Failures (POF) Test ($LR_{\text{uc}}$):** Unconditional coverage testing.
  * **Christoffersen Markov Independence Test ($LR_{\text{ind}}$):** Exception clustering and first-order Markov transition likelihood ratio testing.
  * **Christoffersen Combined Conditional Coverage ($LR_{\text{cc}}$):** Joint evaluation against $\chi^2(2)$.
  * **Basel Committee (BCBS) Traffic Light System:** Exact binomial cumulative density classification into Green, Yellow, and Red capital penalty zones.
* **Modern Python 3.12+ Standards:** Strict type annotations via `numpy.typing.NDArray[np.float64]`, modern type unions (`|`), explicit axis selections, zero-copy `.to_numpy(dtype=np.float64, copy=False)` operations, and clean `src/` layout installation.

---

## System Architecture

```text
VaR-Simulation-Stress-Test/
├── pyproject.toml                     # PEP 517/518 build specification
├── main.py                            # End-to-end execution pipeline
├── src/
│   └── stress_engine/
│       ├── __init__.py                # Package initialization & exports
│       ├── backtest.py                # Kupiec, Christoffersen, & Basel III engines
│       ├── ingestion.py               # Vectorized HTML/distressed patch parsing
│       ├── monte_carlo.py             # 4D tensor stress testing engine
│       ├── portfolio.py               # PortfolioVaR, rolling OOS, & FHS engines
│       ├── visualization.py           # Publication-grade diagnostic charting
│       └── volatility.py              # EWMA and GJR-GARCH(1,1) MLE solvers
├── tests/
│   └── test_stress_engine.py          # 15 mathematical & structural invariants
├── data/
│   ├── patches/                       # Standardized CSV patches
│   ├── plots/                         # 300 DPI exported diagnostics
│   └── raw/                           # Parquet metrics & master numpy tensors
└── research/
    └── archive/                       # Deprecated scratchpads & experiments
```
Mathematical Specification1. Dynamic Conditional VolatilityRiskMetrics EWMA Filter$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_{t-1}^2 \quad (\lambda = 0.94)$$
