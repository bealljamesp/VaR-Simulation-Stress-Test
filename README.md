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

Mathematical Specification1. Dynamic Conditional VolatilityRiskMetrics EWMA Filter$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_{t-1}^2 \quad (\lambda = 0.94)$$Asymmetric GJR-GARCH(1,1)$$\sigma_t^2 = \omega + \left( \alpha + \gamma \cdot \mathbb{I}_{\{r_{t-1} < 0\}} \right) r_{t-1}^2 + \beta \sigma_{t-1}^2$$Subject to parameter constraints:$$\omega > 0, \quad \alpha \ge 0, \quad \gamma \ge 0, \quad \beta \ge 0, \quad \alpha + \beta + \frac{1}{2}\gamma < 1.0$$Filtered Historical Simulation (FHS)Standardized innovations are filtered from conditional volatility:$$z_t = \frac{r_t}{\sigma_t}$$The dynamic out-of-sample loss cutoff is computed via the empirical quantile of historical residuals:$$\text{VaR}_t^{\text{FHS}}(\alpha) = \text{Quantile}_\alpha(\{z_\tau\}_{\tau=t-W}^{t-1}) \cdot \sigma_t$$2. Regulatory Backtesting FrameworkKupiec Unconditional Coverage Test ($LR_{\text{uc}}$)Evaluates whether the empirical failure rate $\hat{p} = x / N$ statistically diverges from target $p = 1 - \alpha$:$$LR_{\text{uc}} = -2 \ln \left[ \frac{(1 - p)^{N-x} p^x}{(1 - \hat{p})^{N-x} \hat{p}^x} \right] \sim \chi^2(1)$$Christoffersen Independence Test ($LR_{\text{ind}}$)Evaluates exception clustering using a first-order Markov chain with transition probabilities $\pi_{ij} = P(I_t = j \mid I_{t-1} = i)$:$$LR_{\text{ind}} = -2 \ln \left[ \frac{L(\hat{\Pi}_1)}{L(\hat{\Pi}_2)} \right] \sim \chi^2(1)$$where:$$L(\hat{\Pi}_1) = (1 - \pi)^{T_{00} + T_{10}} \pi^{T_{01} + T_{11}}, \quad L(\hat{\Pi}_2) = (1 - \pi_{01})^{T_{00}} \pi_{01}^{T_{01}} (1 - \pi_{11})^{T_{10}} \pi_{11}^{T_{11}}$$Combined Conditional Coverage Test ($LR_{\text{cc}}$)$$LR_{\text{cc}} = LR_{\text{uc}} + LR_{\text{ind}} \sim \chi^2(2)$$3. Coherent Risk Measures: Expected Shortfall (CVaR)To resolve the non-subadditivity of VaR ($\text{VaR}_\alpha(X+Y) \not\le \text{VaR}_\alpha(X) + \text{VaR}_\alpha(Y)$), the engine computes Expected Shortfall:$$\text{ES}_\alpha = \mathbb{E}\left[ -r_t \mid -r_t > \text{VaR}_\alpha \right]$$Empirical Benchmark Findings (2008 Crisis Stress Horizon)Tested across $N = 502$ out-of-sample trading days ($W = 252$ lookback) during the 2007–2009 Global Financial Crisis at $\alpha = 0.95$ ($p = 0.05$, Target Breaches $= 25.1$):Portfolio RegimeRisk Model EngineBreaches / ObsEmpirical %Kupiec POF (LRuc​)Christoffersen (LRind​)Regulatory VerdictFinancials (Sector Meltdown)Rolling Historical (252d)46 / 5029.16%$14.861\ (p < 0.001)$$0.469\ (p = 0.493)$Reject UnconditionalRolling Parametric (Normal)44 / 5028.76%$12.355\ (p < 0.001)$$0.248\ (p = 0.618)$Reject UnconditionalDynamic EWMA ($\lambda=0.94$)34 / 5026.77%$3.005\ (p = 0.083)$$1.053\ (p = 0.305)$Accept BothDynamic GJR-GARCH(1,1)34 / 5026.77%$3.005\ (p = 0.083)$$1.207\ (p = 0.272)$Accept BothFiltered Historical (FHS)37 / 5027.37%$5.215\ (p = 0.022)$$0.030\ (p = 0.863)$Reject UnconditionalMulti-Asset (Flight to Safety)Rolling Historical (252d)29 / 5025.78%$0.609\ (p = 0.435)$$12.102\ (p < 0.001)$Reject IndependenceRolling Parametric (Normal)31 / 5026.18%$1.363\ (p = 0.243)$$7.091\ (p = 0.008)$Reject IndependenceDynamic EWMA ($\lambda=0.94$)26 / 5025.18%$0.034\ (p = 0.855)$$0.368\ (p = 0.544)$Accept BothDynamic GJR-GARCH(1,1)39 / 5027.77%$6.983\ (p = 0.008)$$0.001\ (p = 0.979)$Reject UnconditionalFiltered Historical (FHS)30 / 5025.98%$0.950\ (p = 0.330)$$0.043\ (p = 0.835)$Accept BothTakeawaysThe In-Sample Quantile Fallacy: In-sample static historical simulation guarantees an exact $5.00\%$ breach rate by mathematical construction, creating a false sense of security while masking severe temporal shock clustering.Dynamic Volatility Responsiveness: Unweighted rolling windows lag violent market shifts. Dynamic EWMA and GJR-GARCH rapidly scale conditional variance upward, mitigating breach clustering during systemic market panics.Distributional Breakdown in Multi-Asset Portfolios: Standard GJR-GARCH under normal innovations failed on multi-asset portfolios ($7.77\%$ breach rate, $p = 0.0082$) due to tail thinness. Filtered Historical Simulation resolved the failure ($5.98\%$ breach rate, $p = 0.3296$) by combining GJR-GARCH variance scaling with empirical innovation tails.4D Tensor Monte Carlo Simulation ArchitectureThe macroeconomic stress module simulates an 81-node Cartesian parameter space across:Annualized Volatility Tiers ($\sigma_{\text{ann}}$): 10%, 20%, 35%Student-$t$ Tail Thickness ($\nu$): 3.5 (Fat), 8.0 (Normal), 30.0 (Thin)GJR Asymmetric Leverage ($\gamma$): 0.01, 0.05, 0.15Exogenous Macro Shocks ($s$): -15%, -30%, -50% at $t = 30$Memory & Execution ProfileTensor Dimensions: $(K=81\text{ nodes}) \times (P=5,000\text{ paths}) \times (T=90\text{ days})$Total Points Evaluated: $36,450,000$ single-precision float32 values.Contiguous Memory Footprint: $\approx 145.8\text{ MB}$.Execution Latency: $< 3.0\text{ seconds}$ on modern multi-core hardware via broadcasted SIMD vectorization.$$\mathbf{S}_{k, p, t} = 100 \cdot \exp\left( \sum_{\tau=1}^t \mathbf{R}_{k, p, \tau} \right)$$$$\mathbf{MDD}_{k, p} = \min_{t \in [1, T]} \left( \frac{\mathbf{S}_{k, p, t} - \max_{\tau \le t} \mathbf{S}_{k, p, \tau}}{\max_{\tau \le t} \mathbf{S}_{k, p, \tau}} \right)$$Diagnostic FiguresThe visualization pipeline produces publication-quality figures formatted with matplotlib and seaborn:ArtifactPathDescriptionVaR Backtest Diagnosticsdata/plots/var_diagnostics_*.pngTime-series overlay of daily returns, time-varying VaR boundary bands, and exception breach clusters.4D Stress Surface Matrixdata/plots/stress_matrix_4d_surface.pngFaceted categorical surface contrasting shock severity vs. distress probability ($DD \le -40\%$).Getting Started1. Environment SetupBash# Clone the repository
git clone [https://github.com/your-username/VaR-Simulation-Stress-Test.git](https://github.com/your-username/VaR-Simulation-Stress-Test.git)
cd VaR-Simulation-Stress-Test

# Create and activate Conda environment
conda create -n var-simulation python=3.12 -y
conda activate var-simulation

# Install development dependencies
conda install pytest -y
pip install -e .
2. ExecutionRun the complete econometric analysis, backtesting suite, tensor simulation, and visualization generator:Bashpython main.py
To run the standalone 4D Monte Carlo tensor stress simulation:Bashpython src/stress_engine/monte_carlo.py
3. VerificationExecute the test suite verifying all 15 mathematical and behavioral invariants:Bashpytest -v
Technical Invariants TestedThe test suite in tests/test_stress_engine.py asserts:Kupiec POF Bounds: Perfect calibration non-rejection vs. severe underestimation failure.Christoffersen Markov Transitions: Clustering detection ($p < 0.05$) vs. boundary conditions.EWMA Monotonic Decay: Exponential variance attenuation under post-shock zero-return regimes.GJR-GARCH Stationarity: Non-negativity constraints ($\omega, \alpha, \gamma, \beta > 0$) and persistence boundedness ($\alpha + \beta + 0.5\gamma < 1.0$).Expected Shortfall Coherence: Structural guarantee that $\text{ES}_\alpha > \text{VaR}_\alpha$.BCBS Basel III Boundaries: Exact binomial zone assignment (Green $\le 4$, Yellow $5\text{--}9$, Red $\ge 10$).Tensor Geometry: Bounded probabilities $[0.0, 1.0]$ and strict dimensional integrity across $(81, 5000, 90)$.AuthorJames BeallM.S. Applied Business Analytics | B.S. Theoretical Mathematics (UCSB)Quantitative Risk Management & Quantitative Finance Portfolio Project
