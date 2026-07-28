# portfolio-stress-testing-var-simulation

An institutional-grade market risk analytics and Monte Carlo stress-testing engine built in Python. This framework evaluates the systemic underestimation of traditional parametric (Gaussian) and historical Value at Risk (VaR) models during macroeconomic shocks, isolating true structural risk factors from historical coincidences.

---

## 🎯 Overview & Objectives

Traditional linear risk models (e.g., standard normal parametric VaR) notoriously fail during severe regime shifts because they assume Gaussian asset return distributions and stable cross-asset correlations. This repository implements a two-stage hybrid quantitative methodology:

1. **Empirical Historical Crisis Analysis:** Backtests multi-asset portfolios through real-world market crashes (2008 Global Financial Crisis, 2020 COVID Shock, and 2022 Rate Hike Cycle) to quantify VaR breach magnitudes, tail kurtosis, and correlation breakdown.
2. **Stochastic Monte Carlo Simulation Engine:** Generates $10,000+$ synthetic market shock scenarios across non-Gaussian distributions (Student's $t$-distribution, GARCH dynamic volatility, dynamic copula correlations) to test whether identified risk variables act as true structural drivers or mere historical artifacts.

---

## 📐 Key Risk Metrics & Methodologies

* **Risk Metrics Evaluated:**
  * Parametric VaR (Normal vs. Heavy-Tailed Student's $t$)
  * Historical Simulation VaR
  * GARCH(1,1)-Filtered Volatility & Dynamic Conditional Correlation (DCC)
  * Expected Shortfall (ES / Tail VaR)
* **Statistical Backtesting:**
  * Kupiec Proportion of Failures (POF) Test
  * Christoffersen Interval Independence Test
  * VaR Exceedance Ratios ($\frac{\text{Actual Loss}}{\text{Estimated VaR}}$)

---

## 🛠️ Data Sources & Architecture

* **Market & Macro Data:** Yahoo Finance API (`yfinance`) and Federal Reserve Economic Data (`FRED API`).
* **Technology Stack:** Python (`numpy`, `pandas`, `scipy`, `statsmodels`, `arch`, `matplotlib`, `seaborn`).
* **Design Pattern:** Modular, object-oriented pipeline designed for automated data ingestion, backtesting, stochastic path generation, and risk report generation.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/macro-crisis-var-stress-engine.git](https://github.com/YOUR_USERNAME/macro-crisis-var-stress-engine.git)
cd macro-crisis-var-stress-engine

# Install dependencies
pip install -r requirements.txt

# Run full historical backtest and Monte Carlo simulation
python main.py
