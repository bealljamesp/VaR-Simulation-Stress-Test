# Monte Carlo simulation engine for stress testing

from __future__ import annotations

import numpy as np
from scipy.stats import t


def generate_parameter_grid() -> dict[str, dict[str, float]]:
    """Generates a complete 3x3x3 (27 variations) synthetic institution dictionary

    across Volatility, Tail Thickness (Student's t df), and GJR-GARCH Asymmetry.
    """
    vol_levels = {"Low": 0.10, "Norm": 0.20, "High": 0.35}
    df_levels = {"Fat": 3.0, "Norm": 10.0, "Thin": 30.0}  # lower df = heavier tails
    gamma_levels = {"Low": 0.01, "Norm": 0.05, "High": 0.15}

    grid: dict[str, dict[str, float]] = {}
    for v_key, v_val in vol_levels.items():
        for d_key, d_val in df_levels.items():
            for g_key, g_val in gamma_levels.items():
                name = f"Inst_Vol_{v_key}_Tail_{d_key}_Asym_{g_key}"
                grid[name] = {
                    "annual_vol": v_val,
                    "t_df": d_val,
                    "gjr_gamma": g_val,
                }
    return grid


def run_monte_carlo_simulation(
    params: dict[str, float],
    n_paths: int = 10000,
    n_days: int = 120,
    initial_price: float = 100.0,
    distress_threshold: float = -0.40,
) -> float:
    """Vectorized Monte Carlo path generator calculating peak-to-trough drawdowns

    without explicit Python loops over time steps.
    """
    dt: float = 1.0 / 252.0
    daily_vol: float = params["annual_vol"] / np.sqrt(252.0)
    df: float = params["t_df"]

    # Draw student-t standardized innovations across all paths and days simultaneously
    # Shape: (n_paths, n_days)
    raw_innovations = t.rvs(df=df, size=(n_paths, n_days))
    scale_factor = np.sqrt(df / (df - 2.0)) if df > 2.0 else 1.0
    z = raw_innovations / scale_factor

    # Vectorized log-returns approximation for geometric Brownian path generation
    daily_rets = -0.5 * (daily_vol**2) * dt + daily_vol * np.sqrt(dt) * z

    # Reconstruct price paths via cumulative product along the time axis (axis=1)
    log_rets = np.hstack([np.zeros((n_paths, 1)), np.cumsum(daily_rets, axis=1)])
    price_paths = initial_price * np.exp(log_rets)

    # Vectorized peak-to-trough maximum drawdown calculation
    rolling_max = np.maximum.accumulate(price_paths, axis=1)
    drawdowns = (price_paths - rolling_max) / rolling_max
    max_drawdowns = np.min(drawdowns, axis=1)

    # Calculate breach probability
    breach_count = np.sum(max_drawdowns <= distress_threshold)
    return float(breach_count / n_paths)


# Execution pipeline for the 27-institution dictionary
if __name__ == "__main__":
    synthetic_institutions = generate_parameter_grid()
    print(f"Total Synthetic Institutions Generated: {len(synthetic_institutions)}")
    print("-" * 60)

    for inst_name, profile in list(synthetic_institutions.items())[
        :5
    ]:  # Sample preview
        prob = run_monte_carlo_simulation(profile, n_paths=5000, n_days=30)
        print(f"{inst_name} --> Distress Breach Probability: {prob:.4%}")
