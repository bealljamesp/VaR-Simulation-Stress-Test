from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t


def generate_parameter_grid() -> dict[str, dict[str, float | str]]:
    vol_levels = {"Low": 0.10, "Norm": 0.20, "High": 0.35}
    df_levels = {"Fat": 3.5, "Norm": 8.0, "Thin": 30.0}
    gamma_levels = {"Low": 0.01, "Norm": 0.05, "High": 0.15}

    grid: dict[str, dict[str, float | str]] = {}
    for v_key, v_val in vol_levels.items():
        for d_key, d_val in df_levels.items():
            for g_key, g_val in gamma_levels.items():
                name = f"Vol-{v_key}_Tail-{d_key}_Asym-{g_key}"
                grid[name] = {
                    "annual_vol": v_val,
                    "t_df": d_val,
                    "gjr_gamma": g_val,
                    "Vol_Tier": v_key,
                    "Tail_Tier": d_key,
                    "Asym_Tier": g_key,
                }
    return grid


# def run_shock_monte_carlo(
#     params: dict[str, float | str],
#     n_paths: int = 5000,
#     n_days: int = 90,
#     initial_price: float = 100.0,
#     distress_threshold: float = -0.40,
# ) -> tuple[float, np.ndarray]:
#     dt: float = 1.0 / 252.0
#     daily_vol: float = float(params["annual_vol"]) / np.sqrt(252.0)
#     df: float = float(params["t_df"])
#     gamma: float = float(params["gjr_gamma"])

#     raw_innovations = t.rvs(df=df, size=(n_paths, n_days))
#     scale_factor = np.sqrt(df / (df - 2.0)) if df > 2.0 else 1.0
#     z = raw_innovations / scale_factor

#     # Inject exogenous market shock on Day 30
#     shock_day = 30
#     if n_days > shock_day:
#         shock_magnitude = -0.15 * (1.0 + gamma * 5.0)
#         z[:, shock_day] = shock_magnitude / (daily_vol * np.sqrt(dt))

#     daily_rets = -0.5 * (daily_vol**2) * dt + daily_vol * np.sqrt(dt) * z
#     log_rets = np.hstack([np.zeros((n_paths, 1)), np.cumsum(daily_rets, axis=1)])
#     price_paths = initial_price * np.exp(log_rets)

#     rolling_max = np.maximum.accumulate(price_paths, axis=1)
#     drawdowns = (price_paths - rolling_max) / rolling_max
#     max_drawdowns = np.min(drawdowns, axis=1)

#     breach_count = np.sum(max_drawdowns <= distress_threshold)
#     return float(breach_count / n_paths), max_drawdowns


def run_shock_monte_carlo(
    params: dict[str, float | str],
    n_paths: int = 3000,
    n_days: int = 90,
    initial_price: float = 100.0,
    distress_threshold: float = -0.40,
) -> tuple[float, np.ndarray]:
    dt: float = 1.0 / 252.0
    daily_vol: float = float(params["annual_vol"]) / np.sqrt(252.0)
    df: float = float(params["t_df"])
    gamma: float = float(params["gjr_gamma"])

    raw_innovations = t.rvs(df=df, size=(n_paths, n_days))
    scale_factor = np.sqrt(df / (df - 2.0)) if df > 2.0 else 1.0
    z = raw_innovations / scale_factor

    # Inject a much more severe systemic shock on Day 30
    # Scaled dynamically by volatility tier, tail thickness (lower df = heavier tail), and asymmetry

    # 1st attempt at a shock scenario, commented out for now. Results were 24%-15% shock simulation compared to 1.5%-0.4% non-shock simulation. This is a significant increase in the probability of distress, indicating that the shock scenario has a substantial impact on the risk profile of the institutions being analyzed.

    # shock_day = 30
    # if n_days > shock_day:
    #     tail_multiplier = (
    #         3.0 / df
    #     )  # Heavier tails (e.g., df=3.5) amplify the shock impact
    #     shock_magnitude = -0.28 * (1.0 + gamma * 10.0) * tail_multiplier
    #     z[:, shock_day] = shock_magnitude / (daily_vol * np.sqrt(dt))

    # Increase base shock severity and tail sensitivity
    shock_day = 30
    if n_days > shock_day:
        tail_multiplier = 4.0 / df
        shock_magnitude = -0.42 * (1.0 + gamma * 15.0) * tail_multiplier
        z[:, shock_day] = shock_magnitude / (daily_vol * np.sqrt(dt))

        daily_rets = -0.5 * (daily_vol**2) * dt + daily_vol * np.sqrt(dt) * z
        log_rets = np.hstack([np.zeros((n_paths, 1)), np.cumsum(daily_rets, axis=1)])
        price_paths = initial_price * np.exp(log_rets)

        rolling_max = np.maximum.accumulate(price_paths, axis=1)
        drawdowns = (price_paths - rolling_max) / rolling_max
        max_drawdowns = np.min(drawdowns, axis=1)

        breach_count = np.sum(max_drawdowns <= distress_threshold)
        return float(breach_count / n_paths), max_drawdowns


if __name__ == "__main__":
    grid = generate_parameter_grid()
    ledger: list[dict[str, float | str]] = []

    print("Executing Shock-Enabled Monte Carlo Stress Tests...")
    for inst_name, profile in grid.items():
        prob, max_dds = run_shock_monte_carlo(profile, n_paths=3000, n_days=90)
        ledger.append(
            {
                "Institution": inst_name,
                "Annual_Vol": profile["annual_vol"],
                "Tail_DF": profile["t_df"],
                "GJR_Gamma": profile["gjr_gamma"],
                "Distress_Probability": prob,
                "Mean_Max_DD": float(np.mean(max_dds)),
                "P95_Max_DD": float(np.percentile(max_dds, 5)),
            }
        )

    df_out = pd.DataFrame(ledger)

    # Resolve path to data/raw/ relative to the repository layout
    output_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "synthetic_monte_carlo_shock_results.parquet"

    df_out.to_parquet(output_file, index=False)
    print(f"Shock simulation results successfully saved to: {output_file}")
