"""Monte Carlo Stress Testing Engine (4D Parametric Grid)."""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd


def generate_4d_parameter_grid() -> dict[str, dict[str, float | str]]:
    vol_tiers = {"Vol-Low": 0.10, "Vol-Norm": 0.20, "Vol-High": 0.35}
    tail_tiers = {"Tail-Fat": 3.5, "Tail-Norm": 8.0, "Tail-Thin": 30.0}
    asym_tiers = {"Asym-Low": 0.01, "Asym-Norm": 0.05, "Asym-High": 0.15}
    shock_tiers = {
        "Shock-Mild": -0.15,
        "Shock-Mod": -0.30,
        "Shock-Sev": -0.50,
    }

    grid = {}
    for v_name, v_val, t_name, t_val, a_name, a_val, s_name, s_val in itertools.product(
        vol_tiers.keys(),
        vol_tiers.values(),
        tail_tiers.keys(),
        tail_tiers.values(),
        asym_tiers.keys(),
        asym_tiers.values(),
        shock_tiers.keys(),
        shock_tiers.values(),
    ):
        key = f"{v_name}_{t_name}_{a_name}_{s_name}"
        grid[key] = {
            "annual_vol": v_val,
            "Vol_Tier": v_name,
            "t_df": t_val,
            "Tail_Tier": t_name,
            "gjr_gamma": a_val,
            "Asym_Tier": a_name,
            "base_shock": s_val,
            "Shock_Tier": s_name,
        }
    return grid


def run_comprehensive_stress_engine(
    n_paths: int = 5000, n_days: int = 90
) -> pd.DataFrame:
    grid = generate_4d_parameter_grid()
    records = []

    print(
        f"Running 4D Stress Engine across {len(grid)} nodes ({n_paths} paths"
        " each)..."
    )
    start_time = time.time()

    for inst_name, profile in grid.items():
        daily_vol = float(profile["annual_vol"]) / np.sqrt(252)
        t_df = float(profile["t_df"])
        gjr_gamma = float(profile["gjr_gamma"])
        base_shock = float(profile["base_shock"])

        # Vectorized Student-t generation
        innovations = np.random.standard_t(df=t_df, size=(n_paths, n_days))
        returns_matrix = daily_vol * innovations

        # Scaled shock calculation: explicitly couples tail weight and asymmetry
        # so that vulnerable profiles experience amplified drawdowns.
        shock_day = 30
        if shock_day < n_days:
            scaled_shock = base_shock * (1.0 + gjr_gamma * 10.0) * (6.0 / t_df)
            returns_matrix[:, shock_day] += scaled_shock

        price_paths = 100 * np.exp(np.cumsum(returns_matrix, axis=1))
        running_max = np.maximum.accumulate(price_paths, axis=1)
        drawdowns = (price_paths - running_max) / running_max
        max_dds = np.min(drawdowns, axis=1)

        records.append(
            {
                "Institution_Key": inst_name,
                "Volatility": profile["Vol_Tier"],
                "Tail_Tier": profile["Tail_Tier"],
                "Asymmetry": profile["Asym_Tier"],
                "Shock_Severity": profile["Shock_Tier"],
                "Mean_Max_DD": float(np.mean(max_dds)),
                "P95_Max_DD": float(np.percentile(max_dds, 5)),
                "Distress_Probability": float(np.mean(max_dds <= -0.40)),
            }
        )

    df_results = pd.DataFrame(records)

    # Save artifact back to data/raw/
    output_dir = Path("../data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_4d_monte_carlo_results.parquet"
    df_results.to_parquet(output_path)

    print(
        f"Simulation complete in {time.time() - start_time:.2f}s. Saved to"
        f" {output_path}"
    )
    return df_results


if __name__ == "__main__":
    run_comprehensive_stress_engine()
