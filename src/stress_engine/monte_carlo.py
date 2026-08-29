"""Monte Carlo Stress Testing Engine (4D Parametric Grid).

Consolidates baseline simulation and shock-testing architecture into a single,
future-proof module leveraging explicit vectorization and NumPy linear algebra.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd


def generate_4d_parameter_grid() -> dict[str, dict[str, float | str]]:
    """Generates the 81-node 4D parametric stress grid."""
    vol_tiers = {
        "Vol-Low": 0.10,
        "Vol-Norm": 0.20,
        "Vol-High": 0.35,
    }
    tail_tiers = {
        "Tail-Fat": 3.5,
        "Tail-Norm": 8.0,
        "Tail-Thin": 30.0,
    }
    asym_tiers = {
        "Asym-Low": 0.01,
        "Asym-Norm": 0.05,
        "Asym-High": 0.15,
    }
    shock_tiers = {
        "Shock-Mild": -0.15,
        "Shock-Mod": -0.30,
        "Shock-Sev": -0.50,
    }

    grid: dict[str, dict[str, float | str]] = {}
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
    """Executes the 4D Monte Carlo simulation engine across all 81 nodes."""
    grid = generate_4d_parameter_grid()
    records: list[dict[str, float | str]] = []

    print(
        f"Initializing Unified Stress Engine: Evaluating {len(grid)} structural"
        f" nodes ({n_paths} paths each)..."
    )
    start_time = time.time()

    for inst_name, profile in grid.items():
        daily_vol = float(profile["annual_vol"]) / np.sqrt(252)
        t_df = float(profile["t_df"])
        gjr_gamma = float(profile["gjr_gamma"])
        base_shock = float(profile["base_shock"])

        # Vectorized path generation block
        # Generate Student-t innovations for all paths simultaneously: shape (n_paths, n_days)
        innovations = np.random.standard_t(df=t_df, size=(n_paths, n_days))
        returns_matrix = daily_vol * innovations

        # Inject Day-30 systemic shock scaled dynamically by asymmetry and tail thickness
        shock_day = 30
        if shock_day < n_days:
            scaled_shock = base_shock * (1.0 + gjr_gamma * 3.0) * (5.0 / t_df)
            returns_matrix[:, shock_day] += scaled_shock

        # Convert returns to cumulative price paths starting at base 100
        price_paths = 100 * np.exp(np.cumsum(returns_matrix, axis=1))

        # Calculate peak-to-trough drawdowns across paths
        running_max = np.maximum.accumulate(price_paths, axis=1)
        drawdowns = (price_paths - running_max) / running_max
        max_dds = np.min(drawdowns, axis=1)

        # Aggregate outcome metrics
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

    # Ensure output directory exists and save parquet artifact
    output_dir = Path("../data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_4d_monte_carlo_results.parquet"
    df_results.to_parquet(output_path)

    elapsed = time.time() - start_time
    print(
        f"Simulation complete in {elapsed:.2f} seconds. Results saved to"
        f" '{output_path}'."
    )
    return df_results


if __name__ == "__main__":
    df_output = run_comprehensive_stress_engine(n_paths=5000, n_days=90)
    print(df_output.head())
