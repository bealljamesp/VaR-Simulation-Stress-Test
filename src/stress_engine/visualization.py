from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t

# Set aesthetic styling for professional presentation
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    }
)


def generate_parameter_grid() -> dict[str, dict[str, float]]:
    vol_levels = {"Low": 0.10, "Norm": 0.20, "High": 0.35}
    df_levels = {"Fat": 3.5, "Norm": 8.0, "Thin": 30.0}
    gamma_levels = {"Low": 0.01, "Norm": 0.05, "High": 0.15}

    grid: dict[str, dict[str, float]] = {}
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


def run_simulation_paths(
    params: dict[str, float],
    n_paths: int = 5000,
    n_days: int = 90,
    initial_price: float = 100.0,
) -> np.ndarray:
    dt: float = 1.0 / 252.0
    daily_vol: float = params["annual_vol"] / np.sqrt(252.0)
    df: float = params["t_df"]

    # Vectorized Student-t innovations across all paths and days
    raw_innovations = t.rvs(df=df, size=(n_paths, n_days))
    scale_factor = np.sqrt(df / (df - 2.0)) if df > 2.0 else 1.0
    z = raw_innovations / scale_factor

    daily_rets = -0.5 * (daily_vol**2) * dt + daily_vol * np.sqrt(dt) * z
    log_rets = np.hstack([np.zeros((n_paths, 1)), np.cumsum(daily_rets, axis=1)])
    price_paths = initial_price * np.exp(log_rets)

    # Calculate peak-to-trough maximum drawdown for every path
    rolling_max = np.maximum.accumulate(price_paths, axis=1)
    drawdowns = (price_paths - rolling_max) / rolling_max
    return np.min(drawdowns, axis=1)


def build_visualization_dataset() -> pd.DataFrame:
    grid = generate_parameter_grid()
    records: list[dict[str, float | str]] = []

    print("Executing simulation sweeps for visualization...")
    for inst_name, profile in grid.items():
        max_drawdowns = run_simulation_paths(profile, n_paths=3000, n_days=90)
        for dd in max_drawdowns:
            records.append(
                {
                    "Institution": inst_name,
                    "Max_Drawdown": dd,
                    "Volatility": profile["Vol_Tier"],
                    "Tail_Thickness": profile["Tail_Tier"],
                    "Asymmetry": profile["Asym_Tier"],
                }
            )

    return pd.DataFrame(records)


def plot_drawdown_distributions(df_viz: pd.DataFrame) -> None:
    # Create a FacetGrid comparing Volatility Tiers across Tail Thicknesses
    g = sns.displot(
        df_viz,
        x="Max_Drawdown",
        hue="Volatility",
        col="Tail_Thickness",
        kind="kde",
        fill=True,
        common_norm=False,
        alpha=0.3,
        palette="tab10",
        height=4,
        aspect=1.1,
    )

    # Add a vertical threshold line at -40% distress boundary
    for ax in g.axes.flat:
        ax.axvline(
            x=-0.40,
            color="firebrick",
            linestyle="--",
            linewidth=1.5,
            label="40% Distress Threshold",
        )
        ax.set_xlabel("Maximum Peak-to-Trough Drawdown")
        ax.set_ylabel("Density")

    g.set_titles(col_template="Tail Tier: {col_name}")
    g.add_legend(title="Vol Regime")
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle(
        "Synthetic Institution Drawdown Distributions (3x3x3 Grid Analysis)",
        fontsize=14,
        fontweight="bold",
    )

    output_plot_path = "monte_carlo_drawdown_distributions.png"
    plt.savefig(output_plot_path, bbox_inches="tight")
    print(f"Visualization successfully saved to disk as '{output_plot_path}'")
    plt.close()


if __name__ == "__main__":
    df_simulation_data = build_visualization_dataset()
    plot_drawdown_distributions(df_simulation_data)
