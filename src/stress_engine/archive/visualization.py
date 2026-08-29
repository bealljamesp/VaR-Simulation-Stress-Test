from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from stress_engine.visualization import (
    generate_parameter_grid,
    run_extended_monte_carlo,
)

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


def build_visualization_dataset() -> pd.DataFrame:
    grid = generate_parameter_grid()
    records: list[dict[str, float | str]] = []

    print("Compiling maximum drawdown paths for visualization...")
    for _, profile in grid.items():
        _, max_drawdowns = run_extended_monte_carlo(profile, n_paths=2000, n_days=90)
        for dd in max_drawdowns:
            records.append(
                {
                    "Max_Drawdown": dd,
                    "Volatility": profile["Vol_Tier"],
                    "Tail_Thickness": profile["Tail_Tier"],
                    "Asymmetry": profile["Asym_Tier"],
                }
            )

    return pd.DataFrame(records)


def plot_drawdown_distributions(df_viz: pd.DataFrame) -> None:
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

    output_path = "monte_carlo_drawdown_distributions.png"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Visualization successfully saved to '{output_path}'")
    plt.close()


if __name__ == "__main__":
    df_data = build_visualization_dataset()
    plot_drawdown_distributions(df_data)
