# %% [code]
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
    }
)

df_results = pd.read_parquet("../data/raw/synthetic_4d_monte_carlo_results.parquet")

# Ensure correct ordering of categorical tiers for clean visualization
shock_order = ["Shock-Mild", "Shock-Mod", "Shock-Sev"]
df_results["Shock_Severity"] = pd.Categorical(
    df_results["Shock_Severity"], categories=shock_order, ordered=True
)

# Build a clean 3x3 faceted bar chart where X=Shock, Hue=Asymmetry, Col=Tail, Row=Vol
g = sns.catplot(
    data=df_results,
    x="Shock_Severity",
    y="Distress_Probability",
    hue="Asymmetry",
    col="Tail_Tier",
    row="Volatility",
    kind="bar",
    height=3.0,
    aspect=1.2,
    palette="deep",
    sharey=True,
)

g.set_axis_labels("Shock Severity Tier", "Probability of Distress (DD <= -40%)")
g.set_titles(col_template="Tail: {col_name}", row_template="Vol: {row_name}")
g.add_legend(title="GJR Asymmetry")
g.fig.suptitle(
    "4D Stress Matrix: Full Parametric Breakdown",
    y=1.03,
    fontsize=13,
    fontweight="bold",
)

output_path = "../data/raw/stress_matrix_4d_grid.png"
plt.savefig(output_path, bbox_inches="tight")
plt.show()
