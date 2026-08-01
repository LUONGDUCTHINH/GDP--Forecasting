import json
from pathlib import Path


NOTEBOOK_PATH = Path("/Users/tonytony/Final Project/Analysis/gdp_final_model_leaderboard.ipynb")

NEW_SOURCE = """# Figure 6.2 — Train + holdout paths for the best algorithm within each main GDP specification
# This version addresses the final-report note:
# - training years are shown first
# - the holdout period is separated visually
# - all panels use a common y-scale for cleaner comparison

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("/Users/tonytony/Final Project/output/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

context_path = Path(
    "/Users/tonytony/Final Project/Data/Cleaned/gdp_all_models_train_test_context_yearly_summary.csv"
)
context_df = pd.read_csv(context_path)

context_df["plot_year"] = pd.to_numeric(context_df["plot_year"], errors="coerce")
context_df["mean_gdp"] = pd.to_numeric(context_df["mean_gdp"], errors="coerce")
context_df = context_df.dropna(subset=["plot_year", "mean_gdp"]).copy()
context_df["plot_year"] = context_df["plot_year"].astype(int)

spec_order = sorted(best_by_spec_df["spec"].tolist(), key=lambda x: int(x.split()[-1]))
best_algo_map = dict(zip(best_by_spec_df["spec"], best_by_spec_df["algorithm"]))

train_end_year = int(
    context_df.loc[context_df["segment"].str.contains("train", na=False), "plot_year"].max()
)
test_start_year = int(
    context_df.loc[context_df["segment"].str.contains("test", na=False), "plot_year"].min()
)
x_min = int(context_df["plot_year"].min())
x_max = int(context_df["plot_year"].max())

all_values = []
for spec_label in spec_order:
    best_algo = best_algo_map[spec_label]
    spec_slice = context_df[
        (context_df["spec"] == spec_label)
        & (context_df["algorithm"].isin(["Actual", best_algo]))
    ].copy()
    all_values.extend(spec_slice["mean_gdp"].tolist())

y_min = min(all_values)
y_max = max(all_values)
y_pad = max((y_max - y_min) * 0.08, 250)

fig, axes = plt.subplots(
    len(spec_order),
    1,
    figsize=(14.5, 4.8 * len(spec_order)),
    sharex=True,
    sharey=True
)

if len(spec_order) == 1:
    axes = [axes]

for ax, spec_label in zip(axes, spec_order):
    best_algo = best_algo_map[spec_label]

    actual_spec = context_df[
        (context_df["spec"] == spec_label)
        & (context_df["algorithm"] == "Actual")
    ].copy()
    actual_spec["segment_order"] = actual_spec["segment"].map(
        {"train_actual": 0, "test_actual": 1}
    )
    actual_spec = actual_spec.sort_values(["plot_year", "segment_order"])

    pred_spec = context_df[
        (context_df["spec"] == spec_label)
        & (context_df["algorithm"] == best_algo)
    ].copy()
    pred_spec["segment_order"] = pred_spec["segment"].map(
        {"train_prediction": 0, "test_prediction": 1}
    )
    pred_spec = pred_spec.sort_values(["plot_year", "segment_order"])

    ax.axvspan(
        x_min - 0.5,
        train_end_year + 0.5,
        color="#eef4ff",
        alpha=0.95,
        zorder=0
    )
    ax.axvspan(
        test_start_year - 0.5,
        x_max + 0.5,
        color="#fff6e8",
        alpha=0.95,
        zorder=0
    )
    ax.axvline(
        test_start_year - 0.5,
        color="#7d8ca3",
        linestyle=":",
        linewidth=1.6,
        zorder=1
    )

    ax.plot(
        actual_spec["plot_year"],
        actual_spec["mean_gdp"],
        marker="o",
        markersize=4.2,
        linewidth=2.5,
        color="black",
        label="Actual GDP per capita"
    )

    ax.plot(
        pred_spec["plot_year"],
        pred_spec["mean_gdp"],
        marker="o",
        markersize=3.8,
        linestyle="--",
        linewidth=2.3,
        color="#4f7cac",
        label=f"Predicted GDP per capita ({best_algo})"
    )

    ax.set_title(
        f"{spec_label}: train-to-holdout path using the best algorithm ({best_algo})",
        fontsize=13,
        fontweight="bold"
    )
    ax.set_ylabel("Mean GDP per Capita (US$)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.20)
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.legend(frameon=False, loc="upper left")

    ax.text(
        0.02,
        0.93,
        "Train period",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#48627a",
        fontweight="bold"
    )
    ax.text(
        0.84,
        0.93,
        "Holdout period",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#b8742b",
        fontweight="bold"
    )

    year_ticks = list(range(x_min, x_max + 1, 2))
    if x_max not in year_ticks:
        year_ticks.append(x_max)
    ax.set_xticks(sorted(set(year_ticks)))

axes[-1].set_xlabel("Year", fontsize=11)

fig.suptitle(
    "Figure 6.2. Actual and best-predicted GDP paths across the train and shared holdout periods",
    fontsize=15,
    fontweight="bold",
    y=0.995
)
fig.text(
    0.5,
    0.972,
    "Training years are shown first, followed by the shared holdout period. All panels use a common vertical scale for direct comparison.",
    ha="center",
    va="top",
    fontsize=10
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

save_path = OUTPUT_DIR / "figure_6_2_best_by_spec_holdout_paths.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", save_path)
display(best_by_spec_df[["spec", "algorithm", "RMSE", "MAPE_pct", "MAE", "R_squared"]])
"""


def main():
    nb = json.loads(NOTEBOOK_PATH.read_text())
    replaced = False

    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        if source.startswith("# Figure 6.2 — Best holdout prediction path within each main GDP specification"):
            cell["source"] = NEW_SOURCE.splitlines(keepends=True)
            replaced = True
            break

    if not replaced:
        raise RuntimeError("Target Figure 6.2 cell was not found.")

    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"Updated notebook cell in: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
