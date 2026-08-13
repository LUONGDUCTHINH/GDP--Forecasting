from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BASE_DIR = Path("/Users/tonytony/Final Project")
OUTPUT_DIR = BASE_DIR / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "figure_3_1_project_pipeline_clean.png"


plt.rcParams["font.family"] = "DejaVu Sans"


def add_box(ax, x, y, w, h, label):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        linewidth=1.0,
        edgecolor="#4b5563",
        facecolor="#424242",
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        fontweight="medium",
    )


def add_arrow(ax, start, end, rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.4,
        color="#6b7280",
        shrinkA=2,
        shrinkB=2,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)


fig, ax = plt.subplots(figsize=(18.5, 4.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 1.20)
ax.set_ylim(0, 1)
ax.axis("off")


# Row-1 pipeline
top_y = 0.52
box_h = 0.11
top_positions = {
    "raw": (0.03, 0.11),
    "clean": (0.15, 0.10),
    "meta": (0.265, 0.115),
    "panel": (0.395, 0.12),
    "eda": (0.535, 0.07),
}

labels_top = {
    "raw": "Raw Data",
    "clean": "Cleaning",
    "meta": "Metadata Merge",
    "panel": "Panel + Features",
    "eda": "EDA",
}

for key, (x, w) in top_positions.items():
    add_box(ax, x, top_y, w, box_h, labels_top[key])

top_keys = ["raw", "clean", "meta", "panel", "eda"]
for left, right in zip(top_keys[:-1], top_keys[1:]):
    x1, w1 = top_positions[left]
    x2, _ = top_positions[right]
    add_arrow(ax, (x1 + w1, top_y + box_h / 2), (x2, top_y + box_h / 2))


# Upper branch
upper_y = 0.72
ts_x, ts_w = 0.63, 0.11
fi_x, fi_w = 0.775, 0.10

add_box(ax, ts_x, upper_y, ts_w, box_h, "Time-Series Models")
add_box(ax, fi_x, upper_y, fi_w, box_h, "Future Inputs")


# Lower branch
lower_y = 0.30
main_x, main_w = 0.63, 0.11
bench_x, bench_w = 0.775, 0.12

add_box(ax, main_x, lower_y, main_w, box_h, "Main GDP Models")
add_box(ax, bench_x, lower_y, bench_w, box_h, "Benchmark + Robustness")


# Final output row
forecast_x, forecast_w = 0.90, 0.12
forecast_y = 0.51
dash_x, dash_w = 1.04, 0.13
dash_y = 0.51

add_box(ax, forecast_x, forecast_y, forecast_w, box_h, "Future Forecasting")
add_box(ax, dash_x, dash_y, dash_w, box_h, "Dashboard + Report")


# Branch arrows from EDA
eda_x, eda_w = top_positions["eda"]
add_arrow(
    ax,
    (eda_x + eda_w, top_y + box_h * 0.78),
    (ts_x, upper_y + box_h / 2),
    rad=-0.10,
)
add_arrow(
    ax,
    (eda_x + eda_w, top_y + box_h * 0.22),
    (main_x, lower_y + box_h / 2),
    rad=0.10,
)


# Branch internals
add_arrow(ax, (ts_x + ts_w, upper_y + box_h / 2), (fi_x, upper_y + box_h / 2))
add_arrow(
    ax,
    (fi_x + fi_w, upper_y + box_h / 2),
    (forecast_x, forecast_y + box_h * 0.70),
    rad=-0.18,
)

add_arrow(ax, (main_x + main_w, lower_y + box_h / 2), (bench_x, lower_y + box_h / 2))
add_arrow(
    ax,
    (bench_x + bench_w, lower_y + box_h / 2),
    (forecast_x, forecast_y + box_h * 0.30),
    rad=-0.12,
)
add_arrow(ax, (forecast_x + forecast_w, forecast_y + box_h / 2), (dash_x, dash_y + box_h / 2))


fig.suptitle(
    "Figure 3.1. End-to-End Analytical Pipeline of the Project",
    fontsize=18,
    fontweight="bold",
    y=0.97,
)
fig.text(
    0.5,
    0.90,
    "From raw World Bank data to cleaned panel construction, exploratory analysis, modelling, future forecasting, and report delivery.",
    ha="center",
    fontsize=11,
    color="#4b5563",
)

plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved: {OUTPUT_PATH}")
