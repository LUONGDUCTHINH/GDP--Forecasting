from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BASE_DIR = Path("/Users/tonytony/Final Project")
OUTPUT_DIR = BASE_DIR / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "figure_3_1_project_pipeline.png"


plt.rcParams["font.family"] = "DejaVu Sans"


def add_box(ax, xy, width, height, title, body, facecolor, edgecolor="#1f2937"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.6,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)

    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        x + width / 2,
        y + height * 0.35,
        body,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#334155",
        linespacing=1.35,
    )


def add_arrow(ax, start, end, color="#475569", lw=2.0, connectionstyle="arc3,rad=0"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=lw,
        color=color,
        shrinkA=6,
        shrinkB=6,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


fig, ax = plt.subplots(figsize=(16, 8.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Main top-row flow
box_w = 0.145
box_h = 0.13
y_top = 0.70
x_positions = [0.04, 0.225, 0.41, 0.595, 0.78]

add_box(
    ax,
    (x_positions[0], y_top),
    box_w,
    box_h,
    "1. Raw Data",
    "World Bank GDP,\nPopulation, Life Expectancy,\nInflation, Unemployment, Internet",
    facecolor="#eef2ff",
)
add_box(
    ax,
    (x_positions[1], y_top),
    box_w,
    box_h,
    "2. Preparation",
    "Cleaning, standardisation,\nmetadata merge,\nregion mapping",
    facecolor="#eff6ff",
)
add_box(
    ax,
    (x_positions[2], y_top),
    box_w,
    box_h,
    "3. Panel Build",
    "Country-year panel,\nlog transforms,\nlagged and event features",
    facecolor="#ecfeff",
)
add_box(
    ax,
    (x_positions[3], y_top),
    box_w,
    box_h,
    "4. EDA",
    "Coverage checks,\nregional comparison,\nrelationship analysis",
    facecolor="#f0fdf4",
)
add_box(
    ax,
    (x_positions[4], y_top),
    box_w,
    box_h,
    "5. Outputs",
    "Evaluation tables,\nreport figures,\ninteractive dashboard",
    facecolor="#fff7ed",
)

for i in range(4):
    add_arrow(
        ax,
        (x_positions[i] + box_w, y_top + box_h / 2),
        (x_positions[i + 1], y_top + box_h / 2),
    )

# Bottom-row modelling branch
lower_w = 0.185
lower_h = 0.12
y_bottom = 0.38

left_x = 0.20
mid_x = 0.435
right_x = 0.72

add_box(
    ax,
    (left_x, y_bottom),
    lower_w,
    lower_h,
    "6A. Time-Series Models",
    "GDP, population, and life expectancy\nwith rolling 10-year backtesting",
    facecolor="#ede9fe",
)
add_box(
    ax,
    (mid_x, y_bottom),
    lower_w,
    lower_h,
    "6B. Main GDP Models",
    "Model 1, Model 2, Model 3,\nplus algorithm benchmarking",
    facecolor="#fae8ff",
)
add_box(
    ax,
    (right_x, y_bottom),
    lower_w,
    lower_h,
    "7. Future Forecasting",
    "Forecast future inputs and\npredict GDP for target years",
    facecolor="#fef3c7",
)

# Connections from EDA / panel build to modelling
anchor_x = x_positions[3] + box_w / 2
add_arrow(ax, (anchor_x - 0.05, y_top), (left_x + lower_w / 2, y_bottom + lower_h))
add_arrow(ax, (anchor_x + 0.02, y_top), (mid_x + lower_w / 2, y_bottom + lower_h))

add_arrow(
    ax,
    (left_x + lower_w, y_bottom + lower_h / 2),
    (right_x, y_bottom + lower_h * 0.38),
    connectionstyle="arc3,rad=-0.22",
)
add_arrow(
    ax,
    (mid_x + lower_w, y_bottom + lower_h / 2),
    (right_x, y_bottom + lower_h / 2),
)

# Route forecasting back into outputs
add_arrow(
    ax,
    (right_x + lower_w / 2, y_bottom + lower_h),
    (x_positions[4] + box_w / 2, y_top),
)

# Section labels
ax.text(0.13, 0.86, "Inputs", fontsize=11, fontweight="bold", color="#6366f1")
ax.text(0.32, 0.86, "Preparation", fontsize=11, fontweight="bold", color="#2563eb")
ax.text(0.51, 0.86, "Engineering", fontsize=11, fontweight="bold", color="#0891b2")
ax.text(0.70, 0.86, "Analysis", fontsize=11, fontweight="bold", color="#16a34a")
ax.text(0.88, 0.86, "Delivery", fontsize=11, fontweight="bold", color="#ea580c")
ax.text(0.50, 0.585, "Modelling layer", fontsize=11, fontweight="bold", color="#7c3aed", ha="center")

# Title and subtitle
fig.suptitle(
    "Figure 3.1. End-to-End Analytical Pipeline of the Project",
    fontsize=19,
    fontweight="bold",
    y=0.96,
)
fig.text(
    0.5,
    0.915,
    "The workflow moves from raw World Bank data to cleaned panel construction, exploratory analysis, "
    "separate forecasting, main GDP modelling, future prediction, and dashboard/report delivery.",
    ha="center",
    fontsize=11,
    color="#475569",
)

plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved: {OUTPUT_PATH}")
