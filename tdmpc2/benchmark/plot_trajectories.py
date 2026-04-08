"""Plot trajectories + goals for each agent from benchmark CSVs."""

import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
AGENTS = ["TDMPC2", "PID", "PPO"]
GOAL_COLORS = {"front": "tab:blue", "left_30deg": "tab:green", "right_30deg": "tab:red"}

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)

for ax, agent in zip(axes, AGENTS):
    csv_path = os.path.join(OUTPUT_DIR, f"benchmark_log_{agent}.csv")
    df = pd.read_csv(csv_path)

    for goal_name, color in GOAL_COLORS.items():
        subset = df[df["goal_name"] == goal_name]
        if subset.empty:
            continue
        # Plot trajectory
        ax.plot(subset["x_m"], subset["y_m"], color=color, linewidth=1.5, label=f"{goal_name} trajectory")
        # Plot start
        ax.plot(subset["x_m"].iloc[0], subset["y_m"].iloc[0], "o", color=color, markersize=8)
        # Plot goal
        gx, gy = subset["goal_x_m"].iloc[0], subset["goal_y_m"].iloc[0]
        ax.plot(gx, gy, "*", color=color, markersize=14, markeredgecolor="k", markeredgewidth=0.5)

    ax.set_title(agent, fontsize=14, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("Agent Trajectories & Goals", fontsize=16, fontweight="bold")
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, "trajectories.png")
plt.savefig(save_path, dpi=150)
print(f"Saved to {save_path}")
plt.show()
