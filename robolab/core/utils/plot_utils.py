# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_objects(obj_poses: list[dict] | dict, episode_id = 0, title="", image_path=None):

    fig = plt.figure()
    colors = ['y', 'r', 'b', 'g', 'm', 'c', 'k']

    if isinstance(obj_poses, dict):
        obj_poses = [obj_poses]
        episode_ids = [episode_id]
    elif isinstance(obj_poses, list):
        episode_ids = [i for i in range(len(obj_poses))]

    # Create subplot once, outside the loop
    ax = fig.add_subplot(111, projection='3d')

    for i, obj_pose_dict in enumerate(obj_poses):
        for idx, (object, pose) in enumerate(obj_pose_dict.items()):
            position = pose[:3]
            object_label = f"{object}_{episode_ids[i]}"
            ax.scatter(position[0], position[1], position[2], color=colors[idx], s=30, marker='o', label=object_label)
            ax.text(position[0], position[1], position[2], object_label, color=colors[idx], fontsize=9)

    # Configure axes once after all data is plotted
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    # Set top-down view (elevation=90 looks straight down, azimuth=0)
    ax.view_init(elev=45, azim=120)
    # Set axis limits from -1 to 1
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0.5, 0.5)

    plt.tight_layout()

    # Save to file
    plt.savefig(image_path)
    plt.close(fig)


def plot_ee_path_xy(ee_positions: np.ndarray, title: str = "", image_path: str | None = None):
    """Plot the end-effector trajectory in the XY plane.

    Args:
        ee_positions: Array of shape (T, 3) or (T, 2) containing EE positions.
        title: Plot title.
        image_path: Optional output path. When provided, the plot is saved.

    Returns:
        The output path when ``image_path`` is provided, otherwise the figure.
    """
    xy = np.asarray(ee_positions, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"Expected ee_positions with shape (T, 2+) but got {xy.shape}")
    if xy.shape[0] == 0:
        raise ValueError("Cannot plot an empty end-effector trajectory.")

    xy = xy[:, :2]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xy[:, 0], xy[:, 1], color="#4C78A8", linewidth=2.0, alpha=0.9, zorder=2)

    if xy.shape[0] > 1:
        time_steps = np.linspace(0.0, 1.0, xy.shape[0], dtype=np.float64)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=time_steps,
            cmap="viridis",
            s=18,
            linewidths=0.0,
            zorder=3,
        )
    else:
        ax.scatter(xy[:, 0], xy[:, 1], color="#4C78A8", s=30, zorder=3)

    ax.scatter(xy[0, 0], xy[0, 1], color="#2CA02C", s=90, marker="o", label="start", zorder=4)
    ax.scatter(xy[-1, 0], xy[-1, 1], color="#D62728", s=110, marker="X", label="end", zorder=4)

    x_min, y_min = np.min(xy, axis=0)
    x_max, y_max = np.max(xy, axis=0)
    x_pad = max((x_max - x_min) * 0.1, 0.02)
    y_pad = max((y_max - y_min) * 0.1, 0.02)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title or "End-effector path (XY)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()

    if image_path is not None:
        plt.savefig(image_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return image_path

    return fig
