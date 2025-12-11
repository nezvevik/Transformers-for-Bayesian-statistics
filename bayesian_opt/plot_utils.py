import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.tri as tri

import math
import matplotlib.pyplot as plt
import torch

def plot_history_1d(history, title="Bayesian Optimization Progress"):
    n_iter = len(history)
    cols = 4
    rows = math.ceil(n_iter / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, step in enumerate(history):
        ax = axes[i]

        x = step["x_candidates"].cpu().numpy().reshape(-1)
        means = step["means"].cpu().numpy().reshape(-1)
        vars_ = step["vars"].cpu().numpy().reshape(-1)
        stds = vars_ ** 0.5

        x_next = step["x_next"].cpu().numpy().reshape(-1)
        y_next = step["y_next"].cpu().numpy().reshape(-1)

        best_x = step["best_x"].cpu().numpy().reshape(-1)
        best_y = step["best_y"].cpu().numpy().reshape(-1)

        # Mean curve
        ax.plot(x, means, label="Mean", linewidth=2)

        # Uncertainty band
        ax.fill_between(
            x,
            means - stds,
            means + stds,
            alpha=0.3,
            label="± Std Dev"
        )

        # Chosen next point
        ax.scatter(
            x_next,
            y_next,
            c="red",
            s=60,
            label="x_next",
            zorder=5,
        )

        # Current best point
        ax.scatter(
            best_x,
            best_y,
            c="green",
            s=60,
            label="best_x",
            marker="X",
            zorder=5,
        )

        ax.set_title(f"Iteration {i+1}")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend(loc="upper left")

    # Hide unused subplots if n_iter is not multiple of cols
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_history_2d(history, k_var=1.96):
    n_iter = len(history)

    fig, axes = plt.subplots(n_iter, 2, figsize=(12, 4 * n_iter))
    if n_iter == 1:
        axes = np.array([axes])  # force shape: [1,2]

    for i, h in enumerate(history):

        ax_mean = axes[i, 0]
        ax_mean_var = axes[i, 1]

        xc = h["x_candidates"].detach().cpu().numpy()
        means = h["means"].detach().cpu().numpy()
        vars_ = h["vars"].detach().cpu().numpy()

        X_all = h["X"].detach().cpu().numpy()

        x_next = h["x_next"].detach().cpu().numpy()
        best_x = h["best_x"].detach().cpu().numpy()

        if x_next.ndim == 1:
            x_next = x_next.reshape(1, -1)
        if best_x.ndim == 1:
            best_x = best_x.reshape(1, -1)

        # Observed points except the newest one
        if len(X_all) > 1:
            X_prev = X_all[:-1]
        else:
            X_prev = X_all

        # Build triangulation
        triang = tri.Triangulation(xc[:, 0], xc[:, 1])

        # Mean prediction contour
        cmean = ax_mean.tricontourf(
            triang,
            means,
            levels=40,
            cmap="viridis"
        )

        # Previous observed points
        if len(X_prev) > 0:
            ax_mean.scatter(
                X_prev[:, 0], X_prev[:, 1],
                c="white",
                edgecolor="black",
                s=60,
                zorder=3
            )

        # Next x
        ax_mean.scatter(
            x_next[:, 0], x_next[:, 1],
            c="red",
            s=100,
            marker="X",
            zorder=4
        )

        # Best x
        ax_mean.scatter(
            best_x[:, 0], best_x[:, 1],
            c="green",
            s=120,
            marker="*",
            edgecolor="black",
            zorder=5
        )

        ax_mean.set_xlim(0, 1)
        ax_mean.set_ylim(0, 1)
        ax_mean.set_title(f"Iter {i+1}: Mean Prediction")
        fig.colorbar(cmean, ax=ax_mean, fraction=0.046, pad=0.04)

        # Mean - variance score (same variance factor everywhere)
        score = means - k_var * vars_

        cvar = ax_mean_var.tricontourf(
            triang,
            score,
            levels=40,
            cmap="viridis"
        )

        # Previous observed points
        if len(X_prev) > 0:
            ax_mean_var.scatter(
                X_prev[:, 0], X_prev[:, 1],
                c="white",
                edgecolor="black",
                s=60,
                zorder=3
            )

        # Next x
        ax_mean_var.scatter(
            x_next[:, 0], x_next[:, 1],
            c="red",
            s=100,
            marker="X",
            zorder=4
        )

        # Best x
        ax_mean_var.scatter(
            best_x[:, 0], best_x[:, 1],
            c="green",
            s=120,
            marker="*",
            edgecolor="black",
            zorder=5
        )

        ax_mean_var.set_xlim(0, 1)
        ax_mean_var.set_ylim(0, 1)
        ax_mean_var.set_title(f"Iter {i+1}: Mean - {k_var} * Var")
        fig.colorbar(cvar, ax=ax_mean_var, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
