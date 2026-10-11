"""Charts for the README."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, ACCENT, MUTED = "#1f2a37", "#2563eb", "#9aa4b2"
LABELS = {
    "career_H": "Career hits", "career_G": "Career games", "AB": "At-bats (prior season)",
    "career_HR": "Career home runs", "seasons_played": "Seasons played", "G": "Games (prior season)",
    "age": "Age", "RBI": "RBI (prior season)", "HR_change": "Change in HR vs. year before",
    "BB": "Walks (prior season)", "HR": "Home runs (prior season)", "OPS": "OPS (prior season)",
}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)
    ax.title.set_color(INK)


def feature_importance(importance: pd.Series, out: Path):
    top = importance.head(8)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [ACCENT if i == len(top) - 1 else MUTED for i in range(len(top))]
    ax.barh([LABELS.get(f, f) for f in top.index], top.values, color=colors)
    ax.set_xlabel("Drop in R² when the feature is shuffled (permutation importance)")
    ax.set_title("What drives MLB salaries? Experience beats last season's stats", loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def predicted_vs_actual(y_true, y_pred, r2: float, out: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    actual, pred = np.exp(y_true), np.exp(y_pred)
    ax.scatter(actual, pred, s=8, alpha=0.35, color=ACCENT, edgecolors="none")
    lims = [0.05, 20]
    ax.plot(lims, lims, color=INK, lw=1, ls="--", label="Perfect prediction")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual salary (× league average)")
    ax.set_ylabel("Predicted salary (× league average)")
    ax.set_title(f"Held-out 2012–2016 seasons (R² = {r2:.2f})", loc="left")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def salary_by_experience(df: pd.DataFrame, out: Path):
    grp = (df.assign(bucket=df["seasons_played"].clip(upper=15))
             .groupby("bucket")["salary_vs_league"].median())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(grp.index, grp.values, marker="o", color=ACCENT)
    ax.axvspan(0.5, 3.5, color=MUTED, alpha=0.15)
    ax.text(2, grp.max() * 0.9, "Pre-arbitration\n(pay near league minimum)",
            ha="center", color=INK, fontsize=9)
    ax.set_xlabel("Seasons played before the salary year (15 = 15+)")
    ax.set_ylabel("Median salary (× league average)")
    ax.set_title("Pay jumps after ~3-4 seasons, when players reach arbitration", loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
