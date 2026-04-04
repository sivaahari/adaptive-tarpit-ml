"""
Visualization dashboard.

Reads the SQLite log database and produces a multi-panel PNG report.

Usage:
    python3 visualization/dashboard.py

Output:
    visualization/tarpit_stats.png
"""

import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns


# ── Path resolution (no hardcoded ~/projects/... paths) ──────────────────────

_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.normpath(os.path.join(_HERE, ".."))
DB_PATH  = os.path.join(_ROOT, "data", "tarpit_logs.db")
OUT_PATH = os.path.join(_HERE, "tarpit_stats.png")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        print(f"No database found at {DB_PATH}. Run main.py first to generate traffic logs.")
        sys.exit(0)

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM connections", conn)

    if df.empty:
        print("Database exists but contains no records yet.")
        sys.exit(0)

    # AFTER
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    return df


# ── Charts ────────────────────────────────────────────────────────────────────

def generate_dashboard(df: pd.DataFrame) -> None:
    sns.set_theme(style="darkgrid", palette="muted")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Adaptive ML Tarpit — Traffic Analysis Dashboard", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

    # ── Panel 1: Classification breakdown (bar) ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["prediction"].map({0: "Benign", 1: "Malicious"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette=["steelblue", "crimson"], ax=ax1)
    ax1.set_title("Traffic Classification")
    ax1.set_xlabel("Verdict")
    ax1.set_ylabel("Connection Count")
    for bar, val in zip(ax1.patches, counts.values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val), ha="center", va="bottom", fontsize=10,
        )

    # ── Panel 2: ML confidence distribution (histogram) ──────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    benign_prob    = df[df["prediction"] == 0]["probability"]
    malicious_prob = df[df["prediction"] == 1]["probability"]
    ax2.hist(benign_prob,    bins=20, alpha=0.7, label="Benign",    color="steelblue")
    ax2.hist(malicious_prob, bins=20, alpha=0.7, label="Malicious", color="crimson")
    ax2.set_title("ML Confidence Distribution")
    ax2.set_xlabel("P(Malicious)")
    ax2.set_ylabel("Count")
    ax2.legend()

    # ── Panel 3: Top offender IPs ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    top_ips = (
        df[df["prediction"] == 1]
        .groupby("src_ip")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    if not top_ips.empty:
        sns.barplot(x=top_ips.values, y=top_ips.index, palette="Reds_r", ax=ax3)
        ax3.set_title("Top 10 Malicious IPs")
        ax3.set_xlabel("Tarpitted Connections")
        ax3.set_ylabel("Source IP")
    else:
        ax3.set_title("Top 10 Malicious IPs")
        ax3.text(0.5, 0.5, "No malicious\nconnections yet",
                 ha="center", va="center", transform=ax3.transAxes)

    # ── Panel 4: Action breakdown (pie) ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    action_counts = df["action_taken"].value_counts()
    ax4.pie(
        action_counts.values,
        labels=action_counts.index,
        autopct="%1.1f%%",
        colors=["steelblue", "crimson"],
        startangle=90,
    )
    ax4.set_title("Action Breakdown")

    # ── Panel 5: Connections over time (time-series) ──────────────────────────
    ax5 = fig.add_subplot(gs[1, 1:])
    df_time = df.set_index("timestamp").sort_index()
    df_time["is_malicious"] = df_time["prediction"]
    df_time["is_benign"]    = 1 - df_time["prediction"]

    resampled = df_time[["is_malicious", "is_benign"]].resample("1min").sum()
    if len(resampled) > 1:
        resampled.plot(
            ax=ax5,
            color=["crimson", "steelblue"],
            label=["Malicious", "Benign"],
        )
        ax5.set_title("Connections per Minute")
        ax5.set_xlabel("Time")
        ax5.set_ylabel("Count")
        ax5.legend(["Malicious", "Benign"])
    else:
        ax5.set_title("Connections over Time")
        ax5.text(0.5, 0.5, "Need more data\nfor time-series",
                 ha="center", va="center", transform=ax5.transAxes)

    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Dashboard saved → {OUT_PATH}")


# ── Entry-point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} connection records.")
    generate_dashboard(df)