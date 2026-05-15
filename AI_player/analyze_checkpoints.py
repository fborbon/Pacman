#!/usr/bin/env python3
"""
Evaluate saved checkpoints and generate training-progress charts.

Loads  dqn_ep100.pt / dqn_ep200.pt / dqn_ep300.pt, runs 60 evaluation
episodes per checkpoint, and produces four charts:

  training_results/chart_score_progress.png
  training_results/chart_dots_eaten.png
  training_results/chart_solve_time.png      (steps to reach ≥1200 score)
  training_results/chart_success_fail.png    (success = score ≥ 1200)
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import PacmanEnv
from dqn_agent   import DQNAgent

_HERE       = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR    = os.path.join(_HERE, "checkpoints")
RESULTS_DIR = os.path.join(_HERE, "training_results")
TOTAL_DOTS  = 244
# Score milestone considered a "partial solve" (≈ 120 dots eaten)
SUCCESS_THRESHOLD = 1200
N_EVAL_GAMES      = 60


# ── evaluation ─────────────────────────────────────────────────────────────

def eval_checkpoint(path: str, n: int = N_EVAL_GAMES) -> dict:
    """Run n near-greedy episodes and return per-episode metrics."""
    env   = PacmanEnv(frame_skip=4)
    agent = DQNAgent(env.OBS_SIZE, env.N_ACTIONS)
    agent.load(path)
    agent.eps = 0.02   # near-greedy evaluation

    scores, dots_eaten, steps_survived, milestone_steps = [], [], [], []

    for _ in range(n):
        state, done, step = env.reset(), False, 0
        milestone_hit = None
        while not done:
            action            = agent.select_action(state)
            state, _, done, info = env.step(action)
            step += 1
            if milestone_hit is None and info["score"] >= SUCCESS_THRESHOLD:
                milestone_hit = step

        scores.append(info["score"])
        dots_eaten.append(TOTAL_DOTS - info["dots_left"])
        steps_survived.append(step)
        milestone_steps.append(milestone_hit)   # None = never reached

    return {
        "scores":         np.array(scores),
        "dots_eaten":     np.array(dots_eaten),
        "steps":          np.array(steps_survived),
        "milestone_steps": milestone_steps,
    }


# ── charts ──────────────────────────────────────────────────────────────────

def _bar_ax(ax, labels, means, stds, color, ylabel, title):
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=color, alpha=0.78, width=0.5, zorder=3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[labels.index(bar.get_label() if hasattr(bar, 'get_label') else '')] * 0.05,
                f"{m:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(means) * 1.25 + 1)


def _palette(n, cmap="Blues"):
    cm = plt.get_cmap(cmap)
    return [cm(0.3 + 0.65 * i / max(n - 1, 1)) for i in range(n)]


def plot_score_progress(stage_labels, data_by_stage, out_path):
    n = len(stage_labels)
    fig, ax = plt.subplots(figsize=(max(9, n * 1.4), 5))
    colors = _palette(n, "Blues")
    for i, (label, d) in enumerate(zip(stage_labels, data_by_stage)):
        scores = d["scores"]
        x = np.full(len(scores), i) + np.random.uniform(-0.18, 0.18, len(scores))
        ax.scatter(x, scores, s=18, alpha=0.35, color=colors[i], zorder=2)
        ax.bar(i, scores.mean(), width=0.45, alpha=0.55, color=colors[i],
               zorder=3, label=f"{label}  μ={scores.mean():.0f}")
        ax.errorbar(i, scores.mean(), yerr=scores.std(), fmt="none",
                    capsize=6, color="black", linewidth=1.5, zorder=4)

    ax.set_xticks(range(len(stage_labels)))
    ax.set_xticklabels(stage_labels, fontsize=max(7, 12 - len(stage_labels)))
    ax.set_xlabel("Training stage (checkpoint)", fontsize=12)
    ax.set_ylabel("Score per episode", fontsize=12)
    ax.set_title("Score Progression Across Training Stages",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_dots_eaten(stage_labels, data_by_stage, out_path):
    n = len(stage_labels)
    fig, ax = plt.subplots(figsize=(max(9, n * 1.4), 5))
    colors = _palette(n, "Greens")
    for i, (label, d) in enumerate(zip(stage_labels, data_by_stage)):
        de = d["dots_eaten"]
        pct = de / TOTAL_DOTS * 100
        x = np.full(len(pct), i) + np.random.uniform(-0.18, 0.18, len(pct))
        ax.scatter(x, pct, s=18, alpha=0.35, color=colors[i], zorder=2)
        ax.bar(i, pct.mean(), width=0.45, alpha=0.55, color=colors[i],
               zorder=3, label=f"{label}  μ={de.mean():.0f} dots ({pct.mean():.1f}%)")
        ax.errorbar(i, pct.mean(), yerr=pct.std(), fmt="none",
                    capsize=6, color="black", linewidth=1.5, zorder=4)

    ax.set_xticks(range(len(stage_labels)))
    ax.set_xticklabels(stage_labels, fontsize=12)
    ax.set_xlabel("Training stage (checkpoint)", fontsize=12)
    ax.set_ylabel("Dots eaten  %  of 244 total", fontsize=12)
    ax.set_title("Maze Completion Progress Across Training Stages",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.axhline(100, color="gold", linestyle="--", linewidth=1.2,
               label="Level clear (100%)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_solve_time(stage_labels, data_by_stage, out_path):
    """
    'Solve time' = steps to first reach the score milestone (≥1200 pts).
    Episodes that never hit the milestone are excluded from the mean
    and their count is shown separately.
    """
    n = len(stage_labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, n * 1.8), 5))
    colors = _palette(n, "Oranges")

    means, stds, hit_rates = [], [], []
    for label, d in zip(stage_labels, data_by_stage):
        ms  = [s for s in d["milestone_steps"] if s is not None]
        hit = len(ms) / N_EVAL_GAMES * 100
        means.append(np.mean(ms) if ms else 0)
        stds.append(np.std(ms) if ms else 0)
        hit_rates.append(hit)

    # Left: mean steps to milestone (only episodes that hit it)
    x = np.arange(len(stage_labels))
    bars = ax1.bar(x, means, yerr=stds, capsize=6,
                   color=colors, alpha=0.78, width=0.5, zorder=3)
    for bar, m, hr in zip(bars, means, hit_rates):
        if m > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, m + 5,
                     f"{m:.0f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_labels, fontsize=12)
    ax1.set_ylabel(f"Steps to score ≥{SUCCESS_THRESHOLD}", fontsize=11)
    ax1.set_title(f"Time to Reach {SUCCESS_THRESHOLD}-Point Milestone\n"
                  "(fewer steps = faster solve)", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Right: % of episodes that hit the milestone
    ax2.bar(x, hit_rates, color=colors, alpha=0.78, width=0.5, zorder=3)
    for i, (xi, hr) in enumerate(zip(x, hit_rates)):
        ax2.text(xi, hr + 1.5, f"{hr:.0f}%",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stage_labels, fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel(f"Episodes reaching {SUCCESS_THRESHOLD} pts  %", fontsize=11)
    ax2.set_title(f"% Episodes Solving the {SUCCESS_THRESHOLD}-Point Milestone",
                  fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Time to Solve the Labyrinth vs Training Stage",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_success_fail(stage_labels, data_by_stage, out_path):
    """
    Stacked bar: success (score ≥ threshold) vs fail per training stage.
    Line overlay: mean score trend.
    """
    n = len(stage_labels)
    fig, ax1 = plt.subplots(figsize=(max(9, n * 1.4), 5))
    x       = np.arange(n)
    width   = 0.5

    n_suc, n_fail, mean_scores = [], [], []
    for d in data_by_stage:
        suc  = (d["scores"] >= SUCCESS_THRESHOLD).sum()
        fail = N_EVAL_GAMES - suc
        n_suc.append(suc)
        n_fail.append(fail)
        mean_scores.append(d["scores"].mean())

    ax1.bar(x, n_suc,  width, label=f"Success (score ≥ {SUCCESS_THRESHOLD})",
            color="#4CAF50", alpha=0.82, zorder=3)
    ax1.bar(x, n_fail, width, bottom=n_suc,
            label="Fail (score < threshold)", color="#F44336", alpha=0.82, zorder=3)

    for xi, s, f in zip(x, n_suc, n_fail):
        if s > 0:
            ax1.text(xi, s / 2, f"{s}", ha="center", va="center",
                     fontsize=11, fontweight="bold", color="white")
        ax1.text(xi, s + f / 2, f"{f}", ha="center", va="center",
                 fontsize=11, fontweight="bold", color="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_labels, fontsize=12)
    ax1.set_xlabel("Training stage (checkpoint)", fontsize=12)
    ax1.set_ylabel(f"Episodes  (out of {N_EVAL_GAMES})", fontsize=12)
    ax1.set_ylim(0, N_EVAL_GAMES * 1.1)

    # Mean score trend on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, mean_scores, "o--", color="#FF9800", linewidth=2,
             markersize=8, label="Mean score", zorder=5)
    for xi, ms in zip(x, mean_scores):
        ax2.text(xi, ms + 25, f"{ms:.0f}", ha="center", va="bottom",
                 fontsize=10, color="#E65100", fontweight="bold")
    ax2.set_ylabel("Mean score per episode", fontsize=12, color="#E65100")
    ax2.tick_params(axis="y", labelcolor="#E65100")
    ax2.set_ylim(0, max(mean_scores) * 1.4)

    lines  = ax1.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax1.legend(lines[0] + lines2[0], lines[1] + lines2[1],
               loc="upper left", fontsize=10)

    ax1.set_title(f"Successes and Failures vs Training Stage\n"
                  f"(success = score ≥ {SUCCESS_THRESHOLD})",
                  fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    # All candidate checkpoints in chronological order.
    # Only those that exist on disk are evaluated.
    candidates = [
        ("ep 100",   os.path.join(CKPT_DIR, "dqn_ep100.pt")),
        ("ep 200",   os.path.join(CKPT_DIR, "dqn_ep200.pt")),
        ("ep 300",   os.path.join(CKPT_DIR, "dqn_ep300.pt")),
        ("ep 1300",  os.path.join(CKPT_DIR, "dqn_ep1300.pt")),
        ("ep 2300",  os.path.join(CKPT_DIR, "dqn_ep2300.pt")),
        ("ep 3300",  os.path.join(CKPT_DIR, "dqn_ep3300.pt")),
        ("ep 4300",  os.path.join(CKPT_DIR, "dqn_ep4300.pt")),
        ("ep 5300",  os.path.join(CKPT_DIR, "dqn_ep5300.pt")),
        ("ep 6300",  os.path.join(CKPT_DIR, "dqn_ep1000.pt")),
        ("ep 7300",  os.path.join(CKPT_DIR, "dqn_ep2000.pt")),
        ("ep 8300",  os.path.join(CKPT_DIR, "dqn_ep3000.pt")),
        ("ep 9300",  os.path.join(CKPT_DIR, "dqn_ep4000.pt")),
        ("ep 10300", os.path.join(CKPT_DIR, "dqn_ep5000.pt")),
    ]
    checkpoints = [(l, p) for l, p in candidates if os.path.exists(p)]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stage_labels = []
    data_by_stage = []

    for label, path in checkpoints:
        if not os.path.exists(path):
            print(f"  skipping {path} (not found)")
            continue
        print(f"Evaluating {label} ({N_EVAL_GAMES} games)…")
        d = eval_checkpoint(path)
        stage_labels.append(label)
        data_by_stage.append(d)
        print(f"  score  μ={d['scores'].mean():.0f}  σ={d['scores'].std():.0f}"
              f"  dots  μ={d['dots_eaten'].mean():.0f}"
              f"  milestone hit={sum(s is not None for s in d['milestone_steps'])}/{N_EVAL_GAMES}")

    print("\nGenerating charts…")
    plot_score_progress(stage_labels, data_by_stage,
        os.path.join(RESULTS_DIR, "chart_score_progress.png"))
    plot_dots_eaten(stage_labels, data_by_stage,
        os.path.join(RESULTS_DIR, "chart_dots_eaten.png"))
    plot_solve_time(stage_labels, data_by_stage,
        os.path.join(RESULTS_DIR, "chart_solve_time.png"))
    plot_success_fail(stage_labels, data_by_stage,
        os.path.join(RESULTS_DIR, "chart_success_fail.png"))

    print(f"\nAll charts saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
