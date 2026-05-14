#!/usr/bin/env python3
"""
Train the DQN agent and produce learning visualisations.

Outputs  →  AI_player/training_results/
  screenshots/ep_NNNN.png      game-state snapshot every --snap-every episodes
  chart_solve_time.png         steps to clear level 1 vs episode
  chart_success_fail.png       cumulative successes / failures + rolling rate %
  chart_progression.png        grid of all screenshots showing policy evolution

Usage
-----
  python train_with_viz.py                          # 300 episodes, default settings
  python train_with_viz.py --episodes 1000          # longer run
  python train_with_viz.py --load checkpoints/dqn_final.pt  # resume
"""

import argparse, os, sys, time

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

sys.path.insert(0, os.path.dirname(__file__))
from environment import PacmanEnv
from dqn_agent   import DQNAgent

# ── output directories ─────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(_HERE, "training_results")
SNAP_DIR     = os.path.join(RESULTS_DIR, "screenshots")
CKPT_DIR     = os.path.join(_HERE, "checkpoints")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train DQN Pac-Man agent with learning visualisations")
    p.add_argument("--episodes",   type=int, default=300,
                   help="total training episodes (default 300)")
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--snap-every", type=int, default=30,
                   help="save a game screenshot every N episodes")
    p.add_argument("--log-every",  type=int, default=30,
                   help="print training stats every N episodes")
    p.add_argument("--save-every", type=int, default=100,
                   help="save checkpoint every N episodes")
    p.add_argument("--load",       type=str, default=None,
                   help="resume from checkpoint path")
    return p.parse_args()


# ── screenshot capture ─────────────────────────────────────────────────────

def capture_screenshot(agent: DQNAgent, episode: int) -> str:
    """
    Run a short near-greedy episode and save a mid-game PNG.
    Returns the saved path.
    """
    env       = PacmanEnv(frame_skip=4)
    saved_eps = agent.eps
    agent.eps = 0.02   # near-pure exploitation for the demo

    state, done, step = env.reset(), False, 0
    while not done and step < 400:
        action = agent.select_action(state)
        state, _, done, _ = env.step(action)
        step += 1
        if step == 60:   # snapshot after agent has navigated a bit
            break

    env._game._draw()
    path = os.path.join(SNAP_DIR, f"ep_{episode:04d}.png")
    pygame.image.save(env._surface, path)
    agent.eps = saved_eps
    return path


# ── charts ─────────────────────────────────────────────────────────────────

def plot_solve_time(records: list[dict], out_path: str):
    """
    Scatter + rolling-mean line of steps to clear level 1.
    Failed episodes (agent never cleared) are shown as red ticks at y=0.
    """
    ep  = np.array([r["episode"]    for r in records], dtype=float)
    raw = np.array([r["solve_steps"] if r["solve_steps"] is not None
                    else np.nan for r in records], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 5))
    mask = ~np.isnan(raw)

    # Successes
    ax.scatter(ep[mask], raw[mask], s=16, alpha=0.55,
               color="#43A047", zorder=3, label="Level cleared")

    # Rolling mean of successful episodes
    if mask.sum() >= 5:
        win = min(20, mask.sum())
        rm  = np.convolve(raw[mask], np.ones(win) / win, mode="valid")
        ax.plot(ep[mask][win - 1:], rm, color="#E53935",
                linewidth=2, label=f"Rolling mean ({win} ep)")

    # Failures as small ticks along the x-axis
    ax.scatter(ep[~mask], np.zeros(mask.size - mask.sum()),
               marker="|", s=50, color="#EF9A9A",
               alpha=0.45, label="Failed (no clear)")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Steps to clear level 1", fontsize=12)
    ax.set_title("Time to Solve the Labyrinth vs Episode",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_success_fail(records: list[dict], out_path: str):
    """
    Cumulative success / failure lines + rolling success-rate % (right axis).
    """
    ep       = np.array([r["episode"] for r in records])
    suc      = np.array([int(r["solved"]) for r in records], dtype=float)
    fail     = 1.0 - suc
    cum_suc  = np.cumsum(suc)
    cum_fail = np.cumsum(fail)

    fig, ax1 = plt.subplots(figsize=(11, 5))

    ax1.fill_between(ep, cum_suc,  alpha=0.12, color="#43A047")
    ax1.fill_between(ep, cum_fail, alpha=0.12, color="#E53935")
    ax1.plot(ep, cum_suc,  color="#43A047", linewidth=2,
             label="Cumulative successes (level cleared)")
    ax1.plot(ep, cum_fail, color="#E53935", linewidth=2,
             label="Cumulative failures (game over)")
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Cumulative count", fontsize=12)

    # Rolling success rate on secondary axis
    ax2 = ax1.twinx()
    win = max(5, min(50, len(suc) // 4))
    if len(suc) >= win:
        rate = np.convolve(suc, np.ones(win) / win, mode="valid") * 100
        ax2.plot(ep[win - 1:], rate, color="#FB8C00",
                 linewidth=2, linestyle="--",
                 label=f"Success rate % (rolling {win} ep)")
    ax2.set_ylabel("Success rate %", fontsize=12, color="#FB8C00")
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="y", labelcolor="#FB8C00")

    lines  = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=10)
    ax1.set_title("Successes and Failures vs Episode",
                  fontsize=14, fontweight="bold")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_progression_grid(snap_paths: list[tuple[int, str]], out_path: str):
    """
    Tile all saved screenshots into a single image, labelled by episode.
    Shows how the agent's behaviour changes over training.
    """
    n    = len(snap_paths)
    if n == 0:
        return
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 2.4, rows * 3.2))
    axes = np.array(axes).reshape(-1)   # always 1-D

    for i, (ep, path) in enumerate(snap_paths):
        img = mpimg.imread(path)
        axes[i].imshow(img)
        axes[i].set_title(f"ep {ep}", fontsize=9)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Agent Behaviour Progression", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── training loop ──────────────────────────────────────────────────────────

def train():
    args = parse_args()
    os.makedirs(SNAP_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    env   = PacmanEnv(frame_skip=4)
    agent = DQNAgent(
        obs_size  = env.OBS_SIZE,
        n_actions = env.N_ACTIONS,
        device    = args.device,
    )
    if args.load:
        agent.load(args.load)

    records:    list[dict]         = []
    snap_paths: list[tuple[int, str]] = []
    best_score  = -float("inf")
    t0          = time.time()

    print(f"Training for {args.episodes} episodes  |  device={args.device}")
    print(f"Screenshots every {args.snap_every} ep  →  {SNAP_DIR}")
    print("─" * 60)

    for episode in range(1, args.episodes + 1):
        state       = env.reset()
        done        = False
        total_steps = 0
        prev_level  = env._game.level
        solve_steps = None    # steps when the first level was cleared

        while not done:
            action                          = agent.select_action(state)
            next_state, reward, done, info  = env.step(action)
            agent.push(state, action, reward, next_state, done)
            agent.train_step()
            state        = next_state
            total_steps += 1

            # Detect first level clear
            if solve_steps is None and info["level"] > prev_level:
                solve_steps = total_steps
                prev_level  = info["level"]

        records.append({
            "episode":     episode,
            "solved":      solve_steps is not None,
            "solve_steps": solve_steps,
            "score":       info["score"],
        })

        # Screenshot
        if episode == 1 or episode % args.snap_every == 0:
            path = capture_screenshot(agent, episode)
            snap_paths.append((episode, path))

        # Periodic checkpoint
        if episode % args.save_every == 0:
            agent.save(os.path.join(CKPT_DIR, f"dqn_ep{episode}.pt"))

        # Best model
        if info["score"] > best_score:
            best_score = info["score"]
            agent.save(os.path.join(CKPT_DIR, "dqn_best.pt"))

        # Log
        if episode % args.log_every == 0:
            win       = min(args.log_every, len(records))
            recent    = records[-win:]
            suc_rate  = sum(r["solved"] for r in recent) / win * 100
            mean_sc   = np.mean([r["score"] for r in recent])
            elapsed   = (time.time() - t0) / 60
            print(f"ep {episode:4d}/{args.episodes}"
                  f"  solved {suc_rate:5.1f}%"
                  f"  score {mean_sc:6.0f}"
                  f"  ε {agent.eps:.3f}"
                  f"  buf {len(agent.buffer):5d}"
                  f"  {elapsed:.1f} min")

    agent.save(os.path.join(CKPT_DIR, "dqn_final.pt"))

    # ── charts ─────────────────────────────────────────────────────
    print("\nGenerating charts …")
    plot_solve_time(
        records,
        os.path.join(RESULTS_DIR, "chart_solve_time.png"))
    plot_success_fail(
        records,
        os.path.join(RESULTS_DIR, "chart_success_fail.png"))
    plot_progression_grid(
        snap_paths,
        os.path.join(RESULTS_DIR, "chart_progression.png"))

    # ── summary ────────────────────────────────────────────────────
    elapsed  = (time.time() - t0) / 60
    n_solved = sum(r["solved"] for r in records)
    solved   = [r for r in records if r["solved"]]
    mean_st  = np.mean([r["solve_steps"] for r in solved]) if solved else float("nan")

    print(f"\n{'═'*60}")
    print(f"  Episodes      : {args.episodes}")
    print(f"  Solved        : {n_solved}  ({n_solved/args.episodes*100:.1f}%)")
    print(f"  Mean solve    : {mean_st:.0f} steps  (successful episodes)")
    print(f"  Best score    : {best_score}")
    print(f"  Total time    : {elapsed:.1f} min")
    print(f"  Results in    : {RESULTS_DIR}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    train()
