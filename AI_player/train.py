#!/usr/bin/env python3
"""
Train the DQN agent on Pac-Man.

Usage
-----
  python train.py                          # default settings
  python train.py --episodes 10000        # longer run
  python train.py --load checkpoints/dqn_ep500.pt   # resume
  python train.py --device cuda           # GPU if available

Checkpoints are saved to AI_player/checkpoints/ every --save-every episodes.
The best model (highest mean score over last 100 eps) is saved separately.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from environment import PacmanEnv
from dqn_agent   import DQNAgent

# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DQN Pac-Man agent")
    p.add_argument("--episodes",   type=int,   default=5_000,
                   help="total training episodes (default 5000)")
    p.add_argument("--device",     type=str,   default="cpu",
                   help="torch device: cpu or cuda")
    p.add_argument("--save-every", type=int,   default=500,
                   help="save checkpoint every N episodes")
    p.add_argument("--log-every",  type=int,   default=100,
                   help="print stats every N episodes")
    p.add_argument("--load",       type=str,   default=None,
                   help="resume from checkpoint path")
    p.add_argument("--frame-skip", type=int,   default=4,
                   help="frames per agent step")
    return p.parse_args()


# ── training loop ──────────────────────────────────────────────────────────

def train():
    args  = parse_args()
    env   = PacmanEnv(frame_skip=args.frame_skip)
    agent = DQNAgent(
        obs_size  = env.OBS_SIZE,
        n_actions = env.N_ACTIONS,
        device    = args.device,
    )

    if args.load:
        agent.load(args.load)

    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ep_rewards: list[float] = []
    ep_scores:  list[float] = []
    best_mean_score         = -float("inf")
    t0                      = time.time()

    for episode in range(1, args.episodes + 1):
        state        = env.reset()
        total_reward = 0.0
        done         = False

        while not done:
            action                        = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.push(state, action, reward, next_state, done)
            agent.train_step()
            state        = next_state
            total_reward += reward

        ep_rewards.append(total_reward)
        ep_scores.append(info["score"])

        # Logging
        if episode % args.log_every == 0:
            window      = min(args.log_every, len(ep_rewards))
            mean_reward = np.mean(ep_rewards[-window:])
            mean_score  = np.mean(ep_scores[-window:])
            elapsed     = time.time() - t0
            print(
                f"ep {episode:5d}/{args.episodes}"
                f"  reward {mean_reward:8.1f}"
                f"  score {mean_score:7.0f}"
                f"  ε {agent.eps:.3f}"
                f"  buf {len(agent.buffer):6d}"
                f"  {elapsed/60:.1f} min"
            )

            # Save best model
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                best_path = os.path.join(ckpt_dir, "dqn_best.pt")
                agent.save(best_path)

        # Periodic checkpoint
        if episode % args.save_every == 0:
            path = os.path.join(ckpt_dir, f"dqn_ep{episode}.pt")
            agent.save(path)

    # Final checkpoint
    agent.save(os.path.join(ckpt_dir, "dqn_final.pt"))
    print(f"\nTraining complete in {(time.time()-t0)/60:.1f} min")
    print(f"Best mean score: {best_mean_score:.0f}")


if __name__ == "__main__":
    train()
