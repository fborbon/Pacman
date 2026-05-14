#!/usr/bin/env python3
"""
Watch a trained DQN agent play Pac-Man.

Usage
-----
  python play.py --model checkpoints/dqn_best.pt
  python play.py --model checkpoints/dqn_best.pt --episodes 10
  python play.py --model checkpoints/dqn_best.pt --headless   # no window

The agent runs at 60 FPS with a real pygame window by default.
Pass --headless to benchmark without rendering.
"""

import argparse
import os
import sys

import numpy as np

# Display must be set before environment import
_headless = "--headless" in sys.argv
if _headless:
    import os as _os
    _os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
    _os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(__file__))
from environment import PacmanEnv
from dqn_agent   import DQNAgent


def parse_args():
    p = argparse.ArgumentParser(description="Watch the trained DQN agent play")
    p.add_argument("--model",      type=str, required=True,
                   help="path to checkpoint (.pt)")
    p.add_argument("--episodes",   type=int, default=5,
                   help="number of episodes to play")
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--headless",   action="store_true",
                   help="disable pygame window (benchmarking)")
    return p.parse_args()


def play():
    args  = parse_args()
    env   = PacmanEnv(frame_skip=args.frame_skip, render=not args.headless)
    agent = DQNAgent(
        obs_size  = env.OBS_SIZE,
        n_actions = env.N_ACTIONS,
        device    = args.device,
        eps_start = 0.0,
        eps_end   = 0.0,
        eps_decay = 1,
    )
    agent.load(args.model)
    agent.eps = 0.0   # pure exploitation

    scores  = []
    rewards = []

    for ep in range(1, args.episodes + 1):
        state        = env.reset()
        total_reward = 0.0
        done         = False

        while not done:
            action                         = agent.select_action(state)
            state, reward, done, info      = env.step(action)
            total_reward                  += reward

        scores.append(info["score"])
        rewards.append(total_reward)
        print(f"Episode {ep:3d}  score={info['score']:6d}  "
              f"reward={total_reward:7.1f}  lives={info['lives']}  "
              f"level={info['level']}")

    print(f"\n{'─'*50}")
    print(f"Mean score  : {np.mean(scores):.0f}")
    print(f"Max score   : {np.max(scores)}")
    print(f"Mean reward : {np.mean(rewards):.1f}")


if __name__ == "__main__":
    play()
