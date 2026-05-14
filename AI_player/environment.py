"""
Gym-style headless environment wrapping pacman.py.

Observation  : 81-dim float32 vector (positions, directions, local maze view)
Action space : 0=UP  1=DOWN  2=LEFT  3=RIGHT
Reward       : scaled score delta, death penalty, level-clear bonus, step cost
"""

import os, sys
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Python_version"))
import pacman as pm

_TOTAL_DOTS = sum(1 for row in pm.MAZE_DEF for ch in row if ch in ".o")


class PacmanEnv:
    """
    Wraps the Game class for reinforcement learning.

    Each call to step() advances the simulation by `frame_skip` frames
    (default 4, matching the original Atari DQN paper) and returns a
    compact feature vector as the observation.
    """

    ACTIONS   = [pm.UP, pm.DOWN, pm.LEFT, pm.RIGHT]
    N_ACTIONS = 4
    OBS_SIZE  = 81   # 2 + 4 + 24 + 1 + 1 + 49

    def __init__(self, frame_skip: int = 4, render: bool = False):
        self.frame_skip = frame_skip
        self._render    = render

        pygame.init()
        if render:
            self._surface = pygame.display.set_mode((pm.SCREEN_W, pm.SCREEN_H))
            pygame.display.set_caption("Pac-Man DQN Agent")
        else:
            self._surface = pygame.Surface((pm.SCREEN_W, pm.SCREEN_H))

        self._clock = pygame.time.Clock()
        self._game  = None
        self.reset()

    # ── public interface ───────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self._game  = self._init_game()
        self._steps = 0
        return self._observe()

    def step(self, action: int):
        """
        Apply `action` for `frame_skip` frames.

        Returns
        -------
        obs    : np.ndarray  next observation
        reward : float
        done   : bool
        info   : dict        score / lives / level / dots_left
        """
        g = self._game
        g.pac.set_dir(self.ACTIONS[action])

        score_before = g.score
        reward       = 0.0

        for _ in range(self.frame_skip):
            if g._state == "GAMEOVER":
                break

            if g._state == "LEVEL_CLEAR":
                reward += 100.0
                g.level += 1
                g._new_level()
                g._state = "PLAYING"
                break

            prev_state = g._state
            g._update(1 / 60)

            # Pac-Man just died — fast-forward animation, apply penalty
            if g._state == "DYING" and prev_state == "PLAYING":
                reward -= 50.0
                while g._state == "DYING":
                    g._update(1 / 60)
                if g._state == "READY":
                    g._state = "PLAYING"   # skip the "READY!" countdown
                break

            if self._render:
                g._draw()
                pygame.display.flip()
                self._clock.tick(60)

        reward += (g.score - score_before) * 0.1   # scale game score
        reward -= 0.05                              # per-step cost

        self._steps += 1
        done = g._state == "GAMEOVER" or self._steps >= 10_000

        return self._observe(), reward, done, {
            "score":     g.score,
            "lives":     g.lives,
            "level":     g.level,
            "dots_left": g.dots_left,
        }

    # ── observation builder ────────────────────────────────────────────

    def _observe(self) -> np.ndarray:
        """
        Build the 81-dimensional state vector:
          [0:2]   Pac-Man col/row  (normalised)
          [2:6]   Pac-Man direction one-hot (UP DN LF RT)
          [6:30]  4 ghosts × 6 features
          [30]    fright timer (normalised)
          [31]    dots remaining (normalised)
          [32:81] 7×7 local maze centred on Pac-Man (normalised cell type)
        """
        g   = self._game
        obs = []

        # Pac-Man position
        obs += [g.pac.col / pm.COLS, g.pac.row / pm.ROWS]

        # Pac-Man direction (one-hot over UP DN LF RT)
        for d in pm.ALL_DIRS:
            obs.append(1.0 if g.pac.dir == d else 0.0)

        # Ghost features
        for gh in g.ghosts:
            obs += [
                gh.col / pm.COLS,
                gh.row / pm.ROWS,
                1.0 if gh.is_frightened else 0.0,
                1.0 if gh.is_eaten      else 0.0,
                1.0 if gh.state == pm.GS_INACTIVE else 0.0,
                (abs(gh.col - g.pac.col) + abs(gh.row - g.pac.row))
                / (pm.COLS + pm.ROWS),
            ]

        # Timers
        obs.append(g.fright_timer / max(g.fright_total, 1))
        obs.append(g.dots_left    / _TOTAL_DOTS)

        # 7×7 local maze view (49 cells)
        pr, pc = g.pac.row, g.pac.col
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                r, c = pr + dr, pc + dc
                if 0 <= r < pm.ROWS and 0 <= c < pm.COLS:
                    cell = g.grid[r][c]
                else:
                    cell = pm.WALL_CELL
                obs.append(cell / 5.0)

        return np.array(obs, dtype=np.float32)

    # ── internal helpers ───────────────────────────────────────────────

    def _init_game(self):
        """Construct a Game object bypassing its pygame-coupled __init__."""
        g         = pm.Game.__new__(pm.Game)
        g.screen  = self._surface
        g.clock   = self._clock
        g.font    = pygame.font.SysFont("monospace", 16, bold=True)
        g.sfont   = pygame.font.SysFont("monospace", 12)
        g.high    = 0
        pm.Game._new_game(g)
        g._state  = "PLAYING"
        return g
