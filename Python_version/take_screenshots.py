#!/usr/bin/env python3
"""
Render several game states and save them as PNG screenshots.
Runs headlessly via SDL_VIDEODRIVER=offscreen — no display required.
"""

import os, sys
os.environ["SDL_VIDEODRIVER"] = "offscreen"
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Resolve output folder (screenshots/ at repo root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "screenshots")
os.makedirs(OUT_DIR, exist_ok=True)

# Import the game module (adds its directory to path)
sys.path.insert(0, SCRIPT_DIR)
import pacman as pm   # noqa: E402  (env vars must be set first)
import pygame

# ── helpers ────────────────────────────────────────────────────────

def make_screen():
    pygame.init()
    return pygame.display.set_mode((pm.SCREEN_W, pm.SCREEN_H))

def save(surface, name):
    path = os.path.join(OUT_DIR, name)
    pygame.image.save(surface, path)
    print(f"  saved {path}")

def fresh_game():
    """Return a Game instance reset to its initial state."""
    g = pm.Game.__new__(pm.Game)
    g.screen  = pygame.display.get_surface()
    g.clock   = pygame.time.Clock()
    g.font    = pygame.font.SysFont("monospace", 16, bold=True)
    g.sfont   = pygame.font.SysFont("monospace", 12)
    g.high    = 0
    g._new_game = lambda: None   # prevent re-init loop
    pm.Game._new_game(g)
    return g

def draw_and_save(game, filename):
    game._draw()
    pygame.display.flip()
    save(game.screen, filename)

# ── screenshots ────────────────────────────────────────────────────

screen = make_screen()

# 1. READY screen (initial state)
print("1/5  Ready screen …")
g = fresh_game()
g._state = "READY"
draw_and_save(g, "01_ready_screen.png")

# 2. Gameplay — Pac-Man mid-maze, dots intact
print("2/5  Gameplay (start of level) …")
g = fresh_game()
g._state = "PLAYING"
g.pac.dir  = pm.RIGHT
g.pac.want = pm.RIGHT
# advance a short distance
for _ in range(30):
    g.pac.update(1/60, g.grid)
draw_and_save(g, "02_gameplay_start.png")

# 3. Frightened ghosts (power pellet eaten)
print("3/5  Frightened ghosts …")
g = fresh_game()
g._state = "PLAYING"
g._activate_fright()
# put ghosts into active chase so they show blue
for gh in g.ghosts:
    if gh.state == pm.GS_INACTIVE:
        gh.state = pm.GS_FRIGHTENED
draw_and_save(g, "03_frightened_ghosts.png")

# 4. Death animation mid-frame
print("4/5  Death animation …")
g = fresh_game()
g._state  = "DYING"
g.pac.alive       = False
g.pac.death_frame = 22   # mid-shrink
draw_and_save(g, "04_death_animation.png")

# 5. Game Over screen with a score
print("5/5  Game Over screen …")
g = fresh_game()
g._state = "GAMEOVER"
g.score  = 14_820
g.high   = 14_820
draw_and_save(g, "05_game_over.png")

pygame.quit()
print("\nDone — screenshots saved to screenshots/")
