# PAC-MAN — Atari/Roklan 1982 · Source Code & Python Port

This repository preserves the original **Atari Pac-Man** source code developed by **Roklan Corp** for Atari Inc. (Revision 3.0, 10/03/82), together with a faithful **Python translation** of that code into a playable modern game.

---

## The Story of Pac-Man

The history of Pac-Man begins not with a joystick, but with a pizza.

In 1979, **Toru Iwatani**, a game designer at Namco, was looking for a concept that would break the mold of space shooters and war games that dominated arcades. Staring at a pizza missing one slice, the image clicked: a round, chomping character navigating a maze. His goal was deliberate — create a game welcoming to everyone, not just the usual young male arcade crowd.

**May 1980** — Pac-Man launches in Japanese arcades under the name *Puck-Man* (パックマン). The name was changed to *Pac-Man* for the North American release to prevent obvious vandalism of the cabinet lettering. The concept is deceptively simple: guide Pac-Man through a maze, eat all the dots, dodge four colorful ghosts — Blinky, Pinky, Inky, and Clyde — and use power pellets to briefly turn the tables on your pursuers.

The game spreads like wildfire. By 1981, **Pac-Man Fever** — a literal pop song by Buckner & Garcia — hits the charts. The character appears on lunchboxes, Saturday morning cartoons, board games, and clothing. Pac-Man becomes the first video game character to cross into mainstream popular culture.

**1982** — Atari licenses Pac-Man for its home consoles. Roklan Corp is contracted to develop the Atari 5200 version, producing a port far more faithful to the arcade original than the infamous Atari 2600 release. The source code preserved here — `REVISION 3.0, 10/03/82` — is that very work: six 6502 assembly files, hand-crafted by the Roklan team under strict confidentiality.

Spin-offs follow: *Ms. Pac-Man* (1982) becomes one of the best-selling arcade games ever made, joined by *Super Pac-Man*, *Pac-Land*, *Pac-Mania*, and eventually *Pac-Man Championship Edition*. The franchise never truly went away. Decades later, Pac-Man's silhouette needs no introduction — the "waka-waka" sound is as recognizable as a dial tone.

---

## Repository Contents

```
Pacman/
├── Atari_version/          # Original 6502 assembly source (Roklan Corp, 1982)
│   ├── PACMAN.ASM          # Main game loop, option/attract mode, DLI/VBI
│   ├── PAC1.ASM            # VBlank subroutines: attract, collision, death
│   ├── PAC2.ASM            # Rerack/level-clear, ready sequence
│   ├── PAC3.ASM            # Pac-Man movement (PMSTIK), dot eating (MUNCHY), scoring
│   ├── PAC4.ASM            # Ghost AI: MONSTR / MCHASE / GOHOME / MDIRCT / EYONLY
│   ├── PACDAT1.ASM         # Maze tile data (DATMAZ)
│   ├── PACDAT2.ASM         # Sprite / animation data
│   ├── PACDAT3.ASM         # Additional data tables
│   ├── ATARISYS.ASM        # Atari system definitions
│   └── SYSTEXT.ASM         # Text / display routines
│
├── Python_version/
│   ├── pacman.py           # Python translation of the full assembly codebase
│   ├── requirements.txt    # pygame >= 2.0.0
│   └── take_screenshots.py # headless renderer used to generate screenshots/
│
├── AI_player/
│   ├── environment.py      # gym-style headless wrapper around pacman.py
│   ├── dqn_agent.py        # QNetwork, ReplayBuffer, DQNAgent
│   ├── train.py            # training loop with checkpointing
│   ├── play.py             # watch a saved model play
│   └── requirements.txt    # torch >= 2.0, numpy, pygame
│
└── screenshots/            # rendered gameplay stills
```

### Atari Version — Roklan Corp (Revision 3.0)

The assembly source is annotated with original Roklan labels and register names (e.g., `PMHPOS`, `PMVPOS`, `COLPF2`, `DLICNT`). The header in `PACMAN.ASM` reads:

```
; PAC-MAN
; Developed for Atari Inc. by Roklan Corp.
; This information is confidential and is not for sale or distribution.
; 10/03/82 — DISK VERSION — REVISION 3.0
```

The code targets the **Atari 5200** hardware, exploiting its custom chips — ANTIC (display list), GTIA (sprites/color registers), and POKEY (sound/input) — to produce a close arcade replica. The six `PAC*.ASM` files map directly to the game's functional modules, with `PACDAT*.ASM` holding all static table data.

### Python Version

`pacman.py` is a line-by-line conceptual translation of the assembly into Python + **pygame**, written to be readable alongside the original source. Every class, method, and constant references its assembly counterpart:

| Assembly file | Python equivalent |
|---|---|
| `PACMAN.ASM` | `Game` class (main loop, state machine) |
| `PAC1.ASM` | `Game._update_*` (VBlank subroutines) |
| `PAC2.ASM` | `Game._check_*` (level-clear, ready sequence) |
| `PAC3.ASM` | `PacMan` class (movement, dot eating, scoring) |
| `PAC4.ASM` | `Ghost` class (MONSTR AI, chase/scatter/fright/eaten) |
| `PACDAT1.ASM` | `MAZE_DEF` (28×31 tile grid) |

Ghost AI faithfully reproduces the four classic personalities:
- **Blinky** — direct pursuit (always targets Pac-Man's current tile)
- **Pinky** — ambush (targets 4 tiles ahead of Pac-Man)
- **Inky** — flanking (2-tile lookahead reflected around Blinky)
- **Clyde** — shy (chases when far, retreats to corner when within 8 tiles)

---

## Screenshots

| Ready Screen | Gameplay |
|:---:|:---:|
| ![Ready screen](screenshots/01_ready_screen.png) | ![Gameplay start](screenshots/02_gameplay_start.png) |

| Frightened Ghosts | Death Animation |
|:---:|:---:|
| ![Frightened ghosts](screenshots/03_frightened_ghosts.png) | ![Death animation](screenshots/04_death_animation.png) |

| Game Over |
|:---:|
| ![Game over](screenshots/05_game_over.png) |

---

## Running the Python Version

**Requirements:** Python 3.8+, pygame

```bash
pip install -r Python_version/requirements.txt
python Python_version/pacman.py
```

### Controls

| Key | Action |
|---|---|
| Arrow keys / WASD | Move Pac-Man |
| P | Pause / Resume |
| Q / Esc | Quit |
| Enter | New game (on Game Over screen) |

---

## AI Player — Deep Q-Network (DQN)

The `AI_player/` folder contains a reinforcement learning agent that learns to play Pac-Man from scratch — no hard-coded rules, only the raw game signal.

### How it works

The agent uses a **Deep Q-Network**, the same algorithm DeepMind used to play Atari games at human level (Mnih et al., 2015). It observes the current game state as an 81-dimensional feature vector, selects one of four actions (UP / DOWN / LEFT / RIGHT), and updates its policy based on the reward received:

| Event | Reward |
|---|---|
| Eating a dot | +1.0 |
| Eating a power pellet | +5.0 |
| Eating a ghost | +20 / +40 / +80 / +160 |
| Dying | −50.0 |
| Clearing a level | +100.0 |
| Each step taken | −0.05 |

**Observation vector (81 floats):**
- Pac-Man position and direction (6)
- 4 ghosts — position, state (frightened / eaten / inactive), distance (24)
- Fright timer and dots remaining (2)
- 7×7 local maze view centred on Pac-Man (49)

**Network architecture:**

```
Input (81) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(4 Q-values)
```

**Training details:**

| Hyperparameter | Value |
|---|---|
| Replay buffer | 100 000 transitions |
| Batch size | 64 |
| Discount γ | 0.99 |
| ε decay | 1.0 → 0.05 over 200 k steps |
| Target network sync | every 1 000 steps |
| Optimiser | Adam, lr = 1e-4 |
| Loss | Huber (SmoothL1) |
| Frame skip | 4 frames per action |

### Training the agent

```bash
pip install -r AI_player/requirements.txt
python AI_player/train.py --episodes 5000
```

Resume from a checkpoint:

```bash
python AI_player/train.py --episodes 10000 --load AI_player/checkpoints/dqn_ep5000.pt
```

Progress is printed every 100 episodes. Checkpoints land in `AI_player/checkpoints/`; the best model (highest mean score over the last 100 episodes) is saved as `dqn_best.pt`.

### Watching the agent play

```bash
python AI_player/play.py --model AI_player/checkpoints/dqn_best.pt
```

Add `--headless` to benchmark without opening a window:

```bash
python AI_player/play.py --model AI_player/checkpoints/dqn_best.pt --headless --episodes 20
```

### Learning curve

| Training stage | Typical behaviour |
|---|---|
| 0 – 1 000 eps | Random movement, dies quickly |
| 1 000 – 5 000 eps | Learns to collect dots, avoids most ghosts |
| 5 000 – 20 000 eps | Uses power pellets strategically, hunts frightened ghosts |
| 20 000+ eps | Consistent multi-level runs |

---

## Upstream Repository

The Atari assembly source is mirrored from:
**[github.com/DillonDepeel/Pacman-Source-Code](https://github.com/DillonDepeel/Pacman-Source-Code)**

That repository collects the original Roklan Corp files as they have circulated among preservation communities, along with the game's history documented in its own `README.md`.

---

## License

See `Atari_version/LICENSE` for the terms covering the original assembly source.
The Python translation is provided for educational and preservation purposes.
