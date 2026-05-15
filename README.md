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

---

### Libraries Used

| Library | Version | Role in this project |
|---|---|---|
| **PyTorch** (`torch`) | ≥ 2.0 | Defines and trains the neural network. Provides `nn.Module`, `nn.Linear`, `nn.SmoothL1Loss`, the `Adam` optimiser, automatic differentiation, and optional GPU acceleration via CUDA. |
| **NumPy** (`numpy`) | ≥ 1.24 | Builds the observation vector as a `float32` array, stacks replay-buffer samples into batches before handing them to PyTorch, and computes running-average statistics for logging. |
| **pygame** | ≥ 2.0 | Runs the game simulation. In training mode it drives an offscreen SDL surface (no window); in play mode it renders the full 448 × 568 game window at 60 FPS. |

> **Note on other AI technologies** — this project uses classic **Reinforcement Learning** only. Generative AI, large language models (LLMs), retrieval-augmented generation (RAG), speech-to-text, image diffusion, image classification, and chatbot frameworks are **not** used here because they solve fundamentally different problems (content generation, language understanding, audio transcription, pixel synthesis, supervised recognition, conversational interfaces). RL is the natural fit when an agent must learn a sequential decision policy purely through trial-and-error interaction with an environment.

---

### Data Processing Pipeline

The pipeline runs one full cycle every agent step (4 game frames):

**Step 1 — Raw game state**
The `Game` object from `pacman.py` holds the authoritative state: the 28 × 31 tile grid, Pac-Man's pixel position and direction, each ghost's position, AI state, and fright timer, plus the current score and remaining dots.

**Step 2 — Feature engineering (`_observe`)**
The raw state is converted into a fixed-length **81-dimensional `float32` numpy array** so PyTorch can process it. All values are normalised to `[0, 1]`:

| Slice | Size | Content |
|---|---|---|
| `[0:2]` | 2 | Pac-Man column and row ÷ grid dimensions |
| `[2:6]` | 4 | Pac-Man direction one-hot (UP / DOWN / LEFT / RIGHT) |
| `[6:30]` | 4 × 6 | Per ghost: col, row, is\_frightened, is\_eaten, is\_inactive, Manhattan distance to Pac-Man |
| `[30]` | 1 | Fright timer ÷ fright\_total |
| `[31]` | 1 | Dots remaining ÷ 244 (total dots) |
| `[32:81]` | 49 | 7 × 7 tile grid centred on Pac-Man — cell type ÷ 5 |

**Step 3 — Action selection (ε-greedy)**
The observation is fed to the Q-network, which outputs one Q-value per action (4 values). The agent picks the action with the highest Q-value (`argmax`) — or, with probability ε, a random action. ε starts at 1.0 (fully random) and decays linearly to 0.05 over 200 000 steps, balancing exploration vs. exploitation.

**Step 4 — Environment step**
The chosen action (direction) is applied to the game. The simulation runs 4 frames (`frame_skip = 4`) to reduce temporal correlation between observations, matching the approach from the original Atari DQN paper. State transitions (death, level clear) are handled transparently and translated into reward signals.

**Step 5 — Reward shaping**
The reward at each step combines:
- Scaled game score: `(score_after − score_before) × 0.1`
- Death penalty: `−50.0`
- Level-clear bonus: `+100.0`
- Per-step cost: `−0.05` (discourages idling)

| Game event | Reward |
|---|---|
| Eating a dot | +1.0 |
| Eating a power pellet | +5.0 |
| Eating a ghost (chain) | +20 / +40 / +80 / +160 |
| Dying | −50.0 |
| Clearing a level | +100.0 |
| Each step | −0.05 |

**Step 6 — Replay buffer**
The transition `(s, a, r, s', done)` is pushed into a `collections.deque` capped at 100 000 entries. Storing past transitions and sampling them randomly breaks the temporal correlations that would otherwise destabilise training.

**Step 7 — Mini-batch sampling & training**
Once the buffer holds at least 64 transitions, a random mini-batch is drawn and converted to PyTorch tensors. The Q-network predicts `Q(s, a)` for each sample; the frozen target network predicts `max Q(s', a')` to form the Bellman target `y = r + γ · max Q(s') · (1 − done)`. Huber loss between prediction and target is backpropagated through the Q-network via Adam. The target network weights are hard-copied from the Q-network every 1 000 steps.

---

### Data Flow Diagram

```mermaid
flowchart TD
    subgraph STEP1["① Game State → Observation  (environment.py)"]
        GS["Game Object\ngrid · PacMan · Ghosts · score · fright_timer"]
        FE["_observe\nfeature engineering"]
        OBS["Observation  s\n81 × float32  numpy array  •  all values in 0–1"]
        GS --> FE --> OBS
    end

    subgraph STEP2["② Action Selection  (dqn_agent.py)"]
        EPS{"random() < ε ?"}
        RND["Random action\nexploration"]
        QFW["Q-Network  forward pass\nLinear 81→256→ReLU→256→ReLU→4"]
        AMAX["argmax  →  best action\nexploitation"]
        ACT["Action  int 0–3\nUP · DOWN · LEFT · RIGHT"]
        OBS --> EPS
        EPS -->|"Yes  ε = 1.0 → 0.05"| RND --> ACT
        EPS -->|No| QFW --> AMAX --> ACT
    end

    subgraph STEP3["③ Environment Step  (environment.py)"]
        FSKIP["frame_skip = 4\n4 × _update at 1/60 s\nhandles DYING · LEVEL_CLEAR transitions"]
        NGS["Next Game State"]
        REW["Reward  r\nscore×0.1  ·  −50 death  ·  +100 level  ·  −0.05 step"]
        DONE["done  bool\nGAMEOVER  or  steps ≥ 10 000"]
        ACT --> FSKIP
        FSKIP --> NGS & REW & DONE
    end

    subgraph STEP4["④ Replay Buffer  (dqn_agent.py)\ndeque  maxlen = 100 000"]
        BUF[("Transition tuples\ns · a · r · s' · done")]
        OBS -->|s| BUF
        ACT -->|a| BUF
        REW -->|r| BUF
        NGS -->|s'| BUF
        DONE -->|done| BUF
    end

    subgraph STEP5["⑤ Training Step  (dqn_agent.py)"]
        SAMP["Sample mini-batch\n64 random transitions"]
        QPRED["Q-Network\nQ_pred = Q(s, a)"]
        TGT["Target Network\nq_next = max Q_target(s')"]
        BELL["Bellman target\ny = r + γ·q_next·(1−done)"]
        LOSS["Huber Loss  SmoothL1\nL(Q_pred, y)"]
        BACK["Backprop  +  grad clip  ≤ 10\nAdam  lr = 1e-4"]
        SYNC["Hard-copy weights\nQ-Net → Target Net\nevery 1 000 steps"]
        BUF --> SAMP --> QPRED & TGT
        TGT --> BELL --> LOSS
        QPRED --> LOSS --> BACK --> QPRED
        BACK -.->|"periodic sync"| SYNC -.-> TGT
    end

    NGS -->|"becomes next s"| STEP2
```

---

### The DQN Model — Design Rationale

**Why Reinforcement Learning, not supervised learning?**
There is no dataset of "correct moves" to learn from. The agent must discover strategies — when to flee, when to hunt, which dots to prioritise — purely through interaction. RL formalises this as a Markov Decision Process: the agent takes actions, the environment returns rewards, and the goal is to maximise cumulative discounted reward over time.

**Why DQN specifically?**
DQN (Mnih et al., *Nature* 2015) was chosen because:
- Pac-Man has a **small discrete action space** (4 directions), making Q-value enumeration cheap.
- The game state can be encoded as a **compact feature vector** (81 floats), so pixel-based convolutional approaches (used in the original DeepMind work) are unnecessary.
- DQN is **sample-efficient** relative to policy-gradient methods (PPO, A3C) at this problem scale.
- The two stabilising innovations — **experience replay** and a **target network** — make training robust without requiring distributed rollouts.

No generative, language, or vision-foundation models are involved; the problem does not call for them.

**Key strengths of the configuration**

| Design choice | Rationale |
|---|---|
| **Replay buffer 100 k** | Large enough to break temporal correlations; small enough to fit in CPU RAM. |
| **Frame skip = 4** | Reduces effective decision frequency so each action has time to have a visible effect; matches the Atari DQN paper. |
| **Target network, hard sync every 1 000 steps** | Provides stable regression targets; prevents the oscillation caused by chasing a moving target. |
| **Huber loss (SmoothL1)** | Less sensitive to outliers than MSE — important when rare events (eating 4 ghosts in a chain) produce large Q-value spikes. |
| **Gradient clipping ≤ 10** | Prevents exploding gradients from reward spikes. |
| **γ = 0.99** | High discount favours long-term planning (clearing levels) over myopic dot-chasing. |
| **ε-greedy linear decay 1.0 → 0.05 over 200 k steps** | Ensures thorough early exploration of the maze before exploitation begins. 5 % residual randomness prevents policy collapse on local optima. |
| **Adam lr = 1e-4** | Conservative learning rate for stable convergence; lower than default (3e-4) to account for non-stationary targets. |

**Network architecture**

```
Input (81)
    │
Linear(256) → ReLU
    │
Linear(256) → ReLU
    │
Linear(4)   →  Q-values for [UP, DOWN, LEFT, RIGHT]
```

Two hidden layers of 256 units provide enough capacity to represent ghost-avoidance and dot-collection strategies simultaneously without overfitting on the 81-feature input.

---

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

### Training Results — 300-Episode Run

Results span two training runs totalling **5 300 episodes** (~8 hours, CPU-only).  
Checkpoints at episodes 100, 200, 300, 1 000, 2 000, 3 000, 4 000 and 5 000 were each evaluated over 60 games.

#### Key statistics

| Checkpoint | Mean score | Dots eaten | Maze completion | Steps to milestone | Success rate |
|:---:|---:|---:|---:|---:|---:|
| Episode 100 | 843 | 76 / 244 | 31.1 % | — | 7 % |
| Episode 200 | 1 266 | 110 / 244 | 45.1 % | — | 58 % |
| Episode 300 | 1 601 | 137 / 244 | 56.1 % | — | 82 % |
| Episode 1 000 | 2 479 | 192 / 244 | 78.7 % | — | 98 % |
| Episode 2 000 | 2 349 | 186 / 244 | 76.2 % | — | **100 %** |
| Episode 3 000 | 2 498 | 192 / 244 | 78.7 % | — | **100 %** |
| Episode 4 000 | 2 449 | 191 / 244 | 78.3 % | — | **100 %** |
| Episode 5 000 | **2 471** | **188 / 244** | **77.0 %** | — | **100 %** |

> **Success** is defined as scoring ≥ 1 200 points in a single episode (≈ 120 dots eaten).  
> From episode 2 000 onward the agent hits that milestone in **100 %** of games.  
> The agent eats ~78 % of all maze dots consistently by ep 1 000+; full level clears require continued training beyond 5 000 episodes.

#### Score progression

Mean episode score more than doubles from episode 100 to 300, with variance shrinking as the policy stabilises.

![Score progression](AI_player/training_results/chart_score_progress.png)

#### Maze completion — dots eaten

The agent goes from eating 30 % of the maze at episode 100 to nearly 60 % at episode 300, with the distribution tightening noticeably.

![Dots eaten](AI_player/training_results/chart_dots_eaten.png)

#### Time to solve the labyrinth

Left panel: how many steps the agent needs to reach the 1 200-point milestone (only episodes where it succeeded).  
Right panel: what fraction of episodes hit the milestone at all.

The agent solves the milestone **49 % faster** by episode 300, and goes from succeeding in 1 in 20 games to succeeding in 9 in 10.

![Solve time](AI_player/training_results/chart_solve_time.png)

#### Successes and failures

Stacked bars show raw success / failure counts per checkpoint (out of 60 eval games).  
The dashed orange line tracks mean score on the right axis.

![Success and failures](AI_player/training_results/chart_success_fail.png)

#### Policy evolution — game screenshots every 30 episodes

Each frame is a snapshot of the agent playing after that many training episodes. The maze visibly empties further and further right as the policy matures.

![Progression grid](AI_player/training_results/chart_progression.png)

---

## Upstream Repository

The Atari assembly source is mirrored from:
**[github.com/DillonDepeel/Pacman-Source-Code](https://github.com/DillonDepeel/Pacman-Source-Code)**

That repository collects the original Roklan Corp files as they have circulated among preservation communities, along with the game's history documented in its own `README.md`.

---

## License

See `Atari_version/LICENSE` for the terms covering the original assembly source.
The Python translation is provided for educational and preservation purposes.
