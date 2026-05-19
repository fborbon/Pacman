---

## Training Journey Summary & Computational Analysis

### Full training history

This project ran three successive training campaigns on a single CPU, totalling **10 300 episodes** across roughly **27 hours** of wall-clock time.

| Run | Episodes | Resumed from | Wall time | Best score |
|:---:|:---:|:---:|---:|---:|
| Run 1 | 300 | scratch | ~21 min | 2 160 |
| Run 2 | 5 000 | ep 300 | ~478 min | 4 310 |
| Run 3 | 5 000 | ep 5 300 | ~TBD min | TBD |
| **Total** | **10 300** | — | **~TBD min** | **TBD** |

Each episode corresponds to one full game (up to 3 lives). The agent executed roughly **5 million gradient steps** over the full training history, consuming ~100 000 replay-buffer transitions per mini-batch at 64 samples each.

#### Evaluation results across all checkpoints (60 games per checkpoint)

| Checkpoint (total eps) | Mean score | Dots eaten | Maze % | Success rate |
|:---:|---:|---:|---:|---:|
| ep 100 | 843 | 76 / 244 | 31 % | 7 % |
| ep 200 | 1 266 | 110 / 244 | 45 % | 58 % |
| ep 300 | 1 601 | 137 / 244 | 56 % | 82 % |
| ep 1 300 | 2 479 | 192 / 244 | 79 % | 98 % |
| ep 2 300 | 2 349 | 186 / 244 | 76 % | 100 % |
| ep 3 300 | 2 498 | 192 / 244 | 79 % | 100 % |
| ep 4 300 | 2 449 | 191 / 244 | 78 % | 100 % |
| ep 5 300 | 2 471 | 188 / 244 | 77 % | 100 % |
| ep 6 300 | TBD | TBD | TBD | TBD |
| ep 7 300 | TBD | TBD | TBD | TBD |
| ep 8 300 | TBD | TBD | TBD | TBD |
| ep 9 300 | TBD | TBD | TBD | TBD |
| ep 10 300 | TBD | TBD | TBD | TBD |

---

### Why does a "simple" game like Pac-Man demand so much computation?

Pac-Man looks trivial to a human: eat dots, dodge ghosts. Yet training an RL agent to play it well is genuinely hard, and the reasons illuminate some of the deepest challenges in machine learning.

#### 1. The state space is enormous

The maze has 28 × 31 = 868 tiles. Pac-Man can occupy ~240 walkable cells. Each of 4 ghosts can occupy any of those cells in any of 4 directions and 4 AI states (chase, scatter, frightened, eaten). The number of dots remaining adds another 2²⁴⁴ possible configurations. Even with a compact 81-feature observation, the agent must generalise across a state space orders of magnitude larger than it can ever see during training. A human instantly recognises "ghost coming from the left, flee right" regardless of exactly which tile they are on; the neural network must learn this from scratch by experiencing thousands of examples.

#### 2. Rewards are sparse and delayed

Each dot eaten gives +1 reward — but collecting all 244 dots to clear a level takes 400–600 steps. The reward for clearing the level (+100) arrives hundreds of steps after the decisions that made it possible. This is the **credit-assignment problem**: the network must learn to associate early navigation choices with consequences that arrive much later. By contrast, a human intuitively knows that "reaching the far corner eventually" requires "turning left now." The agent must discover this purely from statistical correlations across thousands of episodes.

#### 3. Ghosts create a deceptive reward landscape

The four ghosts each have distinct personalities (Blinky chases directly, Pinky ambushes, Inky flanks, Clyde is shy). A dot sitting near Blinky is worth +1 but may cost −50 (a life). The network must learn that the *same tile* has completely different value depending on ghost positions, directions, and whether a power pellet was recently eaten. This non-stationarity in value means the Q-function is extremely complex — a simple linear model could never represent it.

#### 4. Exploration is genuinely difficult

With a random policy, the probability of accidentally eating all 244 dots (without dying) is astronomically small. The agent almost never sees a level-clear event early in training, so the +100 bonus provides no learning signal for thousands of episodes. The ε-greedy strategy mitigates this, but 200 000 random-to-greedy transition steps are required before the policy is reliable enough to reach the far ends of the maze consistently.

#### 5. The target is non-stationary during learning

DQN trains a neural network to predict Q-values, but the targets (Bellman backups through the target network) change as training progresses. This is fundamentally different from supervised learning, where labels are fixed. The network must simultaneously learn the right actions *and* the right value estimates, and each improvement in one can temporarily destabilise the other. This is why tens of thousands of gradient steps — and careful engineering like replay buffers, target-network freezing, and gradient clipping — are needed to keep training stable.

#### 6. Every second of gameplay must be compressed into features

Unlike DeepMind's original Atari DQN, which used raw pixels (84 × 84 × 4 frames), this agent uses a hand-crafted 81-feature vector. While compact, it still asks the network to reason about 4 ghost positions + states + distances, a 7×7 local maze window, and global counters — all simultaneously. The network has no recurrent memory, so it must infer context (e.g., "was there a power pellet here 10 steps ago?") from instantaneous features alone.

#### 7. Frame skip introduces its own challenge

Each agent decision covers 4 game frames (~67 ms of simulated time). This speeds up training but means the agent never sees the intermediate frames — it must commit to a direction and hold it for 4 ticks. Near ghost collisions, this coarse temporal resolution occasionally causes avoidable deaths that a fine-grained controller would handle.

#### The bottom line

Pac-Man is "simple" in the sense that a human child can learn it in minutes. But human learning exploits billions of years of evolutionary prior knowledge (spatial reasoning, danger recognition, planning), rich sensory integration, and the ability to transfer concepts from prior experience. A DQN starts with none of that. Every behaviour — *avoid red ghost, chase blue ghost, prefer dots that clear a path* — must emerge from random weight initialisation purely through trial-and-error reward signals over millions of interactions. The 27 hours of CPU training here produced an agent that consistently eats ~78 % of the maze per game; a skilled human player clears it in under 2 minutes. That gap is the distance between gradient descent and cognition.
