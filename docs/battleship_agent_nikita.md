# Battleship agent (`bots/battleship_agent_nikita.py`)

A probabilistic Battleship agent for the moving-ships variant of the tournament. It keeps a Bayesian belief over where the enemy ships are, shoots the cell most likely to hold an undamaged segment, uses sonar when the belief is too flat, and moves its own ships away from where a model of the opponent would shoot next.

Overall result: **91.5 % wins over 1300 games against 13 opponent types**, including **92/100 against `bots/battleship_agent.py`** (the original agent, kept unchanged as the baseline). All three required algorithms pass `check_requirements`.

---

## 1. Game facts the design relies on

| Fact (from the judge) | Consequence |
|---|---|
| A ship that has been hit can no longer move. | Hits are permanent evidence. Hit ships are tracked exactly, without a motion model. |
| Each turn a player may move one unhit ship one cell. | Between two of our shots at most one enemy ship moves one cell. A miss is strong evidence at first and loses value over time. |
| `SUNK` reports only the ship length. | Cruiser and submarine (both length 3) cannot be told apart. The belief handles this ambiguity. |
| Sonar counts all ship segments in a 3x3 area, damaged ones included, and is correct with probability `p` (otherwise off by one). | Sonar likelihood has to include the segments of known hit ships. |
| Shooting an already damaged cell returns `MISS` with `already_damaged`. | Those records are ignored by the filter. |
| 5 s total clock per player per game (inter-process communication included), 150 rounds. | All computation is vectorised numpy. Expensive features switch off when the clock gets tight. |

---

## 2. Strategies

### 2.1 Hunting belief for unhit ships (factorised HMM)

For each ship length `L` and orientation `H`/`V` the agent keeps a 10x10 grid `u[L][o]`: the probability that an unhit ship of that length has its top-left cell there. This is a hidden Markov model per ship:

- **Transition (predict)**: after every action, each grid is pushed through `predict_ship_belief`. With probability `q` the ship moves one cell in a uniformly chosen legal direction, otherwise it stays. `q = MOVE_RATE / expected number of mobile ships`, because only one ship moves per enemy turn.
- **Observation (update)**: a miss zeroes every placement covering that cell. A hit additionally forbids unhit ships from covering it, because the cell now belongs to a hit ship.

From the grids, `coverage()` gives the probability that a ship of each length covers each cell.

### 2.2 Exact hypotheses for hit ships

Hit ships do not move, so they are tracked exactly. The belief holds a weighted set of joint hypotheses. Each hypothesis is a tuple of `(L, row, col, orientation)` placements explaining all hits so far.

- **Hit on a cell not covered by a hypothesis**: the hypothesis branches into every placement through that cell (`consistent_placements`), weighted by the current HMM probability of that placement.
- **Hit, miss or `SUNK` on a covered cell**: the hypothesis survives only if it is consistent. For `SUNK`, the ship must be fully hit and have the reported length. For a plain `HIT`, it must not be complete yet.
- **Weights** are multiplied by the probability that the unhit ships do not cover the shot cell, so hypotheses that leave fewer ships unaccounted for are favoured correctly.
- At most `HYP_CAP` hypotheses are kept. If every hypothesis is eliminated (possible after pruning), `_rebuild` reconstructs them by backtracking search over the hit cells. `_history_allows` checks that a hit ship never saw a miss on its own cells after its first hit, and was reported sunk exactly when its last cell was hit.

`cell_probs()` combines both parts: the exact probability that a hypothesised hit ship has an undamaged segment on the cell, plus the probability that some unhit ship covers it. Damaged cells are marked `-1`.

### 2.3 Model averaging over placement styles (counter to edge, center and cluster placement)

The plain HMM assumes every legal placement is equally likely, and that assumption favours the center. Opponents who put all ships on the border, pack them together, or keep them central would cost us extra rounds. The agent therefore runs several complete beliefs in parallel, one per placement style (`PRIOR_MODELS`):

| Model | `bias` | `touch` | Prior weight |
|---|---|---|---|
| Uniform | 0.0 | 1.0 | 0.55 |
| Edge-heavy | -1.5 | 1.0 | 0.15 |
| Center-heavy | +0.7 | 1.0 | 0.15 |
| Clustered | 0.0 | 3.0 | 0.15 |

- `bias` multiplies the initial weight of each placement by `exp(bias * mean distance of its cells from the border)`. Negative values favour edges, positive values favour the center.
- `touch` multiplies the weight of placements next to an already hit ship. Values above 1 model clustered fleets.
- After every shot, each model's weight is multiplied by the probability that model gave to the observed result (hit probability for a hit, its complement for a miss, and the marginal reading probability for sonar). Shots aim at the weight-averaged cell probabilities.
- Models whose weight falls below `MODEL_PRUNE` are dropped permanently, which keeps the cost low once the opponent's style is clear.

Effect against a mid-strength hunter, 100 games per placement style:

| Opponent placement | Rounds to win, single uniform model | Rounds to win, with model averaging |
|---|---|---|
| Random (judge default) | 43.3 | 44.5 |
| All ships on edges | 48.5 | 37.2 |
| Center-heavy | 40.9 | 42.5 |
| Clustered, touching | 47.0 | 39.0 |

It costs about 1–1.5 rounds against random and center-heavy placement and saves 8–11 rounds against edge and cluster placement. Overall win rate is unchanged, and the agent can no longer be exploited by those placements.

### 2.4 Sonar

- **Bayes update (`Belief.sonar`)**: the true count in the 3x3 area is the hypothesis's fixed segments plus a convolution of each unhit ship's count distribution. `sonar_posterior` updates the count distribution from the noisy reading, and the result is pushed back onto each ship's placement grid and each hypothesis weight.
- **When to scan (`Agent._sonar_choice`)**: only when the best shot has hit probability below `SONAR_TRIGGER`. For each of the 64 interior centers, the agent estimates the expected best hit probability after seeing the reading, treating cells as independent and vectorising over all centers. It scans if the best value exceeds `SONAR_GAIN` times the best shot available now.

### 2.5 Own placement

`PLACE_TRIES` random layouts are generated with no two ships touching (`_random_layout`), and the one with the lowest total hit probability under a uniform hunter's belief is chosen. More tries push ships harder towards the edges, which is predictable. Testing showed no measurable benefit from more than a few tries, so a low value keeps the layout nearly random.

### 2.6 Hiding moves (shadow opponent)

The agent runs a "shadow" belief on its own board from the opponent's point of view (`SHADOW_RATES`):

- Real hits on our fleet are known from `damage_received` and applied exactly.
- On every other opponent turn we don't know where they shot, so the shadow assumes they missed on its own most likely cell.

Each turn the agent picks the legal move that most reduces the risk of its unhit ships: the sum over their cells of `shadow probability ^ RISK_POWER`, multiplied by `1 + EDGE_RISK` on border cells. If no move reduces risk, it doesn't move. Hiding is the single most valuable feature. Without it the win rate drops from about 95 % to about 79 %.

The border weight hedges against hunters that deliberately search the edges. It raised the result against such a hunter from 42/80 to 59/80, at a small cost against plain density hunters.

### 2.7 Time management

- `TIME_RESERVE` seconds are never spent.
- A turn is **rich** when the spare clock exceeds what cheap turns would need for the rest of the game (`rounds left * POOR_TURN`) plus `TIME_SAFETY` times the measured cost of a rich turn.
- Rich turns run everything. Cheap turns keep only the most likely placement model and skip the shadow, hiding and sonar. They only update the belief and shoot.
- The cost of a rich turn is measured as it runs (moving average), so the agent adapts to slower machines. With the clock cut to 2 s, which is like a machine 2.5x slower, it still wins 27/30 against `base`.

---

## 3. Function reference

### Module-level helpers

| Function | Purpose |
|---|---|
| `_cells(row, col, orientation, length)` | Cells occupied by a ship placement. |
| `_shift(a, dr, dc)` | Shift an array over its last two axes, filling with zeros. Used for all moves and coverage sums. |
| `_valid(n, length, orientation)` | 0/1 grid of top-left cells where the ship fits on the board. |
| `_covered_by(mask, length, orientation)` | For each top-left cell, how many cells of `mask` the placement covers. |
| `_coverage(grid, length, orientation)` | Converts a top-left belief grid into the probability that each cell is covered. |
| `consistent_placements(length, board_size, blocked, required)` | **Required function.** Every placement of one ship that avoids `blocked` and covers all `required` cells. |
| `_reading_given_count(count, reading, region_cells, correct_probability)` | Sonar noise model: P(reading given true count), renormalised at the bounds 0 and region size. |
| `sonar_posterior(prior, reading, region_cells, correct_probability)` | **Required function.** Exact Bayes update of the count distribution from one reading. |
| `_move_options(n, length, orientation)` | Cached: which directions are legal from each top-left cell, and how many. |
| `predict_ship_belief(belief, length, orientation, move_probability)` | **Required function, the transition model.** One step: stay with `1 - move_probability`, otherwise move to a uniformly chosen legal neighbour. A ship with no legal move stays. |
| `_depth(n)` | Distance of each cell from the nearest border. |
| `_sonar_region(center, n)` | 0/1 mask of the 3x3 area around a sonar center. |
| `act(observation)` | Entry point. Creates the `Agent` in the placement phase, then delegates. Any exception is logged to stderr and answered with a fallback (random placement or `_fallback_shot`), because an exception would otherwise lose the game. |
| `_fallback_shot(observation)` | Random cell not shot yet. Used only after an internal error. |

### `class Belief(n, lengths, move_rate, bias=0.0, touch=1.0)`

| Method | Purpose |
|---|---|
| `_normalize(L)` | Rescales the length-`L` grids to sum to 1, resetting to uniform if everything was eliminated. |
| `_hyp(key)` | Cached decode of a hypothesis: cell-to-ship map, number of unhit ships per length, occupancy mask. |
| `coverage()` | Per length, the probability that an unhit ship covers each cell. |
| `predict()` | Transition step for all grids, with move probability divided among the expected mobile ships. |
| `_free(nv, c)` | Probability that none of the `nv` unhit ships covers a cell, given per-length coverage `c`. |
| `shot(cell, result, sunk_length)` | Full observation update: branches and filters hypotheses, clears placements and applies the `touch` factor. |
| `_set_hyps(nxt)` | Normalises and keeps the `HYP_CAP` heaviest hypotheses, or rebuilds if none survive. |
| `_rebuild()` | Backtracking search for hypotheses consistent with all hits and the shot history. |
| `_history_allows(L, r, c, o)` | Checks one immobile-ship placement against the full shot history. |
| `sonar(center, reading, correct_probability)` | Bayes update from a scan. Returns the marginal probability of the reading, used for model weights. |
| `cell_probs()` | P(undamaged segment) per cell. Damaged cells are `-1`. |

### `class Agent(obs)`

| Method | Purpose |
|---|---|
| `placement()` | Chooses our layout (section 2.5). |
| `_random_layout()` | One random layout with no touching ships, or `None` if sampling failed. |
| `_sync(obs)` | Replays our new shot and sonar records into every placement model and updates model weights. |
| `_cell_probs()` | Weight-averaged cell probabilities across placement models. |
| `_sync_shadow(obs)` | Replays the opponent's turns into the shadow beliefs about our fleet. |
| `_hide(obs)` | Picks the risk-minimising move (section 2.6). |
| `_sonar_choice(obs, p, best_p)` | Picks a sonar center or `None` (section 2.4). |
| `act(obs)` | One turn: time check, sync, optional hiding move, then sonar or the best shot. |

---

## 4. Constants

| Name | Value | Meaning |
|---|---|---|
| `DIRS` | N/S/E/W offsets | Move directions as `(row change, column change)`. |
| `MOVE_RATE` | 0.05 | Assumed probability per turn that the enemy moves a ship at all, spread over its mobile ships. Low because evidence goes stale slowly in practice. 0.02 and 0.12 tested the same. |
| `SHADOW_RATES` | `(1.0,)` | Move rates for the shadow beliefs about our fleet. 1.0 makes the shadow treat old misses as stale quickly, which suits hunters that re-shoot. |
| `HYP_CAP` | 300 | Maximum number of joint hypotheses kept for hit ships. |
| `PLACE_TRIES` | 3 | Random layouts compared when placing our fleet. |
| `EDGE_RISK` | 1.0 | Extra risk weight on border cells when hiding. |
| `PRIOR_MODELS` | 4 models | `(bias, touch, prior weight)` per placement style (section 2.3). |
| `MODEL_PRUNE` | 0.02 | Placement models below this weight are dropped. |
| `SONAR_TRIGGER` | 0.2 | Sonar is considered only when the best shot's hit probability is below this. |
| `SONAR_GAIN` | 1.5 | Required ratio of expected post-scan hit probability to the best shot now. |
| `RISK_POWER` | 3 | Exponent on shadow probabilities in the hiding risk, so the most exposed cells dominate. |
| `TIME_RESERVE` | 0.6 s | Clock never spent. |
| `TIME_SAFETY` | 3.0 | Multiple of the measured rich-turn cost kept free before running a rich turn. |
| `POOR_TURN` | 0.008 s | Budget reserved per remaining round for cheap turns. |

Main attributes: `Belief.u` (placement grids), `Belief.allowed` (placements still possible for unhit ships), `Belief.hyps` (hypothesis weights), `Belief.hits`/`hit_mask` (damaged cells), `Belief.near_mask` (cells next to hits, used by `touch`), `Belief.events` (shot history), `Agent.models`/`model_w` (placement models and their weights), `Agent.shadows` (opponent's view of our fleet), `Agent.edge_weight` (hiding risk weights), `Agent.turn_cost` (measured rich-turn cost).

---

## 5. Results

### 5.1 Test opponents

| Name | Description |
|---|---|
| `base` | `bots/battleship_agent.py`, the original agent. |
| `demo_a`, `demo_b` | Example teams shipped with the tournament. |
| `dens_static` | Classic density hunter that treats misses as permanent; ships never move. |
| `dens_evasive` | Density hunter, forgets misses after 12 turns, center-heavy placement, moves ships away from danger. |
| `dens_random` | Density hunter, forgets misses after 20 turns, random moves half the time, uses sonar. |
| `dens_mem6` | Density hunter, forgets misses after 6 turns, random moves 30 % of the time. |
| `dens_edge` | Density hunter that also gives extra weight to edge cells, forgets misses after 15 turns, random moves 30 %. |
| `parity` | Checkerboard search with line-following once it hits, early sonar, random moves half the time. |
| `place_random`, `place_edge`, `place_center`, `place_cluster` | The same mid-strength hunter with random, all-on-edge, center-heavy or clustered (touching) placement; ships never move. |

### 5.2 Final evaluation (100 games per opponent, fresh seeds)

| Opponent | Wins / 100 | Average rounds |
|---|---|---|
| `base` (`battleship_agent.py`) | **92** | 61 |
| `demo_a` | 96 | 58 |
| `demo_b` | 100 | 46 |
| `dens_static` | 97 | 45 |
| `dens_evasive` | 88 | 60 |
| `dens_random` | 83 | 66 |
| `dens_mem6` | 90 | 61 |
| `dens_edge` | 69 | 53 |
| `parity` | 86 | 60 |
| `place_random` | 95 | 45 |
| `place_edge` | 97 | 40 |
| `place_center` | 98 | 42 |
| `place_cluster` | 99 | 38 |
| **Total** | **1190 / 1300 = 91.5 %** | |

For comparison, `battleship_agent.py` itself wins 40/50 against `demo_a` and 42/50 against `demo_b`. It sometimes shoots the same cell over and over and fails to find the last ship.

### 5.3 Real tournament runner (subprocesses, real clock)

| Opponent | Wins | Minimum clock left | Mean clock left | Incidents |
|---|---|---|---|---|
| `battleship_agent.py` | 14/15 | 3.42 s | 3.95 s | none |
| `demo_a` | 15/15 | 3.39 s | 3.93 s | none |
| `demo_b` | 15/15 | 3.97 s | 4.18 s | none |

The largest clock use in the 1300 in-process games was 2.64 s of 5 s.

### 5.4 What the experiments showed

| Change | Effect on win rate |
|---|---|
| Remove hiding moves | About 95 % to 79 %. The most important feature. |
| Shadow move rate 0.3 instead of 1.0 | 95.2 % to 92.0 %. |
| Remove sonar | About 1 point lower. |
| Fully random placement instead of anti-density | Within noise. |
| Our assumed move rate 1.0 instead of 0.05 | 92.9 % to 56.3 %. Treating misses as stale too fast wastes shots. |
| Bank of several move rates for the enemy | No gain, removed. |
| Prior penalising touching ships | No effect, replaced by the clustered placement model. |
| Edge-weighted hiding | Same total; `dens_edge` 42/80 to 59/80, small losses elsewhere. |
| Model averaging over placement styles | Same total; 8–11 fewer rounds against edge and cluster placement. |
| Measured-cost time management | With a 2 s clock: 14/30 to 27/30 against `base`. |

### 5.5 Known weaknesses

- `dens_edge` (69 %) and `dens_random` (83 %) are the hardest opponents. Both are hunters with edge bias or long miss memory that also move their ships often.
- The shadow assumes the opponent always misses on its most likely cell. An opponent with a very different search pattern makes hiding less effective, but never harmful, because a move is only made when it reduces estimated risk.
