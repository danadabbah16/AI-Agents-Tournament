# AI Agent Tournament

## Members

- Salome (Catan)
- Dana (Catan)
- Nikita (Battleship)
- Gabi (Battleship)
- Jinin (Chess)
- Ruben (Road Trip)

## Algorithms implemented

| Challenge | Algorithms | Where in the file |
|---|---|---|
| Road Trip | | `roadtrip_agent.py` |
| Chess | | `chess_agent.py` |
| Battleship | | `battleship_agent.py` |
| Catan | Priority heuristic (baseline); goal-directed planner; Q-learning | `heuristic_policy`, `smart_policy`, `act`, `state_features`, `q_update`, `action_value` in `catan_agent.py` |

## Training

**Catan:** offline Q-learning with `train_catan.py` (Dana).

- 20,000 games vs frozen v1 (`bots/catan_v1.py`), sides A/B alternating
- α = 0.1, γ = 0.95, ε 0.30 → 0.05
- Reward: VP gained between decisions; +5 win / −5 loss
- About 9 minutes; 431 keys kept (visited ≥30 times)
- Q-table stored in `catan_agent.py`
- In a match, `act()` uses `smart_policy` (stronger than Q alone)

## Tests

**Catan**

- `check_requirements` on the agent: **4/4**
- Improved (`act` / `smart_policy`) vs v1: **~80.6%** over 500 games (sides balanced)
- Same agent vs demo_A: **~84.7%** over 300 games
- Q-only vs v1: **~47.1%** over 1000 games (`train_catan.py --eval` seeds / Dana’s eval)

Other challenges: fill in rivals, game counts, and `check_requirements` output.

## Dependencies and known limitations

Standard library only in our Catan agent.  
Catan: dice luck; small Q-state (no full board); 5s clock so no deep search.

## Technical explanation

### Catan

#### Problem formulation
1v1 to 7 VP. Observation includes phase, legal actions, me/opponent, board. Return `{"action": id}`. Clock: 5 seconds (`time_remaining_s`).

#### Algorithms
- **Baseline (Salome):** `heuristic_policy` — pip openings, build city/settlement first, trade only to unlock a build, robber prefers steal. Frozen as `bots/catan_v1.py`.
- **Improved (Dana):** `smart_policy` — pick a goal (city / settlement / road / dev), score spots, plan roads (BFS), trade toward the goal, better robber; late game pushes VP; knights when useful. `act()` calls this.
- **Q-learning (Dana):** `state_features` = `[phase, my VP, lead −2..+2, hand bucket]`; learn action types; `q_update` / `action_value` / table in file.

#### Baseline against improved version

| Matchup | Games | Result |
|---|---|---|
| `smart_policy` vs v1 | 500 | ~80.6% improved |
| Q-only vs v1 | 1000 | ~47.1% Q |
| `act` vs demo_A | 300 | ~84.7% us |

We play with `smart_policy`. Q stays for the required functions and the learning section.

#### Time and memory
No training at import. Lookups/rules fit in 5s. On error, `act` falls back to `heuristic_policy`.

#### Limits
Luck; coarse Q-state; planner is hand-written, not full search.

### Road Trip
*(owner)*

### Chess
*(owner)*

### Battleship
*(owner)*

## Use of generative AI

Catan: used coding assistants for drafting/debugging. We can explain all algorithms ourselves.  
Other members: add your note here.
