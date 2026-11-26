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
| Road Trip | UCS; A\* (Haversine); Dijkstra distance table; iterated local search (2-opt, Or-opt, double-bridge); A\* with exact heuristic for no-repeat paths; cheapest insertion; large-neighbourhood search with simulated annealing | `ucs`, `astar`, `local_search`, `solve`, `path_avoiding`, `best_insert`, `ruin_and_recreate` in `roadtrip_agent.py` |
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

**Road Trip**

- `check_requirements` on the agent: **4/4** (A\* expands 176 nodes against 384 for UCS; `local_search` 16,810 → 7,480 km)
- Against the exact optimum (OR-Tools CP-SAT, offline only): optimal on **199/201** tuning draws and **397/400** unseen draws, 5 s each
- Against demo_A: won every game tried, by 2,400–3,500 km
- With only 2 s (slow-CPU stand-in): optimal on 198/201

Other challenges: fill in rivals, game counts, and `check_requirements` output.

## Dependencies and known limitations

Standard library only in our Catan and Road Trip agents.  
Catan: dice luck; small Q-state (no full board); 5s clock so no deep search.  
Road Trip: not always optimal (3 of 400 unseen draws, 7–55 km above the optimum); no exact solver fits in 5 s of pure Python.

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

#### Problem formulation
The map is a fixed weighted graph (105 cities, 262 roads). A route starts and ends in Madrid, visits all 25 required cities, uses real roads, and repeats no city. Goal: minimise total km. We split it into choosing the **order** of the required cities (a TSP on a 25×25 shortest-path table D) and **realising** that order as a road route with no repeated city. D's best tour is a lower bound: matching it proves optimality.

#### Algorithms
All in `bots/roadtrip_agent.py`, pure Python.

| What | Function | Note |
|---|---|---|
| UCS / A\* | `ucs`, `astar` | Haversine heuristic is admissible: no road is shorter than the straight line. A\* expands 176 nodes vs 384 for UCS |
| Distance table | `dijkstra_all` | All-pairs road distances, also used as an exact A\* heuristic for no-repeat paths (`path_avoiding`) |
| Order search | `iterated_local_search` | 2-opt + Or-opt to a local optimum, double-bridge kicks to escape it. `local_search` is the same method |
| Realisation | `realise`, `best_insert`, `refine` | Leg-by-leg paths avoiding used cities; cheapest insertion when that gets stuck |
| Route search | `ruin_and_recreate` | Large-neighbourhood search on real routes: remove and reinsert cities, real-road 2-opt and Or-opt, simulated annealing, 16 chains that restart when stuck |

#### Baseline against improved version
Measured against exact optima computed offline with OR-Tools (not used by the agent), 5 s per draw.

| Version | Tuning draws (201) | Unseen draws (400) |
|---|---|---|
| Baseline: nearest neighbour | 13,284 km avg (draws 1–30) | — |
| v1: table search + small ruins | 188 optimal, +1,390 km total | — |
| Final | **199 optimal, +29 km** (13,098 km avg on 1–30) | **397 optimal, +92 km** |

The gains came from real-road 2-opt/Or-opt moves, a 2.5× faster path search, and restarting stuck chains. Against demo_A we won every game, by 2,400–3,500 km.

#### Time and memory
Stops at 85% of the limit. First valid route within 0.31 s, last improvement by 2.7 s. With only 2 s it is still optimal on 198/201 draws. Memory well under 10 MB.

#### Limits
Not always optimal (3 of 400 unseen draws, 7–55 km over). An exact solver does not fit in 5 s of pure Python.

### Chess
*(owner)*

### Battleship
*(owner)*

## Use of generative AI

Catan: used coding assistants for drafting/debugging. We can explain all algorithms ourselves.  
Road Trip: used Claude Code (Anthropic) to help write, test and benchmark the agent and to draft its explanation.  
Other members: add your note here.
