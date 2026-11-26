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
| Chess | Minimax and alpha-beta (required); PeSTO tapered evaluation; iterative deepening with principal variation search, quiescence, transposition table, killers and history; null-move pruning, late move reductions, futility pruning and aspiration windows in the mailbox core; opening book; time manager | `evaluate`, `minimax`, `alphabeta`, `act`, `PythonChessEngine`, `MailboxEngine` in `chess_agent.py` |
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

**Chess**

- `check_requirements` on the agent: **3/3** (alpha-beta visits 89–313 nodes against 421–1,588 for minimax at depth 2)
- Against demo_A: **20/20** wins
- Self-play, 50 games at 5 s and 50 at 10 s: **0** illegal moves, crashes or time losses; lowest clock left 0.48 s
- Final (Core B) against Core A alone: 17 wins, 1 draw, 2 losses over 20 games
- Move generator: perft exact on 5 reference positions; in 300 random games (58,696 positions) its legal moves equal python-chess's at every ply
- Unit tests: `pytest tests/test_chess_agent.py` (39 tests, not submitted)

Other challenges: fill in rivals, game counts, and `check_requirements` output.

## Dependencies and known limitations

Standard library only in our Catan and Road Trip agents.  
Catan: dice luck; small Q-state (no full board); 5s clock so no deep search.  
Road Trip: not always optimal (3 of 400 unseen draws, 7–55 km above the optimum); no exact solver fits in 5 s of pure Python.  
Chess: python-chess 1.11.2 only; pure-Python search reaches depth 4–6 in about 0.1 s on a laptop, less on Colab; no full static exchange evaluation.

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

The map is a weighted, undirected graph G = (V, E): 105 cities (V) and 262 roads (E), each road weighted by its driving kilometres. It is the same in every match. A match gives a set R of 25 required cities, Madrid included.

- **State space:** the routes. A route is a sequence of cities that starts and ends in Madrid, where each consecutive pair is joined by a road.
- **Constraints:** every city in R appears in the route, and no city appears twice (except Madrid at both ends). Cities outside R may be used as intermediate stops, each at most once.
- **Objective:** minimise the sum of the road kilometres. The judge computes it from the roads, never from straight lines.
- **What the agent observes:** the whole problem. That is the map (cities with latitude and longitude, roads with km), R, and the time limit. It is a fully observable, deterministic, single-agent problem. The opponent does not affect our route.

We split the problem in two levels:
1. **Order:** in which order to visit the 25 required cities. This is a travelling salesman problem (TSP) on a 25×25 table D, where D[a][b] is the shortest road distance from a to b.
2. **Realisation:** turning an order into a real road route that repeats no city. This is what makes it harder than a plain TSP. For example, the shortest paths to Vilnius and back both run through Kaunas, so a route can only use one of them.

The cost of an order in the table is a **lower bound** on the cost of any real route that visits the cities in that order, because each leg can only be as short as or longer than its shortest path. So the optimal table tour is a lower bound on the whole problem. When our real route equals it, our route is optimal.

#### Algorithms

All code is in `bots/roadtrip_agent.py`. It uses pure Python and no extra libraries.

| What | Where | Why |
|---|---|---|
| Uniform-cost search | `ucs()` | Optimal shortest path with non-negative road lengths. A node counts as expanded when it is taken from the frontier |
| A\* with the Haversine heuristic | `astar()` | h(n) is the straight-line distance from n to the goal. It is admissible: every road is at least as long as the straight line between its ends, so no path is shorter than h. It is also consistent, by the triangle inequality on the sphere, so a closed list is safe. It finds the same km as UCS while expanding fewer nodes (check: 176 against 384 over the 8 test pairs) |
| Dijkstra from every city | `dijkstra_all()` | Unconstrained road distances between all 105 cities, in a few milliseconds. They give the 25×25 table D, and an exact heuristic for the next row |
| A\* for paths that avoid used cities | `path_avoiding()` | The workhorse: about 250,000 calls per match. h(n) is the **unconstrained** road distance from n to the goal. Blocking cities can only make a path longer, so h is admissible and consistent. It is much tighter than the straight line, so very few cities are expanded. This made the search 1.4× faster |
| Nearest-neighbour tour | `nearest_neighbour()` | Our **baseline**: a valid route within about 10 ms |
| Iterated local search on D | `iterated_local_search()`, `two_opt()`, `or_opt()` | See the next table |
| Leg-by-leg realisation | `realise()` | Each leg is the shortest path that avoids every city already used, and if possible the required cities not yet visited. We try each order in both directions |
| Leg re-routing | `refine()` | Re-routes each leg with the rest of the route fixed. A leg already equal to the unconstrained distance is skipped without a search, which made the whole agent 2.5× faster |
| Cheapest insertion with no-repeat paths | `best_insert()` | A sturdier builder, used when no order can be realised leg by leg (draw 9: every order failed). It inserts each city at the cheapest place where both new paths avoid the route, pruned by the lower bound dist(u,x) + dist(x,v) − km(u,v) |
| Large-neighbourhood search | `ruin_and_recreate()` | See the table after next |

The iterated local search on D works like this:

| Part | What we use |
|---|---|
| Neighbourhood | 2-opt (reverse a segment) and Or-opt (move a segment of 1–3 cities, possibly reversed) |
| Acceptance | First improvement until neither move helps, which reaches a local optimum |
| Escape from local optima | A double-bridge kick, a 4-opt move that 2-opt cannot undo in one step. We accept equal tours, and worse ones with probability 0.05, and return to the best tour every 200 kicks without progress. Random restarts come on top |

`local_search()` (the required function) is the same iterated local search on any distance table. In the check, it shortens a random 12-city tour from 16,810 to 7,480 km.

The large-neighbourhood search works on **real** routes, so every candidate already obeys the no-repeat rule:

| Part | What we use |
|---|---|
| Moves (`remove_city()`, `reverse_stretch()`, `move_stretch()`) | Ruin a cluster of up to 8 nearby required cities, or a string of up to 8 consecutive ones, and reinsert them by cheapest insertion (60%). Reverse a stretch of the route and reconnect both ends with new no-repeat paths, a real-road 2-opt (20%). Cut a stretch of 1–10 required cities, close the gap, and reinsert the stretch whole, possibly reversed, elsewhere: a real-road Or-opt (20%) |
| Acceptance | Simulated annealing, with a temperature that starts at 1% of the route length and falls linearly to zero. Only routes that pass `is_valid()` are accepted |
| Escape from local optima | 16 independent chains, started from the 16 best distinct routes of the previous phase. A chain without a new best for 150 moves restarts from a fresh random-order insertion route (`random_insertion_route()`) |

**How one match runs:**
1. Build the distance tables.
2. Submit the nearest-neighbour route.
3. Run iterated local search on D for 8% of the time, realising every new local optimum and submitting each shorter route. If the best route equals the lower bound, stop: it is optimal.
4. If no route exists after 0.3 s, use cheapest insertion.
5. Run the large-neighbourhood search until the deadline.

#### Baseline against improved version

**Reference: the true optimum.** To measure the agent honestly, we computed the **exact** optimum of 601 draws offline. We used OR-Tools CP-SAT with a circuit constraint in which skipped cities are self-loops, so every used city is visited exactly once. It is a measuring tool only: it is not in the agent, which may not import it. Each draw took about 1 s on 8 cores, and every one was proven optimal.

**Tuning set:** draws 0–200 (the training draw and seeds 1–200, generated as the judge does). **Validation set:** draws 201–600, never used for tuning. Each run had 5 seconds on a Ryzen 7 5800X.

| Version | What it adds | Tuning set: optimal | Extra km | Validation set: optimal | Extra km |
|---|---|---|---|---|---|
| Baseline | Nearest neighbour only (first submission) | — | 13,284 km on average over draws 1–30 | — | — |
| v1 | Search on D, insertion fallback, small ruin and recreate | 188 / 201 (93.5%) | 1,390 | — | — |
| v2 | Real-road 2-opt, string ruins, 4 chains, plus a bug fix (see below) | 194 / 201 | 347 | — | — |
| v3 | Exact A\* heuristic and skipping optimal legs (2.5× faster) | 196 / 201 | 333 | 392 / 400 (98.0%) | 633 |
| **v5 (final)** | Real-road Or-opt, 16 chains, restarts of stuck chains | **199 / 201 (99.0%)** | **29** | **397 / 400 (99.3%)** | **92** |

On draws 1–30 the final version averages 13,098 km against the baseline's 13,284 km (−1.4%). It improves 18 of the 30 draws, by up to 992 km. On the three validation draws it misses, it is 7–55 km above the optimum.

**Against the example agent** (`opponent="demo"`, which rebuilds random insertion routes for 4 s): we won every game we ran, by 2,400–3,500 km.

**What made the difference, and how we found it:**
1. **Exact reference solutions.** The agent's own long runs and the D lower bound both suggested v1 was already optimal. The exact solver showed it was not, on 6.5% of the draws.
2. **Comparing routes with the optimum.** The misses were not a matter of time: v1 found its route within 0.5 s and was stuck for the next 30 s. The optimal routes reached a city cluster through a different corridor, or moved a whole chain of cities to another part of the loop. That is what the new 2-opt and Or-opt moves on real routes do.
3. **Speed.** The exact A\* heuristic and skipping optimal legs gave 2.5× more search per second.
4. **Diversity.** More chains, plus restarts of stuck chains, solved the last hard draws. 150 moves before a restart worked better than 400 or 1,000.

**A bug the testing found:** when several removed cities are reinserted, the path to one of them can pass through another that is still waiting. Inserting that second one then visited it twice. The judge would have rejected those routes. Now an already-visited city is skipped, and the search accepts only routes that pass `is_valid()`.

#### Time and memory

- **Deadline:** 85% of `time_limit_s` (4.25 s of 5), so the last submission arrives well before the judge stops us.
- **First valid route:** within 0.31 s on every one of the 601 draws, and usually about 0.01 s.
- **Last improvement:** at the latest 2.7 s.
- **Slower CPUs:** with only 2 s, a stand-in for a CPU 2.5× slower, it is still optimal on 198 of 201 draws, with 43 km extra in total.
- **Submissions:** we submit only strictly shorter routes, each checked by our own `is_valid()` (Madrid at both ends, no repeats, all required cities, real roads). The judge rejected none.
- **Early stop:** if the route reaches the lower bound, the agent returns at once.
- **Memory:** distance tables over 105 cities and 16 routes, well under 10 MB.

#### Limits

- **Not always optimal:** on 3 of 400 unseen draws it ends 7–55 km above the optimum. An exact method would fix this, but a pure-Python exact solver (branch and bound over simple cycles) would be too slow for 5 s. The allowed libraries include no solver.
- **No proof inside the match:** the agent can prove optimality only when its route reaches the D lower bound. That bound ignores the no-repeat rule, so on most draws the agent cannot know it is optimal and keeps searching until the deadline.
- **Tuning:** the settings (16 chains, 150 moves before a restart, the move mix) were chosen on draws 0–200 and confirmed on 201–600. They are good settings for this map, not proven best ones.
- **Fixed random seed:** the agent is deterministic per draw apart from timing, which helps us reproduce results.


### Chess

#### Problem formulation

Two-player, zero-sum, deterministic and fully observable: standard chess, with every rule enforced by the judge through python-chess. A match plays two games with colours swapped, 5 seconds of decision time per player per game (or one game with a 10 s clock). An illegal move, a malformed answer, a crash or an empty clock loses the game, so the goals are, in order: never lose that way, then beat the other student engines.

`act(observation)` receives:

| Field | Meaning |
|---|---|
| `fen` | the position |
| `color` | `"white"` or `"black"` |
| `legal_moves` | every legal move in UCI, for example `"e2e4"` |
| `history` | every move so far in UCI |
| `ply`, `is_check` | public game state |
| `game_number`, `games_total` | which game this is |
| `time_remaining_s` | our clock |
| `opponent_time_remaining_s` | the rival's clock |

It returns `{"move": "<uci>"}`. Promotions name the piece (`"e7e8q"`). It never resigns.

#### Algorithms

All code is in `bots/chess_agent.py`. It imports only `chess` (python-chess 1.11.2), `math`, `random` and `time`.

**Required functions.** `minimax` and `alphabeta` are deliberately plain: a python-chess board, push/pop, the same natural move order, no tables and no randomness. They never touch the state used by `act()`.

| Function | Returns |
|---|---|
| `evaluate(fen)` | integer score from White's view; ±100000 for checkmate, 0 for a draw |
| `minimax(fen, depth)` | `{"value", "nodes"}` from plain minimax |
| `alphabeta(fen, depth)` | the same value as `minimax`, with fewer nodes |

**Two search cores** share the evaluation, the time manager, the opening book and the fail-safe wrapper:

| Core | What it is | Role |
|---|---|---|
| A (`PythonChessEngine`) | Alpha-beta (PVS) directly on `chess.Board` with push/pop | Always correct; permanent fallback |
| B (`MailboxBoard`, `MailboxEngine`) | 10x12 mailbox board, packed-integer moves, in-place make/unmake, incremental evaluation and hash | Main engine, roughly twice as fast |

Core B is only trusted while it agrees with python-chess. On every move:

1. The judge's `history` is replayed onto a python-chess board and onto the mailbox.
2. Both are checked against the FEN; the mailbox's incremental scores and hash are checked against a full recompute. A mismatch rebuilds it from the history, then from the FEN.
3. Core B's legal root moves are compared with `legal_moves`. If they ever differ, Core B is switched off for the rest of the game and Core A plays.

**How one move is chosen (`act()`):**

1. Everything runs inside a guard. Any exception, or a move that is not in `legal_moves`, is replaced by the fallback move: the last completed search's best move, else the best MVV-LVA capture, else the first legal move.
2. One legal move: play it at once.
3. Under 0.25 s on the clock: play the best one-ply static move, without a search.
4. Opening book while `ply < 24`: 19 mainline openings, weighted random choice, so two games against the same rival do not repeat a line by force.
5. Otherwise compute the time budget and search with Core B, or Core A if B is off.

**Evaluation features and weights:**

- PeSTO tapered piece-square tables (Chess Programming Wiki), with PeSTO material values: midgame P 82, N 337, B 365, R 477, Q 1025; endgame P 94, N 281, B 297, R 512, Q 936.
- Game phase: N and B count 1, R 2, Q 4; 24 at the start, capped at 24. Score = (midgame × phase + endgame × (24 − phase)) / 24.
- Bishop pair: +30 midgame, +50 endgame.
- Mop-up in pawnless endings against a bare king: 47 × the loser king's centre-Manhattan distance + 16 × (14 − distance between kings). For KBNK the centre term is replaced by 35 × (14 − distance to a corner of the bishop's colour).
- Tempo: +10 for the side to move (search only, not `evaluate(fen)`).

**Search techniques:**

| Part | What we use |
|---|---|
| Depth | Iterative deepening; the previous best move is searched first |
| Search | Principal variation search (negamax, fail-soft) in both cores |
| Horizon | Quiescence search on captures and queen promotions, MVV-LVA ordered, with delta pruning; all evasions when in check at the first quiescence ply |
| Memory | Transposition table (Zobrist hash in Core B, python-chess's position key in Core A) storing packed integers; mate scores adjusted by ply |
| Move ordering | TT move, queen promotions, captures by MVV-LVA (a pawn-defended victim worth less than the attacker sorts as a losing capture), two killers per ply, history |
| Tactics | Check extension, mate-distance pruning, mate scores |
| Draws | 50-move rule, repetition and insufficient material, scored with contempt: 10 to 30 cp depending on phase, 0 when we are 150 cp down and −30 (a draw is welcome) when we are 400 cp down |
| Core B only | Aspiration windows (±40 cp, widening), null-move pruning (R = 2 or 3), reverse futility pruning, futility pruning and late move reductions (Ethereal's table) |

#### Baseline against improved version

Minimax against alpha-beta at depth 2 on the judge's four check positions (same values):

| Position | Minimax nodes | Alpha-beta nodes |
|---|---|---|
| Start | 421 | 89 |
| Italian, Black to move | 1048 | 170 |
| Queen hanging on d5 | 1588 | 240 |
| Queen endgame | 801 | 313 |

Version comparison (5 s clocks, real judge processes, on a laptop):

| Version | What it adds | Result |
|---|---|---|
| v1 | the required alpha-beta at fixed depth 2 | baseline |
| v2 (Core A) | iterative deepening, TT, ordering, quiescence, book, time manager | 20 wins, 0 draws, 0 losses against v1 |
| v3 (Core B, final) | mailbox core with NMP, LMR, futility, aspiration | 17 wins, 1 draw, 2 losses against v2 |

Both v2 and v3 also won 20 of 20 games against `demo_A`. Depth reached and nodes per second are printed per move when `DEBUG_PRINTS = True`; measure them again on Colab for the presentation, since a laptop is several times faster.

#### Time and memory

- 0.5 s of the clock is always kept in reserve.
- Soft limit = usable time / moves to go (at least 18), ×1.3 in check, ×1.2 before ply 24.
- Hard limit = min(3 × soft, 20% of usable time). The clock is read every 64 nodes and the search aborts at the hard limit, keeping the best move of the last completed iteration.
- No new iteration starts after half of the soft limit (half of the hard limit when the score just dropped by 40 cp or more).
- Under 0.6 s (panic): depth 2, and at most 0.02 s or 5% of the time above 0.25 s per move, so the last quarter second is never spent on search. Under 0.25 s: no search at all.
- Every limit is a fraction of `time_remaining_s`, so a 10 s clock scales automatically.
- Memory: the transposition table is capped at 200,000 packed integers per game. Building the tables and the book at import takes about 0.05 s.

#### Limits

- Pure Python: depth 4 to 6 in about 0.1 s on a laptop, less on Colab.
- Repetition inside the search counts after two occurrences, a deliberate simplification.
- No full static exchange evaluation; losing captures are guessed from pawn defence only.
- The 50-move rule is checked before mate detection.

### Battleship
*(owner)*

## Use of generative AI

Catan: used coding assistants for drafting/debugging. We can explain all algorithms ourselves.  
Road Trip: used Claude Code (Anthropic) to help write, test and benchmark the agent and to draft its explanation.  
Chess: the plan and the code were produced with AI assistance (Claude). The team reviewed and tested the code, and is responsible for it.  
Other members: add your note here.
