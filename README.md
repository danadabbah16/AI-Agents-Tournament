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
