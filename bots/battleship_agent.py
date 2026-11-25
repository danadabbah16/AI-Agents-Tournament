"""Battleship agent (v2: HMM belief filtering, CSP target resolution, Bayesian sonar, evasive movement).

Required functions (tested by check_requirements):
  - consistent_placements
  - sonar_posterior
  - predict_ship_belief
"""
import math
import random

FLEET_DEF = [("carrier", 5), ("battleship", 4), ("cruiser", 3), ("submarine", 3), ("destroyer", 2)]
DIRS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}


def _ship_cells(row, col, orientation, length):
    if orientation == "H":
        return [(row, col + i) for i in range(length)]
    return [(row + i, col) for i in range(length)]


def _sonar_region(center, board_size):
    cr, cc = center
    return [
        (r, c)
        for r in range(cr - 1, cr + 2)
        for c in range(cc - 1, cc + 2)
        if 0 <= r < board_size and 0 <= c < board_size
    ]


def _sonar_likelihood(true_count, reading, region_cells, correct_prob):
    error = (1.0 - correct_prob) / 2.0
    raw = {true_count: correct_prob, true_count - 1: error, true_count + 1: error}
    valid = {r: p for r, p in raw.items() if 0 <= r <= region_cells}
    total = sum(valid.values())
    if total <= 0:
        return 0.0
    return valid.get(reading, 0.0) / total


# ---------------------------------------------------------------------------------
# Required algorithms. check_requirements() tests these.
# ---------------------------------------------------------------------------------


def consistent_placements(length, board_size, blocked, required):
    """Every position of one ship consistent with the evidence."""
    blocked_set = {tuple(cell) for cell in blocked}
    required_set = {tuple(cell) for cell in required}
    placements = []

    for orientation in ("H", "V"):
        max_row = board_size - (length if orientation == "V" else 1)
        max_col = board_size - (length if orientation == "H" else 1)
        for row in range(max_row + 1):
            for col in range(max_col + 1):
                cells = _ship_cells(row, col, orientation, length)
                if blocked_set.intersection(cells):
                    continue
                if not required_set.issubset(cells):
                    continue
                placements.append([row, col, orientation])

    return placements


def sonar_posterior(prior, reading, region_cells, correct_probability):
    """Bayes' rule: P(true count | sonar reading)."""
    unnormalized = {}
    for key, prior_p in prior.items():
        likelihood = _sonar_likelihood(int(key), reading, region_cells, correct_probability)
        unnormalized[key] = prior_p * likelihood

    total = sum(unnormalized.values())
    if total <= 0:
        return dict(prior)
    return {key: val / total for key, val in unnormalized.items()}


def predict_ship_belief(belief, length, orientation, move_probability):
    """One prediction step of a belief filter for one unseen ship."""
    size = len(belief)
    new_belief = [[0.0] * size for _ in range(size)]

    def fits(r, c):
        if orientation == "H":
            return 0 <= r < size and 0 <= c and c + length <= size
        return 0 <= r and r + length <= size and 0 <= c < size

    for row in range(size):
        for col in range(size):
            mass = belief[row][col]
            if mass == 0.0:
                continue
            options = [
                (row + dr, col + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if fits(row + dr, col + dc)
            ]
            if not options:
                new_belief[row][col] += mass
            else:
                new_belief[row][col] += mass * (1.0 - move_probability)
                step_mass = mass * move_probability / len(options)
                for nr, nc in options:
                    new_belief[nr][nc] += step_mass

    return new_belief


# ---------------------------------------------------------------------------------
# Agent State & Inference Engine
# ---------------------------------------------------------------------------------


class _BeliefTracker:
    """Maintains HMM beliefs for moving ships and CSP clusters for immobile hit ships."""

    def __init__(self, board_size):
        self.size = board_size
        self.reset()

    def reset(self):
        self.alive_lengths = [5, 4, 3, 3, 2]
        # Per-ship belief: list of {"length": L, "H": grid, "V": grid}
        self.ship_beliefs = [self._uniform_ship(L) for L in self.alive_lengths]
        self.sunk_cells = set()
        # Active hit clusters: list of {"cells": set(), "first_turn": int, "misses_after": set()}
        self.hit_clusters = []
        self.processed_shots = 0
        self.processed_sonar = 0
        self.last_turn_seen = -1

    def _uniform_ship(self, length):
        h_grid = [[0.0] * self.size for _ in range(self.size)]
        v_grid = [[0.0] * self.size for _ in range(self.size)]
        h_count = self.size * (self.size - length + 1)
        v_count = (self.size - length + 1) * self.size
        total = h_count + v_count
        p = 1.0 / total
        for r in range(self.size):
            for c in range(self.size - length + 1):
                h_grid[r][c] = p
        for r in range(self.size - length + 1):
            for c in range(self.size):
                v_grid[r][c] = p
        return {"length": length, "H": h_grid, "V": v_grid}

    def _normalize_ship(self, sb):
        total = sum(sum(row) for row in sb["H"]) + sum(sum(row) for row in sb["V"])
        if total <= 1e-12:
            fresh = self._uniform_ship(sb["length"])
            sb["H"], sb["V"] = fresh["H"], fresh["V"]
            return
        inv = 1.0 / total
        for o in ("H", "V"):
            grid = sb[o]
            for r in range(self.size):
                for c in range(self.size):
                    grid[r][c] *= inv

    def _predict_all(self, move_prob=0.15):
        for sb in self.ship_beliefs:
            L = sb["length"]
            sb["H"] = predict_ship_belief(sb["H"], L, "H", move_prob)
            sb["V"] = predict_ship_belief(sb["V"], L, "V", move_prob)
            # Zero out any placement overlapping known sunk cells
            if self.sunk_cells:
                self._mask_blocked(sb, self.sunk_cells)
            self._normalize_ship(sb)

    def _mask_blocked(self, sb, blocked):
        L = sb["length"]
        for r in range(self.size):
            for c in range(self.size - L + 1):
                if sb["H"][r][c] > 0:
                    if any((r, c + i) in blocked for i in range(L)):
                        sb["H"][r][c] = 0.0
        for r in range(self.size - L + 1):
            for c in range(self.size):
                if sb["V"][r][c] > 0:
                    if any((r + i, c) in blocked for i in range(L)):
                        sb["V"][r][c] = 0.0

    def _apply_miss(self, cell):
        # Record miss for active hit clusters (since hit ships are immobile!)
        for cl in self.hit_clusters:
            cl["misses_after"].add(cell)

        # Mask out cell from all moving ship beliefs
        r0, c0 = cell
        for sb in self.ship_beliefs:
            L = sb["length"]
            for c in range(max(0, c0 - L + 1), min(self.size - L + 1, c0 + 1)):
                sb["H"][r0][c] = 0.0
            for r in range(max(0, r0 - L + 1), min(self.size - L + 1, r0 + 1)):
                sb["V"][r][c0] = 0.0
            self._normalize_ship(sb)

    def _apply_hit(self, cell, turn):
        # Attach to an adjacent cluster if aligned, else create new cluster
        r, c = cell
        merged = False
        for cl in self.hit_clusters:
            if any(abs(r - hr) + abs(c - hc) == 1 for hr, hc in cl["cells"]):
                cl["cells"].add(cell)
                merged = True
                break
        if not merged:
            self.hit_clusters.append({"cells": {cell}, "first_turn": turn, "misses_after": set()})

    def _apply_sunk(self, cell, sunk_length):
        # Find which cluster(s) contain cell or form a line of sunk_length ending at cell
        r, c = cell
        all_hits = set(cell for cl in self.hit_clusters for cell in cl["cells"])
        all_hits.add(cell)

        # Look for a straight line of sunk_length cells in all_hits that contains cell
        chosen = None
        for dr, dc in ((0, 1), (1, 0)):
            line = [cell]
            for sign in (-1, 1):
                k = 1
                while (r + sign * k * dr, c + sign * k * dc) in all_hits:
                    line.append((r + sign * k * dr, c + sign * k * dc))
                    k += 1
            if len(line) >= sunk_length:
                # Prefer exact match or contiguous slice containing cell
                line.sort()
                idx = line.index(cell)
                start = max(0, min(idx, len(line) - sunk_length))
                chosen = set(line[start:start + sunk_length])
                break

        if chosen is None:
            # Fallback: take cluster containing cell or nearest cells
            chosen = {cell}
            for cl in self.hit_clusters:
                if any(abs(r - hr) + abs(c - hc) == 1 for hr, hc in cl["cells"]):
                    chosen.update(cl["cells"])
                    break

        self.sunk_cells.update(chosen)
        # Rebuild hit_clusters without chosen cells
        remaining_hits = all_hits - chosen
        self.hit_clusters = []
        for h in sorted(remaining_hits):
            self._apply_hit(h, 0)

        # Remove one ship of sunk_length from alive_lengths and ship_beliefs
        if sunk_length in self.alive_lengths:
            idx = self.alive_lengths.index(sunk_length)
            self.alive_lengths.pop(idx)
            self.ship_beliefs.pop(idx)
        for sb in self.ship_beliefs:
            self._mask_blocked(sb, self.sunk_cells)
            self._normalize_ship(sb)

    def _apply_sonar(self, center, reading, correct_prob):
        region = _sonar_region(center, self.size)
        region_set = set(region)
        region_cells = len(region)

        # Known damaged segments in region (sunk + active hits)
        damaged = self.sunk_cells | {c for cl in self.hit_clusters for c in cl["cells"]}
        fixed_count = sum(1 for cell in region if cell in damaged)

        # Probability distribution of how many segments each moving ship puts in region
        ship_dists = []
        for sb in self.ship_beliefs:
            L = sb["length"]
            dist = [0.0] * (L + 1)
            for r in range(self.size):
                for c in range(self.size - L + 1):
                    p = sb["H"][r][c]
                    if p > 0:
                        cnt = sum(1 for i in range(L) if (r, c + i) in region_set)
                        dist[cnt] += p
            for r in range(self.size - L + 1):
                for c in range(self.size):
                    p = sb["V"][r][c]
                    if p > 0:
                        cnt = sum(1 for i in range(L) if (r + i, c) in region_set)
                        dist[cnt] += p
            ship_dists.append(dist)

        # Convolve to get total count distribution
        total_dist = {fixed_count: 1.0}
        for dist in ship_dists:
            nxt = {}
            for c1, p1 in total_dist.items():
                for c2, p2 in enumerate(dist):
                    if p2 > 0:
                        nc = min(region_cells, c1 + c2)
                        nxt[nc] = nxt.get(nc, 0.0) + p1 * p2
            total_dist = nxt

        prior = {str(k): v for k, v in total_dist.items()}
        post = sonar_posterior(prior, reading, region_cells, correct_prob)

        # Update each ship's belief softly using marginal likelihood of its segment count
        for idx, sb in enumerate(self.ship_beliefs):
            # Distribution of all OTHER ships
            other_dist = {fixed_count: 1.0}
            for j, dist in enumerate(ship_dists):
                if j == idx:
                    continue
                nxt = {}
                for c1, p1 in other_dist.items():
                    for c2, p2 in enumerate(dist):
                        if p2 > 0:
                            nc = min(region_cells, c1 + c2)
                            nxt[nc] = nxt.get(nc, 0.0) + p1 * p2
                other_dist = nxt

            L = sb["length"]
            count_weight = [0.0] * (L + 1)
            for k_self in range(L + 1):
                w = 0.0
                for k_other, p_other in other_dist.items():
                    tot = min(region_cells, k_self + k_other)
                    # Likelihood * posterior alignment
                    w += p_other * _sonar_likelihood(tot, reading, region_cells, correct_prob)
                count_weight[k_self] = w

            for r in range(self.size):
                for c in range(self.size - L + 1):
                    if sb["H"][r][c] > 0:
                        cnt = sum(1 for i in range(L) if (r, c + i) in region_set)
                        sb["H"][r][c] *= count_weight[cnt]
            for r in range(self.size - L + 1):
                for c in range(self.size):
                    if sb["V"][r][c] > 0:
                        cnt = sum(1 for i in range(L) if (r + i, c) in region_set)
                        sb["V"][r][c] *= count_weight[cnt]
            self._normalize_ship(sb)

    def sync(self, observation):
        turn = observation["turn"]
        if turn < self.last_turn_seen:
            self.reset()
        self.last_turn_seen = turn

        shots = observation["my_shots"]
        sonars = observation["my_sonar"]

        # Predict once per new round if we have already acted before
        if self.processed_shots + self.processed_sonar > 0:
            self._predict_all(move_prob=0.15)

        while self.processed_shots < len(shots) or self.processed_sonar < len(sonars):
            next_shot_turn = shots[self.processed_shots]["turn"] if self.processed_shots < len(shots) else 10**9
            next_sonar_turn = sonars[self.processed_sonar]["turn"] if self.processed_sonar < len(sonars) else 10**9

            if next_shot_turn <= next_sonar_turn:
                shot = shots[self.processed_shots]
                self.processed_shots += 1
                cell = tuple(shot["target"])
                res = shot["result"]
                if res == "MISS":
                    if not shot.get("already_damaged"):
                        self._apply_miss(cell)
                elif res == "HIT":
                    self._apply_hit(cell, shot["turn"])
                elif res == "SUNK":
                    self._apply_sunk(cell, shot["sunk_length"])
            else:
                scan = sonars[self.processed_sonar]
                self.processed_sonar += 1
                self._apply_sonar(
                    tuple(scan["center"]),
                    scan["reading"],
                    observation.get("sonar_correct_probability", 0.8),
                )

    def cell_probabilities(self):
        """Returns 10x10 probability grid of an undamaged moving ship occupying each cell."""
        prob = [[0.0] * self.size for _ in range(self.size)]
        for sb in self.ship_beliefs:
            L = sb["length"]
            for r in range(self.size):
                for c in range(self.size - L + 1):
                    p = sb["H"][r][c]
                    if p > 0:
                        for i in range(L):
                            prob[r][c + i] += p
            for r in range(self.size - L + 1):
                for c in range(self.size):
                    p = sb["V"][r][c]
                    if p > 0:
                        for i in range(L):
                            prob[r + i][c] += p
        return prob


_TRACKER = None


def _choose_placement(observation):
    size = observation["board_size"]
    fleet = observation["fleet"]
    best_placements = None
    best_score = -1e9

    for _ in range(40):
        occupied = set()
        placements = []
        ok = True
        for item in fleet:
            ship, length = item["ship"], item["length"]
            placed = False
            for _ in range(30):
                o = random.choice(("H", "V"))
                r = random.randrange(size - (length - 1 if o == "V" else 0))
                c = random.randrange(size - (length - 1 if o == "H" else 0))
                cells = _ship_cells(r, c, o, length)
                if not occupied.intersection(cells):
                    occupied.update(cells)
                    placements.append({"ship": ship, "row": r, "col": c, "orientation": o})
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        # Score: prefer high initial mobility (legal moves) and spread between ships
        moves = 0
        for p in placements:
            ship_c = set(_ship_cells(p["row"], p["col"], p["orientation"], dict((f["ship"], f["length"]) for f in fleet)[p["ship"]]))
            others = occupied - ship_c
            for dr, dc in DIRS.values():
                nc = _ship_cells(p["row"] + dr, p["col"] + dc, p["orientation"], len(ship_c))
                if all(0 <= rr < size and 0 <= cc < size for rr, cc in nc) and not others.intersection(nc):
                    moves += 1
        # Penalize adjacent ships (reduces sonar cluster hits)
        adj_penalty = 0
        for r, c in occupied:
            for dr, dc in ((1, 0), (0, 1)):
                if (r + dr, c + dc) in occupied:
                    adj_penalty += 1
        score = moves * 2.0 - adj_penalty * 0.5
        if score > best_score:
            best_score = score
            best_placements = placements

    return {"placements": best_placements}


def _csp_target(tracker):
    """If there are active unsunk hit clusters, use consistent_placements to pick the best shot."""
    if not tracker.hit_clusters:
        return None

    size = tracker.size
    all_hits = {c for cl in tracker.hit_clusters for c in cl["cells"]}
    damaged = tracker.sunk_cells | all_hits

    # Pick the largest/oldest active hit cluster to finish first
    cl = max(tracker.hit_clusters, key=lambda x: (len(x["cells"]), -x["first_turn"]))
    blocked = tracker.sunk_cells | cl["misses_after"]
    required = list(cl["cells"])

    cell_scores = {}
    for length in set(tracker.alive_lengths):
        weight = tracker.alive_lengths.count(length)
        placements = consistent_placements(length, size, blocked, required)
        for r, c, o in placements:
            for cr, cc in _ship_cells(r, c, o, length):
                if (cr, cc) not in damaged:
                    cell_scores[(cr, cc)] = cell_scores.get((cr, cc), 0) + weight

    if cell_scores:
        best_score = max(cell_scores.values())
        best_cells = [cell for cell, sc in cell_scores.items() if sc == best_score]
        return random.choice(best_cells)

    # Fallback: orthogonal neighbor of any hit not yet damaged
    for hr, hc in cl["cells"]:
        for dr, dc in DIRS.values():
            nr, nc = hr + dr, hc + dc
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in damaged:
                return (nr, nc)
    return None


def _best_sonar_center(tracker, cell_probs, correct_prob):
    """Pick a 3x3 sonar center that maximizes expected information gain."""
    size = tracker.size
    best_center = None
    best_ig = -1.0
    scanned_centers = set()  # Avoid rescanning same center if needed

    for r in range(1, size - 1):
        for c in range(1, size - 1):
            region = _sonar_region((r, c), size)
            # Approximate count distribution via cell probabilities
            # Clamp expected count and compute variance/entropy proxy
            probs = [min(0.99, max(0.0, cell_probs[rr][cc])) for rr, cc in region]
            exp_sum = sum(probs)
            if exp_sum < 0.35:
                continue
            # Bernoulli-sum DP for count distribution
            dist = [1.0]
            for p in probs:
                nxt = [0.0] * (len(dist) + 1)
                for k, v in enumerate(dist):
                    nxt[k] += v * (1.0 - p)
                    nxt[k + 1] += v * p
                dist = nxt

            # Mutual information I(Count; Reading) = H(Reading) - H(Reading | Count)
            n_cells = len(region)
            read_dist = {}
            h_cond = 0.0
            for k, pk in enumerate(dist):
                if pk < 1e-6:
                    continue
                for rd in (k - 1, k, k + 1):
                    if 0 <= rd <= n_cells:
                        lik = _sonar_likelihood(k, rd, n_cells, correct_prob)
                        if lik > 0:
                            read_dist[rd] = read_dist.get(rd, 0.0) + pk * lik
                            h_cond -= pk * lik * math.log2(lik)
            h_read = -sum(pr * math.log2(pr) for pr in read_dist.values() if pr > 1e-9)
            ig = h_read - h_cond
            if ig > best_ig:
                best_ig = ig
                best_center = (r, c)

    return best_center


def _choose_evasive_move(observation):
    legal_moves = observation["legal_moves"]
    if not legal_moves:
        return None

    size = observation["board_size"]
    my_ships = {s["ship"]: s for s in observation["my_ships"]}
    damaged_all = {tuple(d["cell"]) for d in observation["damage_received"]}

    # Score each legal move: prefer moving ships that have high mobility, away from damaged cells
    best_move = None
    best_score = -1e9
    for mv in legal_moves:
        ship = my_ships[mv["ship"]]
        dr, dc = DIRS[mv["direction"]]
        new_cells = [(r + dr, c + dc) for r, c in ship["cells"]]
        # Distance to board center & away from damaged cells
        min_dmg_dist = min((abs(r - dr2) + abs(c - dc2) for r, c in new_cells for dr2, dc2 in damaged_all), default=10)
        # Prefer interior cells (more future escape options)
        edge_dist = min(min(r, size - 1 - r, c, size - 1 - c) for r, c in new_cells)
        score = min_dmg_dist * 0.5 + edge_dist * 0.8 + random.random() * 0.3
        if score > best_score:
            best_score = score
            best_move = mv

    return best_move


def act(observation):
    global _TRACKER
    if observation["phase"] == "placement":
        _TRACKER = _BeliefTracker(observation["board_size"])
        return _choose_placement(observation)

    size = observation["board_size"]
    if _TRACKER is None or _TRACKER.size != size:
        _TRACKER = _BeliefTracker(size)

    _TRACKER.sync(observation)
    time_left = observation.get("time_remaining_s") or 5.0

    damaged = _TRACKER.sunk_cells | {c for cl in _TRACKER.hit_clusters for c in cl["cells"]}
    move = _choose_evasive_move(observation) if time_left > 0.3 else None

    # 1. Target mode: finish any hit (immobile) ship using CSP
    target = _csp_target(_TRACKER)
    if target is not None:
        action = {"shot": [target[0], target[1]]}
        if move is not None:
            action["move"] = move
        return action

    # 2. Compute cell occupancy probabilities from HMM beliefs
    cell_probs = _TRACKER.cell_probabilities()

    # 3. Strategic Sonar: use in Hunt mode when uncertainty is high & time permits
    sonar_left = observation.get("sonar_remaining", 0)
    max_p = max(
        cell_probs[r][c]
        for r in range(size)
        for c in range(size)
        if (r, c) not in damaged
    )
    if (
        sonar_left > 0
        and time_left > 1.0
        and len(_TRACKER.alive_lengths) >= 2
        and (observation["turn"] <= 4 or max_p < 0.30)
    ):
        center = _best_sonar_center(_TRACKER, cell_probs, observation.get("sonar_correct_probability", 0.8))
        if center is not None:
            action = {"sonar": [center[0], center[1]]}
            if move is not None:
                action["move"] = move
            return action

    # 4. Hunt mode: shoot highest-probability undamaged cell (with slight parity tie-break)
    best_cell = None
    best_val = -1.0
    for r in range(size):
        for c in range(size):
            if (r, c) in damaged:
                continue
            val = cell_probs[r][c] + (1e-4 if (r + c) % 2 == 0 else 0.0)
            if val > best_val:
                best_val = val
                best_cell = (r, c)

    if best_cell is None:
        best_cell = (0, 0)

    action = {"shot": [best_cell[0], best_cell[1]]}
    if move is not None:
        action["move"] = move
    return action
