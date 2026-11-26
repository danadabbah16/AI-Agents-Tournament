"""Battleship agent: per-ship HMM filter for unhit ships, exact hypothesis tracking for hit ships,
Bayesian sonar, model averaging over placement styles, shadow-opponent model for hiding moves."""
import random
import sys
import time
import traceback

import numpy as np

DIRS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
MOVE_RATE = 0.05
SHADOW_RATES = (1.0,)
HYP_CAP = 300
PLACE_TRIES = 3
EDGE_RISK = 1.0
PRIOR_MODELS = ((0.0, 1.0, 0.55), (-1.5, 1.0, 0.15), (0.7, 1.0, 0.15), (0.0, 3.0, 0.15))
MODEL_PRUNE = 0.02
SONAR_TRIGGER = 0.2
SONAR_GAIN = 1.5
RISK_POWER = 3
TIME_RESERVE = 0.6
TIME_SAFETY = 3.0
POOR_TURN = 0.008


def _cells(row, col, orientation, length):
    if orientation == "H":
        return [(row, col + i) for i in range(length)]
    return [(row + i, col) for i in range(length)]


def _shift(a, dr, dc):
    out = np.zeros_like(a)
    n = a.shape[-1]
    out[..., max(dr, 0):n + min(dr, 0), max(dc, 0):n + min(dc, 0)] = \
        a[..., max(-dr, 0):n + min(-dr, 0), max(-dc, 0):n + min(-dc, 0)]
    return out


def _valid(n, length, orientation):
    v = np.zeros((n, n))
    if orientation == "H":
        v[:, :n - length + 1] = 1.0
    else:
        v[:n - length + 1, :] = 1.0
    return v


def _covered_by(mask, length, orientation):
    """For each top-left cell, how many cells of mask the placement covers."""
    if orientation == "H":
        return sum(_shift(mask, 0, -i) for i in range(length))
    return sum(_shift(mask, -i, 0) for i in range(length))


def _coverage(grid, length, orientation):
    """Probability that each cell is covered, from a top-left belief grid."""
    if orientation == "H":
        return sum(_shift(grid, 0, i) for i in range(length))
    return sum(_shift(grid, i, 0) for i in range(length))


def consistent_placements(length, board_size, blocked, required):
    """Every placement of one ship that avoids blocked cells and covers all required cells."""
    blocked = {tuple(c) for c in blocked}
    required = {tuple(c) for c in required}
    if required:
        r0, c0 = next(iter(required))
        candidates = [(r0, c0 - i, "H") for i in range(length)] + [(r0 - i, c0, "V") for i in range(length)]
    else:
        candidates = [(r, c, o) for o in "HV" for r in range(board_size) for c in range(board_size)]
    result = []
    for r, c, o in candidates:
        cells = _cells(r, c, o, length)
        if not all(0 <= rr < board_size and 0 <= cc < board_size for rr, cc in cells):
            continue
        if blocked.intersection(cells) or not required.issubset(cells):
            continue
        result.append([r, c, o])
    return result


def _reading_given_count(count, reading, region_cells, correct_probability):
    error = (1.0 - correct_probability) / 2.0
    raw = {count: correct_probability, count - 1: error, count + 1: error}
    valid = {r: q for r, q in raw.items() if 0 <= r <= region_cells}
    return valid.get(reading, 0.0) / sum(valid.values())


def sonar_posterior(prior, reading, region_cells, correct_probability):
    """Exact Bayes update of P(true count) given a noisy sonar reading."""
    post = {key: p * _reading_given_count(int(key), reading, region_cells, correct_probability)
            for key, p in prior.items()}
    total = sum(post.values())
    return {key: v / total for key, v in post.items()} if total > 0 else dict(prior)


_MOVE_CACHE = {}


def _move_options(n, length, orientation):
    key = (n, length, orientation)
    if key not in _MOVE_CACHE:
        valid = _valid(n, length, orientation)
        can = {d: _shift(valid, -dr, -dc) for d, (dr, dc) in DIRS.items()}
        _MOVE_CACHE[key] = (can, sum(can.values()))
    return _MOVE_CACHE[key]


def predict_ship_belief(belief, length, orientation, move_probability):
    """One transition step: with move_probability the ship moves one cell in a uniformly chosen
    on-board direction, otherwise it stays."""
    b = np.asarray(belief, dtype=float)
    can, k = _move_options(b.shape[-1], length, orientation)
    share = np.where(k > 0, b / np.maximum(k, 1), 0.0) * move_probability
    out = np.where(k > 0, b * (1 - move_probability), b)
    for d, (dr, dc) in DIRS.items():
        out = out + _shift(share * can[d], dr, dc)
    return out


def _depth(n):
    """Distance of each cell from the nearest border."""
    idx = np.arange(n)
    return np.minimum.outer(np.minimum(idx, n - 1 - idx), np.minimum(idx, n - 1 - idx)).astype(float)


def _sonar_region(center, n):
    mask = np.zeros((n, n))
    r, c = center
    mask[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = 1.0
    return mask


class Belief:
    """Belief over an enemy fleet. Unhit ships: one HMM over top-left cells per length and orientation.
    Hit ships are immobile: a weighted set of joint hypotheses assigns them exact placements."""

    def __init__(self, n, lengths, move_rate, bias=0.0, touch=1.0):
        self.n = n
        self.fleet = {L: lengths.count(L) for L in set(lengths)}
        self.lengths = sorted(self.fleet, reverse=True)
        self.move_rate = move_rate
        self.touch = touch
        self.allowed = {L: {o: _valid(n, L, o) for o in "HV"} for L in self.lengths}
        self.u = {L: {o: self.allowed[L][o] * np.exp(bias * _covered_by(_depth(n), L, o) / L) for o in "HV"}
                  for L in self.lengths}
        for L in self.lengths:
            self._normalize(L)
        self.hyps = {(): 1.0}
        self.info = {}
        self.hits = set()
        self.hit_mask = np.zeros((n, n))
        self.near_mask = np.zeros((n, n))
        self.events = []

    def _normalize(self, L):
        u = self.u[L]
        if u["H"].sum() + u["V"].sum() <= 1e-300:
            for o in "HV":
                u[o] = self.allowed[L][o].copy()
        s = u["H"].sum() + u["V"].sum()
        for o in "HV":
            u[o] /= s

    def _hyp(self, key):
        if key not in self.info:
            nv = dict(self.fleet)
            mask = np.zeros((self.n, self.n))
            owner = {}
            for item in key:
                nv[item[0]] -= 1
                for c in _cells(item[1], item[2], item[3], item[0]):
                    owner[c] = item
                    mask[c] = 1.0
            self.info[key] = (owner, tuple(nv[L] for L in self.lengths), mask)
        return self.info[key]

    def coverage(self):
        """Per length: probability that an unhit ship of that length covers each cell."""
        return {L: _coverage(self.u[L]["H"], L, "H") + _coverage(self.u[L]["V"], L, "V") for L in self.lengths}

    def predict(self):
        mobile = sum(w * sum(self._hyp(k)[1]) for k, w in self.hyps.items())
        q = min(1.0, self.move_rate / max(1.0, mobile))
        for L in self.lengths:
            for o in "HV":
                self.u[L][o] = predict_ship_belief(self.u[L][o], L, o, q) * self.allowed[L][o]
            self._normalize(L)

    def _free(self, nv, c):
        return float(np.prod([(1.0 - c[i]) ** nv[i] for i in range(len(nv))]))

    def shot(self, cell, result, sunk_length=None):
        cov = self.coverage()
        cx = [cov[L][cell] for L in self.lengths]
        hit = result in ("HIT", "SUNK")
        new_hits = self.hits | {cell}
        nxt = {}
        for key, w in self.hyps.items():
            owner, nv, _ = self._hyp(key)
            if not hit:
                if cell not in owner:
                    nxt[key] = nxt.get(key, 0.0) + w * self._free(nv, cx)
                continue
            if cell in owner:
                L, r, c, o = owner[cell]
                done = new_hits.issuperset(_cells(r, c, o, L))
                if done == (result == "SUNK") and (not done or L == sunk_length):
                    nxt[key] = nxt.get(key, 0.0) + w * self._free(nv, cx)
                continue
            if result == "SUNK":
                continue
            for i, L in enumerate(self.lengths):
                if nv[i] == 0:
                    continue
                others = list(nv)
                others[i] -= 1
                base = w * nv[i] * self._free(others, cx)
                for r, c, o in consistent_placements(L, self.n, list(owner), [cell]):
                    p = self.u[L][o][r, c]
                    if self.touch != 1.0 and any((a + dr, b + dc) in owner for a, b in _cells(r, c, o, L)
                                                 for dr, dc in DIRS.values()):
                        p *= self.touch
                    if p > 0:
                        child = tuple(sorted(key + ((L, r, c, o),)))
                        nxt[child] = nxt.get(child, 0.0) + base * p
        self.events.append((cell, result, sunk_length))
        shot_mask = np.zeros((self.n, self.n))
        shot_mask[cell] = 1.0
        near = np.zeros((self.n, self.n))
        if hit:
            self.hits = new_hits
            self.hit_mask[cell] = 1.0
            near = (sum(_shift(shot_mask, dr, dc) for dr, dc in DIRS.values()) > 0) & (self.near_mask == 0)
            self.near_mask = np.maximum(self.near_mask, near)
        self._set_hyps(nxt)
        for L in self.lengths:
            for o in "HV":
                clear = _covered_by(shot_mask, L, o) == 0
                self.u[L][o] *= clear
                if hit:
                    self.allowed[L][o] = self.allowed[L][o] * clear
                    if self.touch != 1.0:
                        self.u[L][o] *= np.where(_covered_by(near, L, o) > 0, self.touch, 1.0)
            self._normalize(L)

    def _set_hyps(self, nxt):
        nxt = {k: w for k, w in nxt.items() if w > 0}
        if not nxt:
            nxt = self._rebuild()
        top = sorted(nxt.items(), key=lambda kv: -kv[1])[:HYP_CAP]
        total = sum(w for _, w in top)
        self.hyps = {k: w / total for k, w in top if w / total > 1e-9}

    def _rebuild(self):
        """Backtracking over hit cells (most constrained first) when tracking lost every hypothesis."""
        found = {}

        def search(key, weight, uncovered):
            if len(found) >= 2 * HYP_CAP:
                return
            if not uncovered:
                found[key] = found.get(key, 0.0) + weight
                return
            owner, nv, _ = self._hyp(key)
            best = None
            for cell in uncovered:
                opts = [(L, r, c, o) for i, L in enumerate(self.lengths) if nv[i] > 0
                        for r, c, o in consistent_placements(L, self.n, list(owner), [cell])
                        if self._history_allows(L, r, c, o)]
                if best is None or len(opts) < len(best[1]):
                    best = (cell, opts)
            for L, r, c, o in best[1]:
                cells = set(_cells(r, c, o, L))
                search(tuple(sorted(key + ((L, r, c, o),))), weight * (self.u[L][o][r, c] + 1e-6), uncovered - cells)

        search((), 1.0, set(self.hits))
        return found or {(): 1.0}

    def _history_allows(self, L, r, c, o):
        """An immobile ship never sees a miss after its first hit, and is reported sunk exactly
        when its last cell is hit."""
        cells = set(_cells(r, c, o, L))
        hit_times = [t for t, (x, res, _) in enumerate(self.events) if x in cells and res != "MISS"]
        if not hit_times:
            return False
        if any(x in cells and res == "MISS" and t > hit_times[0] for t, (x, res, _) in enumerate(self.events)):
            return False
        sunk = [t for t in hit_times if self.events[t][1] == "SUNK"]
        if len(hit_times) == L:
            return sunk == [hit_times[-1]] and self.events[hit_times[-1]][2] == L
        return not sunk

    def sonar(self, center, reading, correct_probability):
        """Bayes update from one scan: total count = fixed segments of the hypothesis + independent
        counts of unhit ships; the count posterior is pushed back onto each ship's placements."""
        region = _sonar_region(center, self.n)
        ncell = int(region.sum())
        counts = {L: {o: _covered_by(region, L, o).astype(int) for o in "HV"} for L in self.lengths}
        dist = {L: np.array([sum(self.u[L][o][counts[L][o] == k].sum() for o in "HV") for k in range(L + 1)])
                for L in self.lengths}
        groups = {}
        for key, w in self.hyps.items():
            owner, nv, mask = self._hyp(key)
            g = (int((mask * region).sum()), nv)
            groups[g] = groups.get(g, 0.0) + w

        def total(fixed, nv, skip=None):
            d = np.zeros(fixed + 1)
            d[fixed] = 1.0
            for i, L in enumerate(self.lengths):
                for _ in range(nv[i] - (1 if skip == i else 0)):
                    d = np.convolve(d, dist[L])
            if len(d) > ncell + 1:
                d = np.append(d[:ncell], d[ncell:].sum())
            return d

        factor = {L: np.zeros(L + 1) for L in self.lengths}
        likelihood = {}
        for (fixed, nv), gw in groups.items():
            t = total(fixed, nv)
            likelihood[(fixed, nv)] = sum(t[k] * _reading_given_count(k, reading, ncell, correct_probability)
                                          for k in range(len(t)))
            post = sonar_posterior({str(k): t[k] for k in range(len(t))}, reading, ncell, correct_probability)
            for i, L in enumerate(self.lengths):
                if nv[i] == 0:
                    continue
                rest = total(fixed, nv, skip=i)
                for own in range(L + 1):
                    factor[L][own] += gw * likelihood[(fixed, nv)] * sum(
                        post[str(k)] * rest[k - own] / t[k]
                        for k in range(own, min(len(t), len(rest) + own)) if t[k] > 0)
        marginal = sum(gw * likelihood[g] for g, gw in groups.items())
        if marginal <= 0:
            return 0.0
        for key in list(self.hyps):
            owner, nv, mask = self._hyp(key)
            self.hyps[key] *= likelihood[(int((mask * region).sum()), nv)]
        self._set_hyps(self.hyps)
        for L in self.lengths:
            if factor[L].sum() > 0:
                for o in "HV":
                    self.u[L][o] *= factor[L][counts[L][o]]
            self._normalize(L)
        return marginal

    def cell_probs(self):
        """P(cell holds an undamaged segment): exact for hypothesised immobile ships, plus the
        chance that some unhit ship covers it."""
        cov = self.coverage()
        logfree = np.array([np.log(np.clip(1.0 - cov[L], 1e-12, 1.0)) for L in self.lengths])
        groups = {}
        for key, w in self.hyps.items():
            owner, nv, mask = self._hyp(key)
            g = groups.setdefault(nv, [0.0, np.zeros((self.n, self.n))])
            g[0] += w
            g[1] += w * mask
        p = np.zeros((self.n, self.n))
        for nv, (gw, covw) in groups.items():
            free = np.exp(np.tensordot(np.array(nv, dtype=float), logfree, 1))
            p += covw + (gw - covw) * (1.0 - free)
        p[self.hit_mask > 0] = -1.0
        return p


class Agent:
    def __init__(self, obs):
        self.n = obs["board_size"]
        self.fleet = [(s["ship"], s["length"]) for s in obs["fleet"]]
        lengths = [L for _, L in self.fleet]
        self.models = [Belief(self.n, lengths, MOVE_RATE, bias, touch) for bias, touch, _ in PRIOR_MODELS]
        self.model_w = np.array([w for _, _, w in PRIOR_MODELS])
        self.shadows = [Belief(self.n, lengths, rate) for rate in SHADOW_RATES]
        self.edge_weight = 1.0 + EDGE_RISK * (_depth(self.n) == 0)
        self.done_actions = 0
        self.done_damage = 0
        self.seen_turn = None
        self.started = False
        self.turn_cost = 0.02

    def placement(self):
        density = self.shadows[0].cell_probs()
        best, best_score = None, None
        for _ in range(PLACE_TRIES):
            layout = self._random_layout()
            if layout is None:
                continue
            score = sum(density[c] for _, cells in layout for c in cells) + random.random() * 0.15
            if best_score is None or score < best_score:
                best, best_score = layout, score
        return {"placements": [{"ship": ship, "row": cells[0][0], "col": cells[0][1],
                                "orientation": "H" if cells[0][0] == cells[1][0] else "V"}
                               for ship, cells in best]}

    def _random_layout(self):
        halo, layout = set(), []
        for ship, L in sorted(self.fleet, key=lambda s: -s[1]):
            for _ in range(50):
                o = random.choice("HV")
                r = random.randrange(self.n - (L - 1 if o == "V" else 0))
                c = random.randrange(self.n - (L - 1 if o == "H" else 0))
                cells = _cells(r, c, o, L)
                if not halo.intersection(cells):
                    break
            else:
                return None
            halo.update((rr + dr, cc + dc) for rr, cc in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1))
            layout.append((ship, cells))
        return layout

    def _sync(self, obs):
        records = sorted([("shot", s) for s in obs["my_shots"]] + [("sonar", s) for s in obs["my_sonar"]],
                         key=lambda rec: rec[1]["turn"])
        if not self.started and obs["turn"] > 0:
            for m in self.models:
                m.predict()
        self.started = True
        for kind, rec in records[self.done_actions:]:
            lik = np.ones(len(self.models))
            for i, m in enumerate(self.models):
                if kind == "sonar":
                    lik[i] = m.sonar(tuple(rec["center"]), rec["reading"], obs["sonar_correct_probability"])
                elif not rec.get("already_damaged"):
                    cell = tuple(rec["target"])
                    p = min(max(float(m.cell_probs()[cell]), 1e-6), 1 - 1e-6)
                    lik[i] = p if rec["result"] != "MISS" else 1 - p
                    m.shot(cell, rec["result"], rec.get("sunk_length"))
                m.predict()
            w = self.model_w * lik
            self.model_w = w / w.sum() if w.sum() > 0 else self.model_w
            keep = self.model_w >= MODEL_PRUNE
            self.models = [m for m, k in zip(self.models, keep) if k]
            self.model_w = self.model_w[keep] / self.model_w[keep].sum()
        self.done_actions = len(records)

    def _cell_probs(self):
        return sum(w * m.cell_probs() for w, m in zip(self.model_w, self.models))

    def _sync_shadow(self, obs):
        """Replay the rival's turns on shadow beliefs about our fleet: real hits are known, every
        other turn is assumed to be a miss on the shadow's own best cell."""
        my_turn = obs["turn"] + 1
        last = self.seen_turn if self.seen_turn is not None else (my_turn - 2 if obs["turn"] > 0 else my_turn)
        damage = obs["damage_received"]
        lengths = {s["ship"]: s["length"] for s in obs["my_ships"]}
        for opp_turn in range(last + 1, my_turn, 2):
            real = [d for d in damage[self.done_damage:] if d["turn"] == opp_turn]
            if real:
                self.done_damage += 1
            for shadow in self.shadows:
                if real:
                    d = real[0]
                    shadow.shot(tuple(d["cell"]), "SUNK" if d["sunk"] else "HIT", lengths[d["ship"]] if d["sunk"] else None)
                else:
                    p = shadow.cell_probs()
                    guess = np.unravel_index(int(np.argmax(p)), p.shape)
                    shadow.shot((int(guess[0]), int(guess[1])), "MISS")
                shadow.predict()
        self.seen_turn = my_turn

    def _hide(self, obs):
        moves = obs["legal_moves"]
        if not moves:
            return None
        risk = sum(np.clip(s.cell_probs(), 0, None) ** RISK_POWER for s in self.shadows) * self.edge_weight
        ships = {s["ship"]: s for s in obs["my_ships"]}
        best, best_delta = None, -1e-12
        for mv in moves:
            dr, dc = DIRS[mv["direction"]]
            old = [tuple(c) for c in ships[mv["ship"]]["cells"]]
            delta = sum(risk[r + dr, c + dc] for r, c in old) - sum(risk[c] for c in old)
            if delta < best_delta:
                best, best_delta = mv, delta
        return {"ship": best["ship"], "direction": best["direction"]} if best else None

    def _sonar_choice(self, obs, p, best_p):
        """Scan the interior centre whose expected best hit probability after the reading
        (cells treated as independent) beats SONAR_GAIN times the best shot now."""
        if obs["sonar_remaining"] <= 0 or best_p >= SONAR_TRIGGER:
            return None
        n, acc = self.n, obs["sonar_correct_probability"]
        q = np.where(p < 0, 1.0, np.clip(p, 0.0, 1.0))
        centers = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)]
        cells = np.array([[q[r + dr, c + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)] for r, c in centers])
        open_cell = np.array([[p[r + dr, c + dc] >= 0 for dr in (-1, 0, 1) for dc in (-1, 0, 1)] for r, c in centers])
        lik = np.array([[_reading_given_count(k, r, 9, acc) for k in range(10)] for r in range(10)])

        def counts(skip=None):
            d = np.zeros((len(centers), 10))
            d[:, 0] = 1.0
            for j in range(9):
                if j != skip:
                    d = d * (1 - cells[:, j:j + 1]) + np.pad(d[:, :-1], ((0, 0), (1, 0))) * cells[:, j:j + 1]
            return d

        p_read = counts() @ lik.T
        best_in = np.zeros_like(p_read)
        for j in range(9):
            shifted = np.pad(counts(j)[:, :-1], ((0, 0), (1, 0)))
            joint = cells[:, j:j + 1] * (shifted @ lik.T)
            best_in = np.maximum(best_in, np.where(open_cell[:, j:j + 1], joint, 0.0))
        outside = np.array([np.delete(p, [(r + dr) * n + c + dc for dr in (-1, 0, 1) for dc in (-1, 0, 1)]).max()
                            for r, c in centers])
        value = np.maximum(best_in, p_read * outside[:, None]).sum(1)
        i = int(np.argmax(value))
        return centers[i] if value[i] > SONAR_GAIN * best_p else None

    def act(self, obs):
        start = time.perf_counter()
        clock = obs.get("time_remaining_s")
        clock = 5.0 if clock is None else clock
        rounds_left = max(1, obs["max_rounds"] - obs["turn"] // 2)
        spare = clock - TIME_RESERVE
        rich = spare > rounds_left * POOR_TURN + TIME_SAFETY * self.turn_cost
        if not rich and len(self.models) > 1:
            top = int(np.argmax(self.model_w))
            self.models, self.model_w = [self.models[top]], np.ones(1)
        self._sync(obs)
        p = self._cell_probs()
        best_p = float(p.max())
        action = {}
        if rich:
            self._sync_shadow(obs)
            move = self._hide(obs)
            if move:
                action["move"] = move
        center = self._sonar_choice(obs, p, best_p) if rich else None
        if center is not None:
            action["sonar"] = [int(center[0]), int(center[1])]
        else:
            noise = np.random.default_rng(random.getrandbits(32)).random(p.shape) * 1e-9
            r, c = np.unravel_index(int(np.argmax(p + noise)), p.shape)
            action["shot"] = [int(r), int(c)]
        if rich:
            self.turn_cost = 0.7 * self.turn_cost + 0.3 * (time.perf_counter() - start)
        return action


_AGENT = None


def _fallback_shot(observation):
    """Random cell not shot yet, so an internal error costs one weak turn instead of the game."""
    n = observation["board_size"]
    shot = {tuple(s["target"]) for s in observation["my_shots"]}
    open_cells = [(r, c) for r in range(n) for c in range(n) if (r, c) not in shot]
    r, c = random.choice(open_cells or [(0, 0)])
    return {"shot": [r, c]}


def act(observation):
    global _AGENT
    try:
        if observation["phase"] == "placement":
            _AGENT = Agent(observation)
            return _AGENT.placement()
        return _AGENT.act(observation)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        if observation["phase"] == "placement":
            return {"placements": None}
        return _fallback_shot(observation)
