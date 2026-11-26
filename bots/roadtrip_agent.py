"""European Road Trip agent. Owner: Ruben.

Strategy (pure Python, no extra libraries):

1. Dijkstra from every city gives the unconstrained road distance between any two cities.
   This gives a 25-city distance table D, and an exact A* heuristic for every later
   path search.
2. First route: nearest-neighbour order, turned into a real road route and submitted
   at once, so we always have a valid answer within milliseconds.
3. Optimise the visiting order on D with iterated local search (2-opt, Or-opt,
   double-bridge kicks). Every new local optimum is turned into a real route that
   repeats no city. D's tour cost is a lower bound: reaching it means optimal, so we stop.
4. If no order can be realised leg by leg, build a route by cheapest insertion instead.
5. Large-neighbourhood search on real routes until the deadline, with 16 independent
   chains that restart when stuck. Moves: ruin a cluster or a string of required cities
   and reinsert them, reverse a stretch (2-opt), or move a stretch elsewhere (Or-opt).
   Simulated-annealing acceptance; only valid routes are accepted, and every strictly
   shorter one is submitted at once.
"""
import heapq
import math
import random
import time

EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------------


def build_adjacency(graph):
    adjacency = {city: {} for city in graph["cities"]}
    for road in graph["roads"]:
        a, b, km = road["from"], road["to"], road["km"]
        if km < adjacency[a].get(b, math.inf):
            adjacency[a][b] = km
        if road.get("bidirectional", True) and km < adjacency[b].get(a, math.inf):
            adjacency[b][a] = km
    return adjacency


def haversine_km(cities, a, b):
    lat1, lon1 = math.radians(cities[a]["lat"]), math.radians(cities[a]["lon"])
    lat2, lon2 = math.radians(cities[b]["lat"]), math.radians(cities[b]["lon"])
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))


def dijkstra_all(adjacency, source):
    """Distances and predecessors from source to every city."""
    dist = {source: 0}
    heap = [(0, source)]
    done = set()
    while heap:
        d, city = heapq.heappop(heap)
        if city in done:
            continue
        done.add(city)
        for nxt, km in adjacency[city].items():
            nd = d + km
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return dist


HEURISTIC = {}   # goal -> {city: unconstrained road km to goal}; filled by solve()
REQUIRED = set()  # the required cities of the current problem; filled by solve()


def path_avoiding(adjacency, start, goal, blocked):
    """Shortest path start -> goal that does not enter any city in blocked. (path, km) or None.

    A* search. When solve() has filled HEURISTIC, h(n) is the unconstrained shortest road
    distance from n to goal: blocking cities can only make paths longer, so h never
    overestimates (admissible) and it is consistent. It is much tighter than the straight
    line, so very few cities are expanded. Without it, h = 0 and this is Dijkstra.
    """
    h = HEURISTIC.get(goal)
    dist = {start: 0}
    prev = {start: None}
    heap = [(h[start] if h else 0, 0, start)]
    done = set()
    while heap:
        _, d, city = heapq.heappop(heap)
        if city in done:
            continue
        if city == goal:
            path = [goal]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            return path[::-1], d
        done.add(city)
        for nxt, km in adjacency[city].items():
            if nxt in blocked and nxt != goal:
                continue
            nd = d + km
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                prev[nxt] = city
                heapq.heappush(heap, (nd + (h.get(nxt, math.inf) if h else 0), nd, nxt))
    return None


def route_km(adjacency, route):
    return sum(adjacency[a][b] for a, b in zip(route, route[1:]))


# ---------------------------------------------------------------------------------
# Tour optimisation on a distance matrix (indices 0..n-1, tour[0] fixed)
# ---------------------------------------------------------------------------------


def tour_length(tour, D):
    return sum(D[tour[i - 1]][tour[i]] for i in range(len(tour)))


def two_opt(tour, D):
    """Reverse segments while that shortens the closed tour. Changes tour in place."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            dab = D[a][b]
            for j in range(i + 1, n):
                c, d = tour[j], tour[(j + 1) % n]
                delta = D[a][c] + D[b][d] - dab - D[c][d]
                if delta < -1e-9:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
                    a, b = tour[i - 1], tour[i]
                    dab = D[a][b]
    return tour


def or_opt(tour, D):
    """Move segments of 1-3 cities (optionally reversed) elsewhere. Returns True if it helped."""
    n = len(tour)
    any_gain = False
    improved = True
    while improved:
        improved = False
        for seg_len in (1, 2, 3):
            for i in range(1, n - seg_len + 1):
                j = i + seg_len - 1          # segment tour[i..j]
                prev, nxt = tour[i - 1], tour[(j + 1) % n]
                s, e = tour[i], tour[j]
                remove_gain = D[prev][s] + D[e][nxt] - D[prev][nxt]
                rest = tour[:i] + tour[j + 1:]
                segment = tour[i:j + 1]
                best = None
                for k in range(len(rest)):
                    p, q = rest[k], rest[(k + 1) % len(rest)]
                    base = D[p][q]
                    fwd = D[p][s] + D[e][q] - base
                    rev = D[p][e] + D[s][q] - base
                    if fwd < remove_gain - 1e-9 and (best is None or fwd < best[0]):
                        best = (fwd, k, False)
                    if rev < remove_gain - 1e-9 and (best is None or rev < best[0]):
                        best = (rev, k, True)
                if best is not None:
                    _, k, reverse = best
                    seg = segment[::-1] if reverse else segment
                    tour[:] = rest[:k + 1] + seg + rest[k + 1:]
                    # keep the start city first
                    z = tour.index(0) if 0 in tour else 0
                    tour[:] = tour[z:] + tour[:z]
                    improved = any_gain = True
                    break
            if improved:
                break
    return any_gain


def local_opt(tour, D):
    while True:
        two_opt(tour, D)
        if not or_opt(tour, D):
            return tour


def double_bridge(tour, rng):
    n = len(tour)
    if n < 8:
        t = tour[:]
        i, j = sorted(rng.sample(range(1, n), 2))
        t[i:j + 1] = t[i:j + 1][::-1]
        return t
    a, b, c = sorted(rng.sample(range(1, n), 3))
    return tour[:a] + tour[b:c] + tour[a:b] + tour[c:]


def nearest_neighbour(n, D):
    tour, left = [0], set(range(1, n))
    while left:
        last = tour[-1]
        nxt = min(left, key=lambda c: D[last][c])
        tour.append(nxt)
        left.remove(nxt)
    return tour


def iterated_local_search(n, D, deadline, rng, on_local_optimum=None, start_tour=None, max_stall=None):
    current = local_opt(start_tour[:] if start_tour else nearest_neighbour(n, D), D)
    cur_len = tour_length(current, D)
    best, best_len = current[:], cur_len
    if on_local_optimum:
        on_local_optimum(current)
    stall = 0
    while time.monotonic() < deadline and (max_stall is None or stall < max_stall):
        candidate = local_opt(double_bridge(current, rng), D)
        cand_len = tour_length(candidate, D)
        if on_local_optimum:
            on_local_optimum(candidate)
        if cand_len < best_len - 1e-9:
            best, best_len = candidate[:], cand_len
            stall = 0
        else:
            stall += 1
        # accept equal or slightly worse tours now and then to keep moving
        if cand_len <= cur_len + 1e-9 or rng.random() < 0.05:
            current, cur_len = candidate, cand_len
        if stall and stall % 200 == 0:
            current, cur_len = best[:], best_len
    return best, best_len


# ---------------------------------------------------------------------------------
# Turning a visiting order into a legal road route
# ---------------------------------------------------------------------------------


def realise(order, adjacency, required):
    """order: required cities, order[0] = start. Returns a full legal route or None."""
    start = order[0]
    route, used = [start], {start}
    unvisited = set(required) - {start}
    for target in order[1:] + [start]:
        if target != start and target not in unvisited:
            continue                      # already passed through it on an earlier leg
        current = route[-1]
        found = path_avoiding(adjacency, current, target, used | (unvisited - {target}))
        if found is None:
            found = path_avoiding(adjacency, current, target, used)
        if found is None:
            return None
        for city in found[0][1:]:
            route.append(city)
            used.add(city)
            unvisited.discard(city)
    if unvisited or route[-1] != start:
        return None
    return route


def refine(route, adjacency, required):
    """Re-route each leg between required cities with the rest of the route fixed.

    A leg already as short as the unconstrained shortest path cannot improve, so it is
    skipped without a search. Legs are scanned from the end, so replacing one leaves the
    positions of the legs before it unchanged and one pass can fix several.
    """
    required = set(required)
    improved = True
    while improved:
        improved = False
        anchors = [i for i, c in enumerate(route) if c in required]
        anchors[-1] = len(route) - 1      # closing Madrid
        for a, b in reversed(list(zip(anchors, anchors[1:]))):
            old_km = route_km(adjacency, route[a:b + 1])
            h = HEURISTIC.get(route[b])
            if h is not None and old_km <= h.get(route[a], 0) + 1e-9:
                continue
            blocked = set(route[:a]) | set(route[b + 1:])
            found = path_avoiding(adjacency, route[a], route[b], blocked - {route[b]})
            if found is not None and found[1] < old_km - 1e-9:
                route = route[:a] + found[0] + route[b + 1:]
                improved = True
    return route


def is_valid(route, adjacency, required, start):
    if not route or route[0] != start or route[-1] != start:
        return False
    if len(set(route[:-1])) != len(route) - 1:
        return False
    if not set(required) <= set(route):
        return False
    return all(b in adjacency[a] for a, b in zip(route, route[1:]))


# ---------------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------------


def solve(problem, submit):
    t0 = time.monotonic()
    deadline = t0 + problem["time_limit_s"] * 0.85
    start = problem["start"]
    adjacency = build_adjacency(problem)
    required = [start] + [c for c in problem["required"] if c != start]
    n = len(required)

    dist = {c: dijkstra_all(adjacency, c) for c in adjacency}     # roads are two-way
    HEURISTIC.clear()
    HEURISTIC.update(dist)
    REQUIRED.clear()
    REQUIRED.update(required)
    D = [[dist[a].get(b, math.inf) for b in required] for a in required]

    best = {"km": math.inf, "route": None}
    seen = set()
    pool = {}       # distinct realised routes by km, starting points for step 4

    def try_order(tour):
        key = tuple(tour)
        if key in seen:
            return
        seen.add(key)
        seen.add((tour[0],) + tuple(reversed(tour[1:])))
        for idx in (tour, [tour[0]] + tour[1:][::-1]):
            order = [required[i] for i in idx]
            route = realise(order, adjacency, required)
            if route is None:
                continue
            route = refine(route, adjacency, required)
            km = route_km(adjacency, route)
            if is_valid(route, adjacency, required, start):
                pool.setdefault(km, route)
                if km < best["km"]:
                    best["km"], best["route"] = km, route
                    submit(route)

    # 1) quick first answer
    try_order(nearest_neighbour(n, D))

    # 2) iterated local search on the distance table, realising every new local optimum.
    #    The table cost is a lower bound: if the best route already reaches it, we are done.
    rng = random.Random(12345)
    phase_a_end = min(deadline, t0 + problem["time_limit_s"] * 0.08)
    lower_bound = math.inf
    while time.monotonic() < phase_a_end:
        # without any route yet, give this phase only 0.3 s before the sturdier builder
        end = phase_a_end if best["route"] is not None else min(phase_a_end, t0 + 0.3)
        if time.monotonic() >= end:
            break
        first = list(range(1, n))
        rng.shuffle(first)
        _, lb = iterated_local_search(n, D, end, rng, on_local_optimum=try_order,
                                      start_tour=[0] + first, max_stall=300)
        lower_bound = min(lower_bound, lb)
    if best["km"] <= lower_bound + 1e-9:
        return

    # 3) if no order could be realised leg by leg, build a route by cheapest insertion,
    #    which adapts to the cities already used and almost never gets stuck.
    attempts = 0
    while best["route"] is None and time.monotonic() < deadline:
        others = required[1:]
        if attempts == 0:   # farthest cities first: they shape the route
            others = sorted(others, key=lambda c: -dist[start].get(c, 0))
        else:
            others = rng.sample(others, len(others))
        route = [start, start]
        for city in others:
            if city not in route:
                route = best_insert(route, city, adjacency, dist)
                if route is None:
                    break
        if route is not None:
            route = refine(route, adjacency, required)
            if is_valid(route, adjacency, required, start):
                best["km"], best["route"] = route_km(adjacency, route), route
                pool.setdefault(best["km"], route)
                submit(route)
        attempts += 1
    if best["route"] is None:
        return

    # 4) ruin and recreate on the REAL route: remove a few nearby required cities and put
    #    them back by cheapest insertion. Every candidate respects the no-repeat rule.
    ruin_and_recreate(required, adjacency, dist, best, submit, deadline, rng, lower_bound,
                      [pool[k] for k in sorted(pool)])


def best_insert(route, city, adjacency, dist):
    """Insert city into the closed route at the cheapest place, with paths that repeat no city."""
    used = set(route)
    candidates = []
    for k in range(len(route) - 1):
        u, v = route[k], route[k + 1]
        old = adjacency[u][v] if u != v else 0
        lb = dist[u].get(city, math.inf) + dist[city].get(v, math.inf) - old
        candidates.append((lb, k))
    candidates.sort()
    best_delta, best_route = math.inf, None
    for lb, k in candidates:
        if lb >= best_delta - 1e-9:
            break
        u, v = route[k], route[k + 1]
        old = adjacency[u][v] if u != v else 0
        for first_leg in (0, 1):     # try building either half first
            if first_leg == 0:
                there = path_avoiding(adjacency, u, city, used)
                if there is None:
                    continue
                back = path_avoiding(adjacency, city, v, (used | set(there[0])) - {v})
                if back is None:
                    continue
            else:
                back = path_avoiding(adjacency, city, v, used - {v})
                if back is None:
                    continue
                there = path_avoiding(adjacency, u, city, used | set(back[0]))
                if there is None:
                    continue
            delta = there[1] + back[1] - old
            if delta < best_delta - 1e-9:
                best_delta = delta
                best_route = route[:k] + there[0] + back[0][1:] + route[k + 2:]
    return best_route


def remove_city(route, city, required_set, adjacency):
    """Drop city (and the detour to it) by joining the required cities either side of it."""
    i = route.index(city)
    a = i - 1
    while a > 0 and route[a] not in required_set:
        a -= 1
    b = i + 1
    while b < len(route) - 1 and route[b] not in required_set:
        b += 1
    blocked = set(route[:a]) | set(route[b + 1:])
    if route[a] == route[b]:          # only Madrid would be left
        return [route[a], route[a]]
    found = path_avoiding(adjacency, route[a], route[b], blocked - {route[b]})
    if found is None:
        return None
    return route[:a] + found[0] + route[b + 1:]


def reverse_stretch(route, i, j, anchors, adjacency):
    """2-opt on the real route: reverse the stretch from anchor i to anchor j and reconnect
    both ends with new paths that repeat no city. Returns the new route or None."""
    a_prev, a_i, a_j, a_next = anchors[i - 1], anchors[i], anchors[j], anchors[j + 1]
    head, stretch, tail = route[:a_prev + 1], route[a_i:a_j + 1], route[a_next:]
    fixed = set(head) | set(stretch) | set(tail)
    u, v = route[a_prev], route[a_next]
    first, last = stretch[0], stretch[-1]
    for order in (0, 1):
        if order == 0:
            leg1 = path_avoiding(adjacency, u, last, fixed - {last})
            if leg1 is None:
                continue
            leg2 = path_avoiding(adjacency, first, v, (fixed | set(leg1[0])) - {v})
        else:
            leg2 = path_avoiding(adjacency, first, v, fixed - {v})
            if leg2 is None:
                continue
            leg1 = path_avoiding(adjacency, u, last, (fixed | set(leg2[0])) - {last})
        if leg1 is not None and leg2 is not None:
            return head[:-1] + leg1[0] + stretch[::-1][1:-1] + leg2[0] + tail[1:]
    return None


def move_stretch(route, i, j, anchors, adjacency, rng):
    """Or-opt on the real route: cut the stretch from anchor i to anchor j, close the gap,
    and put the stretch back whole (maybe reversed) between two other required cities."""
    p, n = anchors[i - 1], anchors[j + 1]
    stretch = route[anchors[i]:anchors[j] + 1]
    outside = set(route[:p + 1]) | set(route[n:]) | set(stretch)
    gap = path_avoiding(adjacency, route[p], route[n], outside - {route[n]})
    if gap is None:
        return None
    rest = route[:p + 1] + gap[0][1:-1] + route[n:]
    required_positions = [k for k, c in enumerate(rest) if c in REQUIRED]
    required_positions[-1] = len(rest) - 1
    t = rng.randrange(len(required_positions) - 1)
    u, v = required_positions[t], required_positions[t + 1]
    head, tail = rest[:u + 1], rest[v:]
    seg = stretch[::-1] if rng.random() < 0.5 else stretch
    fixed = set(head) | set(tail) | set(seg)
    leg1 = path_avoiding(adjacency, rest[u], seg[0], fixed - {seg[0]})
    if leg1 is None:
        return None
    leg2 = path_avoiding(adjacency, seg[-1], rest[v], (fixed | set(leg1[0])) - {rest[v]})
    if leg2 is None:
        return None
    return head[:-1] + leg1[0][:-1] + seg + leg2[0][1:] + tail[1:]


STALL_LIMIT = 150   # moves without a new best before a chain restarts


def random_insertion_route(required, adjacency, dist, rng):
    """A fresh valid route: cheapest insertion of the required cities in a random order."""
    route = [required[0], required[0]]
    for city in rng.sample(required[1:], len(required) - 1):
        if city not in route:
            route = best_insert(route, city, adjacency, dist)
            if route is None:
                return None
    return refine(route, adjacency, required)


def ruin_and_recreate(required, adjacency, dist, best, submit, deadline, rng, lower_bound, pool):
    """Large-neighbourhood search on real routes, with several independent chains.

    Moves: (a) ruin a cluster of nearby required cities, (b) ruin a string of consecutive
    required cities, then put them back by cheapest insertion; (c) reverse a stretch of the
    route (2-opt on real roads). Simulated-annealing acceptance with a falling temperature.
    """
    start = required[0]
    required_set = set(required)
    others = required[1:]
    # each chain: [current route, its km, the chain's best km, moves since that best]
    chains = [[r, route_km(adjacency, r), math.inf, 0] for r in pool[:16]] or [[best["route"], best["km"], math.inf, 0]]
    temperature = 0.01 * best["km"]
    t_begin, t_total = time.monotonic(), max(1e-6, deadline - time.monotonic())
    step = 0
    while time.monotonic() < deadline and best["km"] > lower_bound + 1e-9:
        chain = chains[step % len(chains)]
        step += 1
        current, cur_km = chain[0], chain[1]
        if chain[3] > STALL_LIMIT:    # stuck: restart this chain from a fresh random route
            fresh = random_insertion_route(required, adjacency, dist, rng)
            if fresh is not None:
                chain[:] = [fresh, route_km(adjacency, fresh), math.inf, 0]
                current, cur_km = chain[0], chain[1]
            else:
                chain[3] = 0
        chain[3] += 1
        progress = (time.monotonic() - t_begin) / t_total
        temp = temperature * (1 - progress)
        move = rng.random()
        route = current
        if move < 0.4:
            anchors = [i for i, c in enumerate(route) if c in required_set]
            anchors[-1] = len(route) - 1
            if len(anchors) < 4:
                continue
            i = rng.randint(1, len(anchors) - 3)
            if move < 0.2:      # 2-opt: reverse a stretch
                j = rng.randint(i + 1, len(anchors) - 2)
                route = reverse_stretch(route, i, j, anchors, adjacency)
            else:               # or-opt: move a stretch of 1-10 required cities elsewhere
                j = rng.randint(i, min(len(anchors) - 2, i + 9))
                route = move_stretch(route, i, j, anchors, adjacency, rng)
            if route is None:
                continue
        else:
            if move < 0.7:      # cluster of nearby required cities
                seed_city = rng.choice(others)
                k = rng.randint(1, min(8, len(others)))
                victims = sorted(others, key=lambda c: dist[seed_city].get(c, math.inf))[:k]
            else:               # string of consecutive required cities along the route
                order = [c for c in route[1:-1] if c in required_set]
                k = rng.randint(2, min(8, len(order)))
                s = rng.randint(0, len(order) - k)
                victims = order[s:s + k]
            removed = []
            for city in victims:
                if city in route:
                    shorter = remove_city(route, city, required_set, adjacency)
                    if shorter is not None:
                        route = shorter
                        removed.append(city)
            missing = [c for c in removed if c not in route]
            rng.shuffle(missing)
            if rng.random() < 0.5:
                missing.sort(key=lambda c: -dist[start].get(c, 0))
            for city in missing:
                if city in route:       # an earlier insertion already passes through it
                    continue
                route = best_insert(route, city, adjacency, dist)
                if route is None:
                    break
            if route is None:
                continue
        route = refine(route, adjacency, required)
        if not is_valid(route, adjacency, required, start):
            continue
        km = route_km(adjacency, route)
        if km < cur_km - 1e-9 or (temp > 0 and rng.random() < math.exp(-(km - cur_km) / temp)):
            chain[0], chain[1] = route, km
        if km < chain[2] - 1e-9:
            chain[2], chain[3] = km, 0
        if km < best["km"] - 1e-9:
            best["km"], best["route"] = km, route
            submit(route)
        if rng.random() < 0.005:      # rarely pull a chain back to the best route
            chain[0], chain[1] = best["route"], best["km"]


# ---------------------------------------------------------------------------------
# Required algorithms. check_requirements() tests these.
# ---------------------------------------------------------------------------------


def ucs(graph, start, goal):
    """Uniform-cost search. A node counts as expanded when it is popped and explored."""
    adjacency = build_adjacency(graph)
    frontier = [(0, start)]
    best_g = {start: 0}
    parent = {start: None}
    closed = set()
    expanded = 0
    while frontier:
        g, city = heapq.heappop(frontier)
        if city in closed:
            continue
        if city == goal:
            path = [goal]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return {"path": path[::-1], "km": g, "expanded": expanded}
        closed.add(city)
        expanded += 1
        for nxt, km in adjacency[city].items():
            ng = g + km
            if nxt not in closed and ng < best_g.get(nxt, math.inf):
                best_g[nxt] = ng
                parent[nxt] = city
                heapq.heappush(frontier, (ng, nxt))
    return {"path": None, "km": math.inf, "expanded": expanded}


def astar(graph, start, goal):
    """A* with h(n) = Haversine distance to goal (admissible and consistent here)."""
    adjacency = build_adjacency(graph)
    cities = graph["cities"]
    h = {c: haversine_km(cities, c, goal) for c in cities}
    frontier = [(h[start], 0, start)]
    best_g = {start: 0}
    parent = {start: None}
    closed = set()
    expanded = 0
    while frontier:
        f, g, city = heapq.heappop(frontier)
        if city in closed:
            continue
        if city == goal:
            path = [goal]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return {"path": path[::-1], "km": g, "expanded": expanded}
        closed.add(city)
        expanded += 1
        for nxt, km in adjacency[city].items():
            ng = g + km
            if nxt not in closed and ng < best_g.get(nxt, math.inf):
                best_g[nxt] = ng
                parent[nxt] = city
                heapq.heappush(frontier, (ng + h[nxt], ng, nxt))
    return {"path": None, "km": math.inf, "expanded": expanded}


def local_search(tour, distance, time_limit_s):
    """Iterated local search: 2-opt + Or-opt to a local optimum, double-bridge kicks to escape.

    tour: city ids, first city fixed, not repeated at the end. distance: {a: {b: km}}.
    """
    if len(tour) < 4:
        return list(tour)
    deadline = time.monotonic() + time_limit_s * 0.8
    D = [[0 if a == b else distance[a][b] for b in tour] for a in tour]
    best, _ = iterated_local_search(len(tour), D, deadline, random.Random(0),
                                    start_tour=list(range(len(tour))), max_stall=300)
    return [tour[i] for i in best]
