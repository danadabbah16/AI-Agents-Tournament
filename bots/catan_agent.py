"""Catan agent. Salome + Dana

Started from: student_templates/catan_agent.py
Rules: games/catan/README.md
Samples: docs/students/sample_observations/catan_*.json
This file is the one we hand in as: catan_agent.py
"""

PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

PRIORITY = [
    "BUILD_CITY",
    "BUILD_SETTLEMENT",
    "PLAY_KNIGHT_CARD",
    "BUY_DEVELOPMENT_CARD",
    "PLAY_YEAR_OF_PLENTY",
    "PLAY_MONOPOLY",
    "PLAY_ROAD_BUILDING",
    "BUILD_ROAD",
    "MARITIME_TRADE",
    "ROLL",
    "MOVE_ROBBER",
    "END_TURN",
]

COSTS = {
    "BUILD_CITY": {"WHEAT": 2, "ORE": 3},
    "BUILD_SETTLEMENT": {"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1},
    "BUILD_ROAD": {"WOOD": 1, "BRICK": 1},
    "BUY_DEVELOPMENT_CARD": {"SHEEP": 1, "WHEAT": 1, "ORE": 1},
}


def node_pips(board):
    scores = {}
    for tile in board["tiles"]:
        pip = PIPS.get(tile.get("number"), 0)
        for node in tile["nodes"]:
            scores[node] = scores.get(node, 0) + pip
    return scores


def first_of_type(actions, kind):
    for a in actions:
        if a["type"] == kind:
            return a
    return None


def best_settlement(actions, board):
    options = [a for a in actions if a["type"] == "BUILD_SETTLEMENT"]
    if not options:
        return None
    pips = node_pips(board)
    return max(options, key=lambda a: pips.get(a["node"], 0))


def best_city(actions, board):
    options = [a for a in actions if a["type"] == "BUILD_CITY"]
    if not options:
        return None
    pips = node_pips(board)
    return max(options, key=lambda a: pips.get(a["node"], 0))


def best_robber(actions):
    steal = [a for a in actions if a.get("steal_from") == "opponent"]
    if steal:
        return steal[0]
    return actions[0]


def afford(resources, cost):
    return all(resources.get(r, 0) >= n for r, n in cost.items())


def after_trade(resources, trade):
    have = dict(resources)
    if have.get(trade["give"], 0) < trade["give_count"]:
        return None
    have[trade["give"]] -= trade["give_count"]
    have[trade["receive"]] = have.get(trade["receive"], 0) + 1
    return have


def useful_trade(actions, resources):
    trades = [a for a in actions if a["type"] == "MARITIME_TRADE"]
    for goal in ("BUILD_CITY", "BUILD_SETTLEMENT"):
        if afford(resources, COSTS[goal]):
            continue
        for trade in trades:
            have = after_trade(resources, trade)
            if have is not None and afford(have, COSTS[goal]):
                return trade
    return None


def act(observation):
    try:
        return {"action": smart_policy(observation)}
    except Exception:
        try:
            return {"action": heuristic_policy(observation)}
        except Exception:
            return {"action": observation["actions"][0]["id"]}


def learned_policy(observation):
    """v2: choose the action TYPE with the highest learned Q, then the best action of that type.

    Falls back to the baseline when there is nothing to choose between, or when
    the table does not know every option in this state.
    """
    kinds = sorted({a["type"] for a in observation["actions"]})
    if len(kinds) < 2 or observation.get("phase") == "initial_placement":
        return heuristic_policy(observation)
    if any(q_key(observation, kind) not in Q for kind in kinds):
        return heuristic_policy(observation)
    best = max(kinds, key=lambda kind: Q[q_key(observation, kind)])
    return pick_of_type(observation, best)["id"]


def heuristic_policy(observation):
    """Baseline: build when we can, steal with robber, trade only to unlock a build."""
    actions = observation["actions"]
    phase = observation.get("phase")
    board = observation["board"]
    me = observation["me"]
    resources = me.get("resources", {})

    if phase == "initial_placement":
        choice = best_settlement(actions, board)
        return (choice or actions[0])["id"]

    if phase == "move_robber":
        return best_robber(actions)["id"]

    if phase == "free_roads":
        road = first_of_type(actions, "BUILD_ROAD")
        return (road or actions[0])["id"]

    for picker in (best_city, best_settlement):
        choice = picker(actions, board)
        if choice is not None:
            return choice["id"]

    trade = useful_trade(actions, resources)
    if trade is not None:
        return trade["id"]

    missing_for_city = sum(
        max(0, n - resources.get(r, 0)) for r, n in COSTS["BUILD_CITY"].items()
    )
    save_for_city = bool(me.get("settlements")) and missing_for_city <= 2

    for kind in PRIORITY:
        if kind in ("BUILD_CITY", "BUILD_SETTLEMENT", "MARITIME_TRADE"):
            continue
        if save_for_city and kind in ("BUY_DEVELOPMENT_CARD", "BUILD_ROAD"):
            continue
        choice = first_of_type(actions, kind)
        if choice is not None:
            return choice["id"]

    return actions[0]["id"]


def state_features(observation):
    """Small abstraction of the state, so a table can cover it. Must be deterministic."""
    me = observation["me"]
    opp = observation["opponent"]
    my_vp = me.get("victory_points", me.get("visible_victory_points", 0))
    lead = my_vp - opp.get("visible_victory_points", 0)
    lead = max(-2, min(2, lead))
    cards = me.get("resource_cards", 0)
    hand = 0 if cards <= 3 else (1 if cards <= 7 else 2)  # 8+ risks the discard on a 7
    return [observation.get("phase"), my_vp, lead, hand]


def action_value(observation, action_id):
    """Learned Q value of this action's type in this state (0.0 if never learned)."""
    action = next((a for a in observation["actions"] if a["id"] == action_id), None)
    if action is None:
        return 0.0
    return float(Q.get(q_key(observation, action["type"]), 0.0))


def q_update(q, reward, best_next_q, alpha, gamma):
    """Q-learning update rule."""
    return q + alpha * (reward + gamma * best_next_q - q)



def q_key(observation, kind):
    """Table key: the abstract state plus the TYPE of action (ids change every turn)."""
    return str(tuple(state_features(observation))) + "|" + kind


def pick_of_type(observation, kind):
    """The concrete action of this type that the baseline's helpers prefer."""
    actions = observation["actions"]
    board = observation["board"]
    if kind == "BUILD_SETTLEMENT":
        return best_settlement(actions, board)
    if kind == "BUILD_CITY":
        return best_city(actions, board)
    if kind == "MOVE_ROBBER":
        return best_robber([a for a in actions if a["type"] == kind])
    if kind == "MARITIME_TRADE":
        trade = useful_trade(actions, observation["me"].get("resources", {}))
        if trade is not None:
            return trade
    return first_of_type(actions, kind)

# ---------------------------------------------------------------------------
# v3: goal-directed strategy. Uses only the observation and the standard library.
# Idea: pick ONE goal (city, settlement, road or development card), then make
# every decision this turn serve that goal: what to trade, what to build, what to keep.
# ---------------------------------------------------------------------------

GOAL_COST = {
    "CITY": {"WHEAT": 2, "ORE": 3},
    "SETTLEMENT": {"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1},
    "ROAD": {"WOOD": 1, "BRICK": 1},
    "DEV": {"SHEEP": 1, "WHEAT": 1, "ORE": 1},
}

# Weights of the strategy. Hand-set starting values.
W = {
    "res_WOOD": 1.0, "res_BRICK": 1.0, "res_SHEEP": 0.8, "res_WHEAT": 1.1, "res_ORE": 1.0,
    "diversity": 1.5,      # bonus per different resource a corner touches
    "new_resource": 1.5,   # bonus for a resource we do not produce yet
    "port": 0.5,           # bonus for a corner with a port
    "dist_penalty": 2.5,   # cost of each road still needed to reach a corner
    "hand_limit": 7,       # above this many cards a 7 makes us discard
    "dev_when_idle": 1.0,  # how reluctant we are to pick a development card as the goal
    "road_cap": 15,        # roads available in the game
}
_BOARD = {}


def board_info(board):
    info = _BOARD.get("info")
    if info is not None and _BOARD.get("sig") == _sig(board):
        return info
    prod, adj = {}, {}
    for tile in board["tiles"]:
        for node in tile["nodes"]:
            prod.setdefault(node, [])
            if tile.get("resource") and tile.get("number"):
                prod[node].append((tile["resource"], PIPS.get(tile["number"], 0), tuple(tile["tile"])))
        for a, b in tile["edges"]:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    ports = {}
    for port in board.get("ports", []):
        for node in port["nodes"]:
            ports[node] = port.get("resource") or "ANY"
    info = {"prod": prod, "adj": adj, "ports": ports}
    _BOARD["info"], _BOARD["sig"] = info, _sig(board)
    return info


def _sig(board):
    return tuple((t["resource"], t["number"]) for t in board["tiles"])


def node_value(node, info, have=None):
    total, kinds = 0.0, set()
    for res, pip, tile in info["prod"].get(node, []):
        total += pip * W["res_" + res]
        kinds.add(res)
        if have is not None and res not in have:
            total += W["new_resource"] * pip / 3.0
    total += W["diversity"] * len(kinds)
    if node in info["ports"]:
        total += W["port"]
    return total


def my_resources_produced(me, info):
    have = set()
    for node in list(me.get("settlements", [])) + list(me.get("cities", [])):
        for res, pip, tile in info["prod"].get(node, []):
            have.add(res)
    return have


def occupied(obs):
    me, opp = obs["me"], obs["opponent"]
    return set(me["settlements"]) | set(me["cities"]) | set(opp["settlements"]) | set(opp["cities"])


def open_spots(obs, info):
    """Nodes where a settlement could ever be built (free, and no neighbour built)."""
    occ = occupied(obs)
    return [n for n in info["adj"] if n not in occ and not (info["adj"][n] & occ)]


def my_network(me):
    nodes = set(me["settlements"]) | set(me["cities"])
    for a, b in me["roads"]:
        nodes.add(a)
        nodes.add(b)
    return nodes


def missing(resources, cost):
    return sum(max(0, n - resources.get(r, 0)) for r, n in cost.items())


def bfs(start_nodes, info, blocked_nodes, used_edges):
    dist = {n: 0 for n in start_nodes}
    frontier = list(start_nodes)
    while frontier:
        nxt = []
        for u in frontier:
            if u in blocked_nodes and dist[u] > 0:
                continue
            for v in info["adj"][u]:
                e = (min(u, v), max(u, v))
                if e in used_edges or v in dist:
                    continue
                dist[v] = dist[u] + 1
                nxt.append(v)
        frontier = nxt
    return dist


def road_plan(obs, info):
    """Best (score, target, dist) settlement spot reachable from my network."""
    me, opp = obs["me"], obs["opponent"]
    opp_nodes = set(opp["settlements"]) | set(opp["cities"])
    used = {(min(a, b), max(a, b)) for a, b in me["roads"] + opp["roads"]}
    have = my_resources_produced(me, info)
    spots = open_spots(obs, info)
    net = my_network(me)
    dist = bfs(net, info, opp_nodes, used)
    best = None
    for t in spots:
        if t in dist:
            score = node_value(t, info, have) - W["dist_penalty"] * dist[t]
            if best is None or score > best[0]:
                best = (score, t, dist[t])
    return best, spots, have, opp_nodes, used


def best_road(obs, info, roads):
    me = obs["me"]
    net = my_network(me)
    plan, spots, have, opp_nodes, used = road_plan(obs, info)
    best, best_action = None, None
    for action in roads:
        a, b = action["edge"]
        e = (min(a, b), max(a, b))
        ends = [n for n in (a, b) if n not in net] or [a, b]
        d = bfs(ends, info, opp_nodes, used | {e})
        score = -1e9
        for t in spots:
            if t in d:
                score = max(score, node_value(t, info, have) - W["dist_penalty"] * d[t])
        if best is None or score > best:
            best, best_action = score, action
    return best_action


def robber_choice(obs, info, actions):
    me, opp = obs["me"], obs["opponent"]
    tile_nodes = {tuple(t["tile"]): (t["nodes"], PIPS.get(t.get("number"), 0)) for t in obs["board"]["tiles"]}
    mine_s, mine_c = set(me["settlements"]), set(me["cities"])
    opp_s, opp_c = set(opp["settlements"]), set(opp["cities"])
    best, best_action = None, None
    for action in actions:
        nodes, pip = tile_nodes.get(tuple(action["tile"]), ([], 0))
        hurt = sum(1 for n in nodes if n in opp_s) + 2 * sum(1 for n in nodes if n in opp_c)
        self_hurt = sum(1 for n in nodes if n in mine_s) + 2 * sum(1 for n in nodes if n in mine_c)
        score = pip * (hurt - 2.0 * self_hurt)
        if action.get("steal_from") == "opponent" and opp.get("resource_cards", 0) > 0:
            score += 3.0
        if best is None or score > best:
            best, best_action = score, action
    return best_action


def choose_goal(obs, info):
    """What we are saving for: CITY, SETTLEMENT, ROAD or DEV."""
    me = obs["me"]
    res = me["resources"]
    plan = road_plan(obs, info)[0]
    can_city = len(me["settlements"]) > 0 and len(me["cities"]) < 4
    spot_ready = plan is not None and plan[2] == 0 and len(me["settlements"]) < 5
    options = []
    if can_city:
        options.append((missing(res, GOAL_COST["CITY"]), 0, "CITY"))
    if spot_ready:
        options.append((missing(res, GOAL_COST["SETTLEMENT"]), 1, "SETTLEMENT"))
    elif plan is not None and len(me["settlements"]) < 5 and len(me["roads"]) < W["road_cap"]:
        options.append((missing(res, GOAL_COST["ROAD"]) + 1.5 * plan[2], 2, "ROAD"))
    options.append((missing(res, GOAL_COST["DEV"]) + W["dev_when_idle"], 3, "DEV"))
    options.sort()
    return options[0][2], plan


def smart_policy(obs):
    actions = obs["actions"]
    phase = obs.get("phase")
    info = board_info(obs["board"])
    me, opp = obs["me"], obs["opponent"]
    res = me.get("resources", {})
    by = {}
    for a in actions:
        by.setdefault(a["type"], []).append(a)

    if phase == "initial_placement":
        if "BUILD_SETTLEMENT" in by:
            have = my_resources_produced(me, info)
            return max(by["BUILD_SETTLEMENT"], key=lambda a: node_value(a["node"], info, have))["id"]
        if "BUILD_ROAD" in by:
            return best_road(obs, info, by["BUILD_ROAD"])["id"]
        return actions[0]["id"]

    if phase == "move_robber" or (len(by) == 1 and "MOVE_ROBBER" in by):
        return robber_choice(obs, info, by.get("MOVE_ROBBER", actions))["id"]

    if phase == "free_roads":
        if "BUILD_ROAD" in by:
            return best_road(obs, info, by["BUILD_ROAD"])["id"]
        return actions[0]["id"]

    have = my_resources_produced(me, info)
    if "BUILD_CITY" in by:
        return max(by["BUILD_CITY"], key=lambda a: sum(p for _, p, _ in info["prod"].get(a["node"], [])))["id"]
    if "BUILD_SETTLEMENT" in by:
        return max(by["BUILD_SETTLEMENT"], key=lambda a: node_value(a["node"], info, have))["id"]
    if "PLAY_KNIGHT_CARD" in by:
        return by["PLAY_KNIGHT_CARD"][0]["id"]
    if "ROLL" in by:
        return by["ROLL"][0]["id"]

    goal, plan = choose_goal(obs, info)
    cost = GOAL_COST[goal]

    if "PLAY_YEAR_OF_PLENTY" in by:
        def yop_score(a):
            after = dict(res)
            for r in a["resources"]:
                after[r] = after.get(r, 0) + 1
            return -missing(after, cost)
        return max(by["PLAY_YEAR_OF_PLENTY"], key=yop_score)["id"]
    if "PLAY_MONOPOLY" in by:
        need = max(cost, key=lambda r: cost[r] - res.get(r, 0))
        for a in by["PLAY_MONOPOLY"]:
            if a["resource"] == need:
                return a["id"]
    if "PLAY_ROAD_BUILDING" in by and plan is not None and plan[2] > 0:
        return by["PLAY_ROAD_BUILDING"][0]["id"]

    if goal == "ROAD" and "BUILD_ROAD" in by:
        return best_road(obs, info, by["BUILD_ROAD"])["id"]
    if goal == "DEV" and "BUY_DEVELOPMENT_CARD" in by:
        return by["BUY_DEVELOPMENT_CARD"][0]["id"]

    # Trade surplus toward the goal.
    if "MARITIME_TRADE" in by:
        best, best_action = None, None
        before = missing(res, cost)
        for a in by["MARITIME_TRADE"]:
            give, n, get = a["give"], a["give_count"], a["receive"]
            if res.get(give, 0) - n < cost.get(give, 0):
                continue                      # do not trade away what the goal needs
            if cost.get(get, 0) - res.get(get, 0) <= 0:
                continue                      # only receive what the goal is missing
            after = dict(res)
            after[give] -= n
            after[get] = after.get(get, 0) + 1
            gain = before - missing(after, cost)
            score = (gain, -n)
            if gain > 0 and (best is None or score > best):
                best, best_action = score, a
        if best_action is not None:
            return best_action["id"]

    # Spend instead of risking the discard on a 7.
    if me.get("resource_cards", 0) > W["hand_limit"]:
        if "BUY_DEVELOPMENT_CARD" in by:
            return by["BUY_DEVELOPMENT_CARD"][0]["id"]
        if "BUILD_ROAD" in by and plan is not None and plan[2] > 0 and goal != "SETTLEMENT":
            return best_road(obs, info, by["BUILD_ROAD"])["id"]

    if "END_TURN" in by:
        return by["END_TURN"][0]["id"]
    return actions[0]["id"]

# Learned offline by train_catan.py. Empty until Part 5.
Q = {"('play_turn', 2, -1, 0)|BUILD_ROAD": -0.0633, "('play_turn', 2, -1, 0)|BUY_DEVELOPMENT_CARD": 1.2753, "('play_turn', 2, -1, 0)|END_TURN": 0.5477, "('play_turn', 2, -1, 0)|PLAY_KNIGHT_CARD": 1.1755, "('play_turn', 2, -1, 0)|PLAY_MONOPOLY": 0.2349, "('play_turn', 2, -1, 0)|PLAY_YEAR_OF_PLENTY": 0.4072, "('play_turn', 2, -1, 0)|ROLL": 0.8693, "('play_turn', 2, -1, 1)|BUILD_CITY": 2.5348, "('play_turn', 2, -1, 1)|BUILD_ROAD": 1.5785, "('play_turn', 2, -1, 1)|BUILD_SETTLEMENT": 2.1469, "('play_turn', 2, -1, 1)|BUY_DEVELOPMENT_CARD": 1.35, "('play_turn', 2, -1, 1)|END_TURN": 0.8886, "('play_turn', 2, -1, 1)|MARITIME_TRADE": -0.5123, "('play_turn', 2, -1, 1)|PLAY_KNIGHT_CARD": 1.39, "('play_turn', 2, -1, 1)|PLAY_MONOPOLY": -0.006, "('play_turn', 2, -1, 1)|PLAY_ROAD_BUILDING": 1.2366, "('play_turn', 2, -1, 1)|PLAY_YEAR_OF_PLENTY": 0.6755, "('play_turn', 2, -1, 1)|ROLL": 0.7936, "('play_turn', 2, -1, 2)|BUILD_CITY": 2.3328, "('play_turn', 2, -1, 2)|BUILD_ROAD": 0.1116, "('play_turn', 2, -1, 2)|BUILD_SETTLEMENT": 2.5356, "('play_turn', 2, -1, 2)|BUY_DEVELOPMENT_CARD": 1.1969, "('play_turn', 2, -1, 2)|END_TURN": 0.3735, "('play_turn', 2, -1, 2)|MARITIME_TRADE": 1.0705, "('play_turn', 2, -1, 2)|PLAY_KNIGHT_CARD": 1.1884, "('play_turn', 2, -2, 0)|BUILD_ROAD": -1.7936, "('play_turn', 2, -2, 0)|BUY_DEVELOPMENT_CARD": -0.8042, "('play_turn', 2, -2, 0)|END_TURN": -0.7518, "('play_turn', 2, -2, 0)|PLAY_KNIGHT_CARD": -1.4012, "('play_turn', 2, -2, 0)|PLAY_MONOPOLY": -1.2953, "('play_turn', 2, -2, 0)|PLAY_YEAR_OF_PLENTY": -1.06, "('play_turn', 2, -2, 0)|ROLL": -0.7488, "('play_turn', 2, -2, 1)|BUILD_CITY": -0.6573, "('play_turn', 2, -2, 1)|BUILD_ROAD": -0.982, "('play_turn', 2, -2, 1)|BUILD_SETTLEMENT": -0.0553, "('play_turn', 2, -2, 1)|BUY_DEVELOPMENT_CARD": -0.875, "('play_turn', 2, -2, 1)|END_TURN": -2.0991, "('play_turn', 2, -2, 1)|MARITIME_TRADE": -1.6325, "('play_turn', 2, -2, 1)|PLAY_KNIGHT_CARD": -1.0828, "('play_turn', 2, -2, 1)|PLAY_MONOPOLY": -1.1692, "('play_turn', 2, -2, 1)|PLAY_ROAD_BUILDING": -0.417, "('play_turn', 2, -2, 1)|PLAY_YEAR_OF_PLENTY": -1.3339, "('play_turn', 2, -2, 1)|ROLL": -0.7318, "('play_turn', 2, -2, 2)|BUILD_CITY": 0.4799, "('play_turn', 2, -2, 2)|BUILD_ROAD": -1.0198, "('play_turn', 2, -2, 2)|BUILD_SETTLEMENT": 0.7395, "('play_turn', 2, -2, 2)|BUY_DEVELOPMENT_CARD": -0.4808, "('play_turn', 2, -2, 2)|END_TURN": -1.8314, "('play_turn', 2, -2, 2)|MARITIME_TRADE": -1.5361, "('play_turn', 2, -2, 2)|PLAY_KNIGHT_CARD": 0.1175, "('play_turn', 2, -2, 2)|PLAY_MONOPOLY": -0.8204, "('play_turn', 2, -2, 2)|PLAY_ROAD_BUILDING": 0.0191, "('play_turn', 2, -2, 2)|PLAY_YEAR_OF_PLENTY": -0.6832, "('play_turn', 2, 0, 0)|BUILD_ROAD": 1.6527, "('play_turn', 2, 0, 0)|BUY_DEVELOPMENT_CARD": 2.0925, "('play_turn', 2, 0, 0)|END_TURN": 1.1042, "('play_turn', 2, 0, 0)|PLAY_KNIGHT_CARD": 2.1598, "('play_turn', 2, 0, 0)|PLAY_MONOPOLY": 1.4955, "('play_turn', 2, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.553, "('play_turn', 2, 0, 0)|ROLL": 1.6181, "('play_turn', 2, 0, 1)|BUILD_CITY": 3.3305, "('play_turn', 2, 0, 1)|BUILD_ROAD": 1.6343, "('play_turn', 2, 0, 1)|BUILD_SETTLEMENT": 3.3198, "('play_turn', 2, 0, 1)|BUY_DEVELOPMENT_CARD": 2.0287, "('play_turn', 2, 0, 1)|END_TURN": 1.6757, "('play_turn', 2, 0, 1)|MARITIME_TRADE": 1.1026, "('play_turn', 2, 0, 1)|PLAY_KNIGHT_CARD": 1.7635, "('play_turn', 2, 0, 1)|PLAY_MONOPOLY": 1.6014, "('play_turn', 2, 0, 1)|PLAY_ROAD_BUILDING": 1.622, "('play_turn', 2, 0, 1)|PLAY_YEAR_OF_PLENTY": 1.7343, "('play_turn', 2, 0, 1)|ROLL": 1.754, "('play_turn', 2, 0, 2)|BUILD_CITY": 3.5808, "('play_turn', 2, 0, 2)|BUILD_ROAD": 1.9298, "('play_turn', 2, 0, 2)|BUILD_SETTLEMENT": 3.2547, "('play_turn', 2, 0, 2)|BUY_DEVELOPMENT_CARD": 2.0257, "('play_turn', 2, 0, 2)|END_TURN": 1.0869, "('play_turn', 2, 0, 2)|MARITIME_TRADE": 1.6104, "('play_turn', 2, 0, 2)|PLAY_KNIGHT_CARD": 1.8981, "('play_turn', 2, 0, 2)|PLAY_YEAR_OF_PLENTY": 1.5396, "('play_turn', 3, -1, 0)|BUILD_ROAD": -0.2139, "('play_turn', 3, -1, 0)|END_TURN": 0.7852, "('play_turn', 3, -1, 0)|PLAY_KNIGHT_CARD": 0.1805, "('play_turn', 3, -1, 0)|PLAY_MONOPOLY": 0.0454, "('play_turn', 3, -1, 0)|PLAY_YEAR_OF_PLENTY": 0.003, "('play_turn', 3, -1, 0)|ROLL": 0.8791, "('play_turn', 3, -1, 1)|BUILD_CITY": 1.1458, "('play_turn', 3, -1, 1)|BUILD_ROAD": 1.3009, "('play_turn', 3, -1, 1)|BUILD_SETTLEMENT": 1.086, "('play_turn', 3, -1, 1)|BUY_DEVELOPMENT_CARD": 0.7435, "('play_turn', 3, -1, 1)|END_TURN": -0.3866, "('play_turn', 3, -1, 1)|MARITIME_TRADE": 0.4951, "('play_turn', 3, -1, 1)|PLAY_KNIGHT_CARD": 1.6073, "('play_turn', 3, -1, 1)|PLAY_MONOPOLY": 0.1553, "('play_turn', 3, -1, 1)|PLAY_ROAD_BUILDING": 1.0791, "('play_turn', 3, -1, 1)|PLAY_YEAR_OF_PLENTY": 0.5665, "('play_turn', 3, -1, 2)|BUILD_CITY": 1.5571, "('play_turn', 3, -1, 2)|BUILD_ROAD": 0.672, "('play_turn', 3, -1, 2)|BUILD_SETTLEMENT": 1.8949, "('play_turn', 3, -1, 2)|BUY_DEVELOPMENT_CARD": 1.4047, "('play_turn', 3, -1, 2)|END_TURN": 0.1184, "('play_turn', 3, -1, 2)|MARITIME_TRADE": 0.6222, "('play_turn', 3, -1, 2)|PLAY_KNIGHT_CARD": 1.5901, "('play_turn', 3, -2, 0)|BUILD_ROAD": -2.3326, "('play_turn', 3, -2, 0)|BUY_DEVELOPMENT_CARD": -1.6857, "('play_turn', 3, -2, 0)|END_TURN": -1.9443, "('play_turn', 3, -2, 0)|PLAY_KNIGHT_CARD": -1.2813, "('play_turn', 3, -2, 0)|PLAY_MONOPOLY": -2.9691, "('play_turn', 3, -2, 0)|PLAY_YEAR_OF_PLENTY": -2.4335, "('play_turn', 3, -2, 0)|ROLL": -1.3818, "('play_turn', 3, -2, 1)|BUILD_CITY": -1.8972, "('play_turn', 3, -2, 1)|BUILD_ROAD": -1.2922, "('play_turn', 3, -2, 1)|BUILD_SETTLEMENT": -1.7391, "('play_turn', 3, -2, 1)|BUY_DEVELOPMENT_CARD": -0.8948, "('play_turn', 3, -2, 1)|END_TURN": -2.2914, "('play_turn', 3, -2, 1)|MARITIME_TRADE": -2.7114, "('play_turn', 3, -2, 1)|PLAY_KNIGHT_CARD": -0.8847, "('play_turn', 3, -2, 1)|PLAY_MONOPOLY": -1.8605, "('play_turn', 3, -2, 1)|PLAY_ROAD_BUILDING": -0.4643, "('play_turn', 3, -2, 1)|PLAY_YEAR_OF_PLENTY": -1.6611, "('play_turn', 3, -2, 1)|ROLL": -0.8149, "('play_turn', 3, -2, 2)|BUILD_CITY": -1.5101, "('play_turn', 3, -2, 2)|BUILD_ROAD": -0.7224, "('play_turn', 3, -2, 2)|BUILD_SETTLEMENT": -0.9192, "('play_turn', 3, -2, 2)|BUY_DEVELOPMENT_CARD": -1.3802, "('play_turn', 3, -2, 2)|END_TURN": -2.9375, "('play_turn', 3, -2, 2)|MARITIME_TRADE": -1.4859, "('play_turn', 3, -2, 2)|PLAY_KNIGHT_CARD": -0.7551, "('play_turn', 3, -2, 2)|PLAY_ROAD_BUILDING": -0.9182, "('play_turn', 3, 0, 0)|BUILD_ROAD": 0.0889, "('play_turn', 3, 0, 0)|BUY_DEVELOPMENT_CARD": 2.2899, "('play_turn', 3, 0, 0)|END_TURN": 1.5057, "('play_turn', 3, 0, 0)|PLAY_KNIGHT_CARD": 2.3889, "('play_turn', 3, 0, 0)|PLAY_MONOPOLY": 1.331, "('play_turn', 3, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.693, "('play_turn', 3, 0, 1)|BUILD_CITY": 2.5471, "('play_turn', 3, 0, 1)|BUILD_ROAD": 2.1325, "('play_turn', 3, 0, 1)|BUILD_SETTLEMENT": 2.4692, "('play_turn', 3, 0, 1)|BUY_DEVELOPMENT_CARD": 2.2618, "('play_turn', 3, 0, 1)|END_TURN": 1.3949, "('play_turn', 3, 0, 1)|MARITIME_TRADE": 1.1747, "('play_turn', 3, 0, 1)|PLAY_KNIGHT_CARD": 1.6572, "('play_turn', 3, 0, 1)|PLAY_MONOPOLY": 1.3076, "('play_turn', 3, 0, 1)|PLAY_ROAD_BUILDING": 2.2548, "('play_turn', 3, 0, 1)|PLAY_YEAR_OF_PLENTY": 1.6937, "('play_turn', 3, 0, 2)|BUILD_CITY": 3.0054, "('play_turn', 3, 0, 2)|BUILD_ROAD": 1.513, "('play_turn', 3, 0, 2)|BUILD_SETTLEMENT": 3.3173, "('play_turn', 3, 0, 2)|BUY_DEVELOPMENT_CARD": 2.4071, "('play_turn', 3, 0, 2)|END_TURN": 1.5899, "('play_turn', 3, 0, 2)|MARITIME_TRADE": 1.7336, "('play_turn', 3, 0, 2)|PLAY_KNIGHT_CARD": 2.8254, "('play_turn', 3, 1, 0)|BUILD_ROAD": 2.819, "('play_turn', 3, 1, 0)|END_TURN": 1.6924, "('play_turn', 3, 1, 0)|PLAY_KNIGHT_CARD": 3.0885, "('play_turn', 3, 1, 0)|PLAY_MONOPOLY": 2.3088, "('play_turn', 3, 1, 0)|PLAY_YEAR_OF_PLENTY": 2.4813, "('play_turn', 3, 1, 1)|BUILD_CITY": 3.7606, "('play_turn', 3, 1, 1)|BUILD_ROAD": 2.1404, "('play_turn', 3, 1, 1)|BUILD_SETTLEMENT": 3.6819, "('play_turn', 3, 1, 1)|BUY_DEVELOPMENT_CARD": 2.9736, "('play_turn', 3, 1, 1)|END_TURN": 2.0928, "('play_turn', 3, 1, 1)|MARITIME_TRADE": 2.0456, "('play_turn', 3, 1, 1)|PLAY_KNIGHT_CARD": 2.6483, "('play_turn', 3, 1, 1)|PLAY_MONOPOLY": 2.5045, "('play_turn', 3, 1, 1)|PLAY_ROAD_BUILDING": 2.8159, "('play_turn', 3, 1, 1)|PLAY_YEAR_OF_PLENTY": 2.4583, "('play_turn', 3, 1, 2)|BUILD_CITY": 3.7044, "('play_turn', 3, 1, 2)|BUILD_ROAD": 2.8126, "('play_turn', 3, 1, 2)|BUILD_SETTLEMENT": 4.0809, "('play_turn', 3, 1, 2)|BUY_DEVELOPMENT_CARD": 3.0729, "('play_turn', 3, 1, 2)|END_TURN": 2.3178, "('play_turn', 3, 1, 2)|MARITIME_TRADE": 2.5801, "('play_turn', 3, 1, 2)|ROLL": 2.5536, "('play_turn', 4, -1, 0)|BUILD_ROAD": -1.3696, "('play_turn', 4, -1, 0)|END_TURN": -1.1889, "('play_turn', 4, -1, 0)|PLAY_KNIGHT_CARD": -0.0769, "('play_turn', 4, -1, 0)|ROLL": 0.2023, "('play_turn', 4, -1, 1)|BUILD_CITY": -0.4996, "('play_turn', 4, -1, 1)|BUILD_ROAD": -1.3271, "('play_turn', 4, -1, 1)|BUILD_SETTLEMENT": 0.3675, "('play_turn', 4, -1, 1)|BUY_DEVELOPMENT_CARD": -0.1076, "('play_turn', 4, -1, 1)|END_TURN": -1.1679, "('play_turn', 4, -1, 1)|MARITIME_TRADE": -1.7993, "('play_turn', 4, -1, 1)|PLAY_KNIGHT_CARD": 0.7183, "('play_turn', 4, -1, 1)|PLAY_MONOPOLY": -0.8414, "('play_turn', 4, -1, 1)|PLAY_ROAD_BUILDING": -0.37, "('play_turn', 4, -1, 1)|PLAY_YEAR_OF_PLENTY": -0.921, "('play_turn', 4, -1, 2)|BUILD_CITY": 1.2155, "('play_turn', 4, -1, 2)|BUILD_ROAD": -0.8712, "('play_turn', 4, -1, 2)|BUILD_SETTLEMENT": 1.0173, "('play_turn', 4, -1, 2)|BUY_DEVELOPMENT_CARD": 0.6003, "('play_turn', 4, -1, 2)|END_TURN": -1.9329, "('play_turn', 4, -1, 2)|MARITIME_TRADE": -1.5443, "('play_turn', 4, -1, 2)|PLAY_KNIGHT_CARD": 1.6219, "('play_turn', 4, -2, 0)|BUILD_ROAD": -3.4755, "('play_turn', 4, -2, 0)|END_TURN": -3.2232, "('play_turn', 4, -2, 0)|PLAY_KNIGHT_CARD": -1.7927, "('play_turn', 4, -2, 0)|ROLL": -1.2271, "('play_turn', 4, -2, 1)|BUILD_CITY": -0.763, "('play_turn', 4, -2, 1)|BUILD_ROAD": -3.2529, "('play_turn', 4, -2, 1)|BUILD_SETTLEMENT": -1.8216, "('play_turn', 4, -2, 1)|BUY_DEVELOPMENT_CARD": -1.459, "('play_turn', 4, -2, 1)|END_TURN": -2.7039, "('play_turn', 4, -2, 1)|MARITIME_TRADE": -3.4723, "('play_turn', 4, -2, 1)|PLAY_KNIGHT_CARD": -2.0338, "('play_turn', 4, -2, 1)|PLAY_MONOPOLY": -2.8023, "('play_turn', 4, -2, 1)|PLAY_ROAD_BUILDING": -1.9908, "('play_turn', 4, -2, 1)|PLAY_YEAR_OF_PLENTY": -2.5061, "('play_turn', 4, -2, 2)|BUILD_CITY": -0.8277, "('play_turn', 4, -2, 2)|BUILD_ROAD": -2.3536, "('play_turn', 4, -2, 2)|BUILD_SETTLEMENT": -0.6095, "('play_turn', 4, -2, 2)|BUY_DEVELOPMENT_CARD": -1.4798, "('play_turn', 4, -2, 2)|END_TURN": -2.9752, "('play_turn', 4, -2, 2)|MARITIME_TRADE": -2.5286, "('play_turn', 4, -2, 2)|PLAY_KNIGHT_CARD": -0.6763, "('play_turn', 4, 0, 0)|BUILD_ROAD": -0.0025, "('play_turn', 4, 0, 0)|END_TURN": 0.4796, "('play_turn', 4, 0, 0)|PLAY_KNIGHT_CARD": 2.0634, "('play_turn', 4, 0, 0)|PLAY_MONOPOLY": 0.1306, "('play_turn', 4, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.5825, "('play_turn', 4, 0, 1)|BUILD_CITY": 2.0349, "('play_turn', 4, 0, 1)|BUILD_ROAD": -0.0966, "('play_turn', 4, 0, 1)|BUILD_SETTLEMENT": 2.1042, "('play_turn', 4, 0, 1)|BUY_DEVELOPMENT_CARD": 2.1253, "('play_turn', 4, 0, 1)|END_TURN": 0.2249, "('play_turn', 4, 0, 1)|MARITIME_TRADE": 0.0482, "('play_turn', 4, 0, 1)|PLAY_KNIGHT_CARD": 2.1811, "('play_turn', 4, 0, 1)|PLAY_MONOPOLY": 0.4841, "('play_turn', 4, 0, 1)|PLAY_ROAD_BUILDING": 2.3283, "('play_turn', 4, 0, 1)|PLAY_YEAR_OF_PLENTY": 0.7655, "('play_turn', 4, 0, 2)|BUILD_CITY": 2.9329, "('play_turn', 4, 0, 2)|BUILD_ROAD": 1.2586, "('play_turn', 4, 0, 2)|BUILD_SETTLEMENT": 3.1948, "('play_turn', 4, 0, 2)|BUY_DEVELOPMENT_CARD": 1.98, "('play_turn', 4, 0, 2)|END_TURN": -0.027, "('play_turn', 4, 0, 2)|MARITIME_TRADE": 0.2605, "('play_turn', 4, 0, 2)|PLAY_KNIGHT_CARD": 1.4995, "('play_turn', 4, 1, 0)|BUILD_ROAD": 1.8997, "('play_turn', 4, 1, 0)|END_TURN": 1.8586, "('play_turn', 4, 1, 0)|PLAY_KNIGHT_CARD": 2.928, "('play_turn', 4, 1, 0)|PLAY_YEAR_OF_PLENTY": 2.1665, "('play_turn', 4, 1, 1)|BUILD_CITY": 3.2318, "('play_turn', 4, 1, 1)|BUILD_ROAD": 1.8023, "('play_turn', 4, 1, 1)|BUILD_SETTLEMENT": 4.2659, "('play_turn', 4, 1, 1)|BUY_DEVELOPMENT_CARD": 3.0225, "('play_turn', 4, 1, 1)|END_TURN": 1.9343, "('play_turn', 4, 1, 1)|MARITIME_TRADE": 1.8175, "('play_turn', 4, 1, 1)|PLAY_KNIGHT_CARD": 2.9977, "('play_turn', 4, 1, 1)|PLAY_MONOPOLY": 1.9767, "('play_turn', 4, 1, 1)|PLAY_ROAD_BUILDING": 3.5483, "('play_turn', 4, 1, 1)|PLAY_YEAR_OF_PLENTY": 2.1415, "('play_turn', 4, 1, 2)|BUILD_CITY": 3.9937, "('play_turn', 4, 1, 2)|BUILD_ROAD": 2.1708, "('play_turn', 4, 1, 2)|BUILD_SETTLEMENT": 4.1644, "('play_turn', 4, 1, 2)|BUY_DEVELOPMENT_CARD": 3.2365, "('play_turn', 4, 1, 2)|END_TURN": 1.5957, "('play_turn', 4, 1, 2)|MARITIME_TRADE": 2.0182, "('play_turn', 4, 1, 2)|ROLL": 2.3236, "('play_turn', 4, 2, 0)|BUILD_ROAD": 2.1326, "('play_turn', 4, 2, 0)|END_TURN": 2.6928, "('play_turn', 4, 2, 0)|PLAY_KNIGHT_CARD": 3.734, "('play_turn', 4, 2, 0)|PLAY_MONOPOLY": 2.8227, "('play_turn', 4, 2, 1)|BUILD_CITY": 4.4875, "('play_turn', 4, 2, 1)|BUILD_ROAD": 2.7712, "('play_turn', 4, 2, 1)|BUILD_SETTLEMENT": 4.2524, "('play_turn', 4, 2, 1)|BUY_DEVELOPMENT_CARD": 3.2083, "('play_turn', 4, 2, 1)|END_TURN": 2.5899, "('play_turn', 4, 2, 1)|MARITIME_TRADE": 2.6831, "('play_turn', 4, 2, 1)|PLAY_KNIGHT_CARD": 3.8399, "('play_turn', 4, 2, 1)|PLAY_MONOPOLY": 2.7825, "('play_turn', 4, 2, 1)|PLAY_YEAR_OF_PLENTY": 3.1378, "('play_turn', 4, 2, 2)|BUILD_CITY": 4.2779, "('play_turn', 4, 2, 2)|BUILD_ROAD": 3.0585, "('play_turn', 4, 2, 2)|BUILD_SETTLEMENT": 4.5135, "('play_turn', 4, 2, 2)|BUY_DEVELOPMENT_CARD": 3.4346, "('play_turn', 4, 2, 2)|END_TURN": 2.5976, "('play_turn', 4, 2, 2)|MARITIME_TRADE": 2.7871, "('play_turn', 4, 2, 2)|PLAY_KNIGHT_CARD": 3.5486, "('play_turn', 5, -1, 0)|BUILD_ROAD": -2.7899, "('play_turn', 5, -1, 0)|END_TURN": -2.1928, "('play_turn', 5, -1, 0)|PLAY_KNIGHT_CARD": -0.5768, "('play_turn', 5, -1, 1)|BUILD_CITY": 0.7545, "('play_turn', 5, -1, 1)|BUILD_ROAD": -2.4264, "('play_turn', 5, -1, 1)|BUILD_SETTLEMENT": -0.087, "('play_turn', 5, -1, 1)|BUY_DEVELOPMENT_CARD": -0.3774, "('play_turn', 5, -1, 1)|END_TURN": -2.6876, "('play_turn', 5, -1, 1)|MARITIME_TRADE": -2.3421, "('play_turn', 5, -1, 1)|PLAY_KNIGHT_CARD": 1.1815, "('play_turn', 5, -1, 1)|PLAY_MONOPOLY": -1.1271, "('play_turn', 5, -1, 1)|PLAY_ROAD_BUILDING": -0.9507, "('play_turn', 5, -1, 1)|PLAY_YEAR_OF_PLENTY": -2.0187, "('play_turn', 5, -1, 2)|BUILD_CITY": 0.0217, "('play_turn', 5, -1, 2)|BUILD_ROAD": -1.7512, "('play_turn', 5, -1, 2)|BUILD_SETTLEMENT": 1.8787, "('play_turn', 5, -1, 2)|BUY_DEVELOPMENT_CARD": 0.6882, "('play_turn', 5, -1, 2)|END_TURN": -1.762, "('play_turn', 5, -1, 2)|MARITIME_TRADE": -2.1344, "('play_turn', 5, 0, 0)|BUILD_ROAD": -1.3658, "('play_turn', 5, 0, 0)|END_TURN": -0.4885, "('play_turn', 5, 0, 0)|PLAY_KNIGHT_CARD": 2.4839, "('play_turn', 5, 0, 0)|PLAY_MONOPOLY": 0.3006, "('play_turn', 5, 0, 0)|ROLL": 1.337, "('play_turn', 5, 0, 1)|BUILD_CITY": 0.4252, "('play_turn', 5, 0, 1)|BUILD_ROAD": 0.7148, "('play_turn', 5, 0, 1)|BUILD_SETTLEMENT": 1.1251, "('play_turn', 5, 0, 1)|BUY_DEVELOPMENT_CARD": 1.2504, "('play_turn', 5, 0, 1)|END_TURN": -1.0588, "('play_turn', 5, 0, 1)|MARITIME_TRADE": -1.1229, "('play_turn', 5, 0, 1)|PLAY_KNIGHT_CARD": 1.9909, "('play_turn', 5, 0, 1)|PLAY_MONOPOLY": -0.2611, "('play_turn', 5, 0, 1)|PLAY_ROAD_BUILDING": 1.4962, "('play_turn', 5, 0, 1)|PLAY_YEAR_OF_PLENTY": -0.4172, "('play_turn', 5, 0, 1)|ROLL": 1.1054, "('play_turn', 5, 0, 2)|BUILD_CITY": 2.4453, "('play_turn', 5, 0, 2)|BUILD_ROAD": 0.168, "('play_turn', 5, 0, 2)|BUILD_SETTLEMENT": 2.6951, "('play_turn', 5, 0, 2)|BUY_DEVELOPMENT_CARD": 1.77, "('play_turn', 5, 0, 2)|END_TURN": -0.9755, "('play_turn', 5, 0, 2)|MARITIME_TRADE": -0.4793, "('play_turn', 5, 0, 2)|PLAY_KNIGHT_CARD": 3.5027, "('play_turn', 5, 1, 0)|BUILD_ROAD": 1.3579, "('play_turn', 5, 1, 0)|END_TURN": 2.1495, "('play_turn', 5, 1, 0)|PLAY_KNIGHT_CARD": 1.5113, "('play_turn', 5, 1, 0)|PLAY_MONOPOLY": 1.5331, "('play_turn', 5, 1, 1)|BUILD_CITY": 4.4026, "('play_turn', 5, 1, 1)|BUILD_ROAD": 1.0347, "('play_turn', 5, 1, 1)|BUILD_SETTLEMENT": 4.1361, "('play_turn', 5, 1, 1)|BUY_DEVELOPMENT_CARD": 1.4419, "('play_turn', 5, 1, 1)|END_TURN": 2.5363, "('play_turn', 5, 1, 1)|MARITIME_TRADE": 1.7163, "('play_turn', 5, 1, 1)|PLAY_KNIGHT_CARD": 3.1321, "('play_turn', 5, 1, 1)|PLAY_MONOPOLY": 2.25, "('play_turn', 5, 1, 1)|PLAY_ROAD_BUILDING": 3.3178, "('play_turn', 5, 1, 1)|PLAY_YEAR_OF_PLENTY": 2.1279, "('play_turn', 5, 1, 2)|BUILD_CITY": 4.1438, "('play_turn', 5, 1, 2)|BUILD_ROAD": 2.2172, "('play_turn', 5, 1, 2)|BUILD_SETTLEMENT": 4.9245, "('play_turn', 5, 1, 2)|BUY_DEVELOPMENT_CARD": 3.4342, "('play_turn', 5, 1, 2)|END_TURN": 1.3565, "('play_turn', 5, 1, 2)|MARITIME_TRADE": 2.5312, "('play_turn', 5, 1, 2)|PLAY_KNIGHT_CARD": 3.4996, "('play_turn', 5, 1, 2)|PLAY_ROAD_BUILDING": 3.9104, "('play_turn', 5, 2, 0)|BUILD_ROAD": 2.9711, "('play_turn', 5, 2, 0)|END_TURN": 3.0323, "('play_turn', 5, 2, 0)|PLAY_KNIGHT_CARD": 2.7979, "('play_turn', 5, 2, 0)|PLAY_MONOPOLY": 2.9551, "('play_turn', 5, 2, 0)|PLAY_YEAR_OF_PLENTY": 3.4447, "('play_turn', 5, 2, 0)|ROLL": 3.1995, "('play_turn', 5, 2, 1)|BUILD_CITY": 4.9256, "('play_turn', 5, 2, 1)|BUILD_ROAD": 4.3377, "('play_turn', 5, 2, 1)|BUILD_SETTLEMENT": 4.6668, "('play_turn', 5, 2, 1)|BUY_DEVELOPMENT_CARD": 4.1041, "('play_turn', 5, 2, 1)|END_TURN": 3.1807, "('play_turn', 5, 2, 1)|MARITIME_TRADE": 2.8128, "('play_turn', 5, 2, 1)|PLAY_KNIGHT_CARD": 3.7958, "('play_turn', 5, 2, 1)|PLAY_MONOPOLY": 3.3776, "('play_turn', 5, 2, 1)|PLAY_ROAD_BUILDING": 4.2843, "('play_turn', 5, 2, 1)|PLAY_YEAR_OF_PLENTY": 3.3164, "('play_turn', 5, 2, 2)|BUILD_CITY": 4.9172, "('play_turn', 5, 2, 2)|BUILD_ROAD": 3.654, "('play_turn', 5, 2, 2)|BUILD_SETTLEMENT": 4.678, "('play_turn', 5, 2, 2)|BUY_DEVELOPMENT_CARD": 3.6781, "('play_turn', 5, 2, 2)|END_TURN": 3.1357, "('play_turn', 5, 2, 2)|MARITIME_TRADE": 2.5796, "('play_turn', 5, 2, 2)|PLAY_KNIGHT_CARD": 4.8915, "('play_turn', 6, 0, 0)|BUILD_ROAD": -1.8865, "('play_turn', 6, 0, 0)|END_TURN": 0.0482, "('play_turn', 6, 0, 0)|PLAY_KNIGHT_CARD": -0.486, "('play_turn', 6, 0, 0)|ROLL": 0.7364, "('play_turn', 6, 0, 1)|BUILD_CITY": 5.9817, "('play_turn', 6, 0, 1)|BUILD_ROAD": -1.1739, "('play_turn', 6, 0, 1)|BUILD_SETTLEMENT": 6.0, "('play_turn', 6, 0, 1)|BUY_DEVELOPMENT_CARD": 0.6496, "('play_turn', 6, 0, 1)|END_TURN": 0.0335, "('play_turn', 6, 0, 1)|MARITIME_TRADE": 0.036, "('play_turn', 6, 0, 1)|PLAY_KNIGHT_CARD": 2.3628, "('play_turn', 6, 0, 1)|PLAY_MONOPOLY": 1.4518, "('play_turn', 6, 0, 1)|PLAY_ROAD_BUILDING": 1.2903, "('play_turn', 6, 0, 1)|PLAY_YEAR_OF_PLENTY": 0.9126, "('play_turn', 6, 0, 2)|BUILD_CITY": 5.9975, "('play_turn', 6, 0, 2)|BUILD_ROAD": 0.576, "('play_turn', 6, 0, 2)|BUILD_SETTLEMENT": 6.0, "('play_turn', 6, 0, 2)|BUY_DEVELOPMENT_CARD": 1.8681, "('play_turn', 6, 0, 2)|END_TURN": -0.8405, "('play_turn', 6, 0, 2)|MARITIME_TRADE": 0.2334, "('play_turn', 6, 0, 2)|PLAY_KNIGHT_CARD": 2.5271, "('play_turn', 6, 1, 0)|BUILD_ROAD": -0.2226, "('play_turn', 6, 1, 0)|END_TURN": 1.2136, "('play_turn', 6, 1, 0)|PLAY_KNIGHT_CARD": 2.8423, "('play_turn', 6, 1, 1)|BUILD_CITY": 5.9913, "('play_turn', 6, 1, 1)|BUILD_ROAD": 0.9638, "('play_turn', 6, 1, 1)|BUILD_SETTLEMENT": 6.0, "('play_turn', 6, 1, 1)|BUY_DEVELOPMENT_CARD": 2.6301, "('play_turn', 6, 1, 1)|END_TURN": 0.1327, "('play_turn', 6, 1, 1)|MARITIME_TRADE": 0.5144, "('play_turn', 6, 1, 1)|PLAY_KNIGHT_CARD": 2.0733, "('play_turn', 6, 1, 1)|PLAY_MONOPOLY": 1.6292, "('play_turn', 6, 1, 1)|PLAY_ROAD_BUILDING": 3.1023, "('play_turn', 6, 1, 1)|PLAY_YEAR_OF_PLENTY": 2.3934, "('play_turn', 6, 1, 2)|BUILD_CITY": 6.0, "('play_turn', 6, 1, 2)|BUILD_ROAD": 1.466, "('play_turn', 6, 1, 2)|BUILD_SETTLEMENT": 6.0, "('play_turn', 6, 1, 2)|BUY_DEVELOPMENT_CARD": 2.8293, "('play_turn', 6, 1, 2)|END_TURN": 1.026, "('play_turn', 6, 1, 2)|MARITIME_TRADE": 1.9043, "('play_turn', 6, 1, 2)|PLAY_KNIGHT_CARD": 4.566, "('play_turn', 6, 2, 0)|BUILD_ROAD": 3.1526, "('play_turn', 6, 2, 0)|BUY_DEVELOPMENT_CARD": 4.5694, "('play_turn', 6, 2, 0)|END_TURN": 3.7216, "('play_turn', 6, 2, 0)|PLAY_KNIGHT_CARD": 4.6389, "('play_turn', 6, 2, 0)|PLAY_MONOPOLY": 2.6755, "('play_turn', 6, 2, 0)|PLAY_YEAR_OF_PLENTY": 3.6491, "('play_turn', 6, 2, 0)|ROLL": 3.6675, "('play_turn', 6, 2, 1)|BUILD_CITY": 6.0, "('play_turn', 6, 2, 1)|BUILD_ROAD": 4.3219, "('play_turn', 6, 2, 1)|BUILD_SETTLEMENT": 6.0, "('play_turn', 6, 2, 1)|BUY_DEVELOPMENT_CARD": 4.6233, "('play_turn', 6, 2, 1)|END_TURN": 3.8957, "('play_turn', 6, 2, 1)|MARITIME_TRADE": 3.1169, "('play_turn', 6, 2, 1)|PLAY_KNIGHT_CARD": 3.9706, "('play_turn', 6, 2, 1)|PLAY_MONOPOLY": 3.4313, "('play_turn', 6, 2, 1)|PLAY_ROAD_BUILDING": 3.7063, "('play_turn', 6, 2, 1)|PLAY_YEAR_OF_PLENTY": 3.7507, "('play_turn', 6, 2, 1)|ROLL": 3.9218, "('play_turn', 6, 2, 2)|BUILD_CITY": 6.0, "('play_turn', 6, 2, 2)|BUILD_ROAD": 3.6866, "('play_turn', 6, 2, 2)|BUILD_SETTLEMENT": 6.0, "('play_turn', 6, 2, 2)|BUY_DEVELOPMENT_CARD": 4.1041, "('play_turn', 6, 2, 2)|END_TURN": 2.839, "('play_turn', 6, 2, 2)|MARITIME_TRADE": 4.1378, "('play_turn', 6, 2, 2)|PLAY_KNIGHT_CARD": 4.8081}

