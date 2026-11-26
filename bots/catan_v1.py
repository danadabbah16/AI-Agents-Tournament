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
        return {"action": heuristic_policy(observation)}
    except Exception:
        return {"action": observation["actions"][0]["id"]}


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
    """Short list describing the state. Must be deterministic."""
    me = observation["me"]
    opp = observation["opponent"]
    return [
        observation.get("phase"),
        me.get("victory_points", me.get("visible_victory_points", 0)),
        opp.get("visible_victory_points", 0),
        me.get("resource_cards", 0),
        len(me.get("settlements", [])),
        len(me.get("cities", [])),
        me.get("longest_road_length", 0),
        me.get("played_development_cards", {}).get("KNIGHT", 0),
        int(me.get("has_longest_road", False)),
        int(me.get("has_largest_army", False)),
    ]


def action_value(observation, action_id):
    """Estimate of how good this action is."""
    actions = observation["actions"]
    action = next((a for a in actions if a["id"] == action_id), None)
    if action is None:
        return 0.0
    try:
        if action_id == heuristic_policy(observation):
            return 10.0
    except Exception:
        pass
    rank = {kind: len(PRIORITY) - i for i, kind in enumerate(PRIORITY)}
    return float(rank.get(action["type"], 0))


def q_update(q, reward, best_next_q, alpha, gamma):
    """Q-learning update rule."""
    return q + alpha * (reward + gamma * best_next_q - q)
