"""Offline Q-learning for the Catan agent. Run from the repo root:  python train_catan.py

The learner chooses an action TYPE with epsilon-greedy over the Q-table. The
concrete action of that type comes from the agent's own pick_of_type().
The opponent is the frozen v1 baseline. The result is written to q_table_line.py.
"""
import importlib.util
import random
import time

from games.catan.engine import SIDE_OF, adjudicate, legal_actions, new_game, observation_for, victory_points


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = load("bots/catan_agent.py", "catan_v2")   # same state_features as the file we hand in
v1 = load("bots/catan_v1.py", "catan_v1")         # frozen baseline

TRAIN_GAMES = 20000
ALPHA = 0.1                # learning rate
GAMMA = 0.95               # discount
EPS_START, EPS_END = 0.30, 0.05   # exploration, decays over training
WIN_BONUS = 5.0            # extra reward for winning, minus this for losing
MIN_VISITS = 30            # keys tried fewer times than this are not exported

rng = random.Random(1)     # our own generator: new_game() reseeds the global one


def play_one_game(policy_a, policy_b, seed, on_decision=None):
    """From docs/students/TRAINING.md, unchanged."""
    game = new_game(seed, vps_to_win=7)
    policies = {"A": policy_a, "B": policy_b}
    while game.winning_color() is None and game.state.num_turns // 2 < 100:
        actions = legal_actions(game)
        if len(actions) == 1:
            game.execute(actions[0])
            continue
        side = SIDE_OF[game.state.current_color()]
        observation = observation_for(game, side, actions, None, {"max_rounds": 100})
        action_id = policies[side](observation)
        if on_decision is not None:
            on_decision(side, observation, action_id)
        game.execute(actions[action_id])
    champion = game.winning_color()
    winner = SIDE_OF[champion] if champion is not None else adjudicate(game)[0]
    return winner, victory_points(game)


Q = {}        # key -> learned value
visits = {}   # key -> how many times we tried it
state = {"eps": EPS_START, "learner": "A", "prev": None}


def kinds_of(obs):
    return sorted({a["type"] for a in obs["actions"]})


def is_choice(obs):
    """True when the table has something to decide: 2+ action types, after placement."""
    return len(kinds_of(obs)) >= 2 and obs["phase"] != "initial_placement"


def learner_policy(obs):
    if not is_choice(obs):
        return v1.heuristic_policy(obs)
    kinds = kinds_of(obs)
    if rng.random() < state["eps"]:
        kind = rng.choice(kinds)                     # explore
    else:                                            # exploit; ties go to the baseline's choice
        base_id = v1.heuristic_policy(obs)
        base_kind = next(a["type"] for a in obs["actions"] if a["id"] == base_id)
        kind = max(kinds, key=lambda k: (Q.get(agent.q_key(obs, k), 0.0), k == base_kind))
    return agent.pick_of_type(obs, kind)["id"]


def on_decision(side, obs, action_id):
    if side != state["learner"] or not is_choice(obs):
        return
    vp = obs["me"]["victory_points"]
    if state["prev"] is not None:                    # close the previous step
        key, old_vp = state["prev"]
        best_next = max(Q.get(agent.q_key(obs, k), 0.0) for k in kinds_of(obs))
        Q[key] = agent.q_update(Q.get(key, 0.0), vp - old_vp, best_next, ALPHA, GAMMA)
    kind = next(a["type"] for a in obs["actions"] if a["id"] == action_id)
    key = agent.q_key(obs, kind)
    visits[key] = visits.get(key, 0) + 1
    state["prev"] = (key, vp)


def train():
    started, wins = time.time(), 0
    for game_number in range(1, TRAIN_GAMES + 1):
        state["eps"] = EPS_START + (EPS_END - EPS_START) * game_number / TRAIN_GAMES
        state["learner"] = "A" if game_number % 2 else "B"   # alternate sides
        state["prev"] = None
        if state["learner"] == "A":
            winner, points = play_one_game(learner_policy, v1.heuristic_policy, game_number, on_decision)
        else:
            winner, points = play_one_game(v1.heuristic_policy, learner_policy, game_number, on_decision)
        if state["prev"] is not None:                # last step: there is no next state
            key, old_vp = state["prev"]
            bonus = WIN_BONUS if winner == state["learner"] else (-WIN_BONUS if winner else 0.0)
            reward = points[state["learner"]] - old_vp + bonus
            Q[key] = agent.q_update(Q.get(key, 0.0), reward, 0.0, ALPHA, GAMMA)
        wins += winner == state["learner"]
        if game_number % 1000 == 0:
            print(f"game {game_number}: learner won {wins / 10:.1f}% of the last 1000 "
                  f"(eps {state['eps']:.2f}, {len(Q)} keys, {time.time() - started:.0f}s)")
            wins = 0

    table = {k: round(v, 4) for k, v in sorted(Q.items()) if visits.get(k, 0) >= MIN_VISITS}
    with open("q_table_line.py", "w") as f:
        f.write(f"Q = {table!r}\n")
    print(f"wrote q_table_line.py: kept {len(table)} of {len(Q)} keys (>= {MIN_VISITS} visits)")


if __name__ == "__main__":
    train()
