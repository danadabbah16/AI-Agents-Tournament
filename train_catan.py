import random
import importlib.util

from games.catan.engine import SIDE_OF, adjudicate, legal_actions, new_game, observation_for, victory_points


def play_one_game(policy_a, policy_b, seed, on_decision=None):
    """Plays a whole game. A policy takes an observation and returns an action id.

    on_decision(side, observation, action_id) is called at every decision, so you
    can record what happened. Returns the winner ("A", "B" or None for a draw) and
    the points, like {"A": 7, "B": 2}, with the same result the judge would give.
    Use a positive seed: with 0, the engine picks a random game.
    """
    game = new_game(seed, vps_to_win=7)
    policies = {"A": policy_a, "B": policy_b}
    while game.winning_color() is None and game.state.num_turns // 2 < 100:
        actions = legal_actions(game)
        if len(actions) == 1:
            game.execute(actions[0])  # the judge plays forced actions for you
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



def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = load("bots/catan_agent.py", "catan_v2")
v1 = load("bots/catan_v1.py", "catan_v1")

states = set()
decisions = 0


def on_decision(side, observation, action_id):
    global decisions
    decisions += 1
    states.add(tuple(agent.state_features(observation)))


for seed in range(1, 201):
    play_one_game(v1.heuristic_policy, v1.heuristic_policy, seed, on_decision)
print(decisions, "decisions,", len(states), "distinct states")