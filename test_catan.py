import json
import bots.catan_agent as agent

FILES = ["catan_initial_placement", "catan_play_turn", "catan_move_robber"]

for name in FILES:
    obs = json.load(open(f"docs/students/sample_observations/{name}.json"))
    answer = agent.act(obs)
    legal = [a["id"] for a in obs["actions"]]
    assert answer["action"] in legal, f"{name}: illegal action {answer}"
    print(name, "-> action", answer["action"], "OK")
