import argparse
import numpy as np
from multiagent.TAD_core import World
from multiagent.TAD_rand_2t1a1d import Scenario
from multiagent.TAD_environment import MultiAgentGraphEnv
from onpolicy import global_var as glv

glv._init()

parser = argparse.ArgumentParser()
parser.add_argument("--num_target", type=int, default=2)
parser.add_argument("--num_attacker", type=int, default=1)
parser.add_argument("--num_defender", type=int, default=1)
parser.add_argument("--max_edge_dist", type=float, default=2.0)
parser.add_argument("--graph_feat_type", type=str, default="global")
parser.add_argument("--episode_length", type=int, default=100)
parser.add_argument("--use_curriculum", action="store_true", default=False)
args = parser.parse_args([])

scenario = Scenario()
world = scenario.make_world(args)

# Map our gym env
env = MultiAgentGraphEnv(
    world=world,
    reset_callback=scenario.reset_world,
    reward_callback=scenario.reward,
    observation_callback=scenario.observation,
    graph_observation_callback=scenario.graph_observation,
    id_callback=scenario.get_id,
    info_callback=scenario.benchmark_data,
    done_callback=scenario.done if hasattr(scenario, "done") else None,
    discrete_action=False,
    update_graph=scenario.update_graph
)

obs = env.reset()

# MultiDiscrete actions: target belief and lock switch. Let's pass integers.
# The original code expects action[0], action[1] from a MultiDiscrete output.
actions = [[np.array([0]), np.array([0])]]

obs_n, reward_n, done_n, info_n = env.step(actions)
print("Step completed successfully!")
print("Node obs shape after step:", obs_n[2][0].shape)
print("Adjacency shape after step:", obs_n[3][0].shape)
