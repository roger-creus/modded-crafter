import crafter
from crafter.env import Env
from IPython import embed
from utils.utils import plot_local_semantic_map, plot_local_semantic_map_from_global
import matplotlib.pyplot as plt

env = Env()
env = crafter.Recorder(
  env, 'crafter_logs',
  save_stats=False,
  save_video=False,
  save_episode=False,
)

obs = env.reset()
done = False
while not done:
  action = env.action_space.sample()
  obs, reward,  done, info = env.step(action)
  #img = plot_local_semantic_map_from_global(info["semantic"], info["player_pos"], info["inventory"])