import crafter
from crafter.env import Env
from IPython import embed
from utils.utils import plot_semantic_map
import matplotlib.pyplot as plt

env = Env()
env = crafter.Recorder(
  env, 'crafter_logs',
  save_stats=False,
  save_video=True,
  save_episode=False,
)

obs = env.reset()
done = False
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
  img = plot_semantic_map((info["semantic"]), info["player_pos"])
  break
  env.render()