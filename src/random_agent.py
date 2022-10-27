import crafter
from crafter.env import Env
from IPython import embed

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
  print(info["semantic"])
  env.render()