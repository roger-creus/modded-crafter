import collections

import numpy as np

from . import constants
from . import engine
from . import objects
from . import worldgen

from IPython import embed
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Gym is an optional dependency.
try:
  import gym
  DiscreteSpace = gym.spaces.Discrete
  BoxSpace = gym.spaces.Box
  DictSpace = gym.spaces.Dict
  BaseClass = gym.Env
except ImportError:
  DiscreteSpace = collections.namedtuple('DiscreteSpace', 'n')
  BoxSpace = collections.namedtuple('BoxSpace', 'low, high, shape, dtype')
  DictSpace = collections.namedtuple('DictSpace', 'spaces')
  BaseClass = object


class Env(BaseClass):

  def __init__(
      self, area=(64, 64), view=(9, 9), size=(64, 64),
      reward=True, length=10000, seed=None, use_semantic = False):
    view = np.array(view if hasattr(view, '__len__') else (view, view))
    size = np.array(size if hasattr(size, '__len__') else (size, size))
    seed = np.random.randint(0, 2**31 - 1) if seed is None else seed
    self._area = area
    self._view = view
    self._size = size
    self._reward = reward
    self._length = length
    self._seed = seed
    self._episode = 0
    self._world = engine.World(area, constants.materials, (12, 12))
    self._textures = engine.Textures(constants.root / 'assets')
    item_rows = int(np.ceil(len(constants.items) / view[0]))
    self._global_view = engine.GlobalView(
        self._world, self._textures, [area[0], area[1]])    
    self._local_view = engine.LocalView(
        self._world, self._textures, [view[0], view[1] - item_rows])
    self._item_view = engine.ItemView(
        self._textures, [view[0], item_rows])
    self._sem_view = engine.SemanticView(self._world, [
        objects.Player, objects.Cow, objects.Zombie,
        objects.Skeleton, objects.Arrow, objects.Plant])
    self._step = None
    self._player = None
    self._last_health = None
    self._unlocked = None
    # Some libraries expect these attributes to be set.
    self.reward_range = None
    self.metadata = None

    obj_types = ["player", "cow", "zombie", "skeleton", "arrow", "plant"]
    self._mat_ids = self._world._mat_ids.copy()
    self._obj_ids = {obj_types[i]: len(self._mat_ids) + i for i in range(len(obj_types))}

    self.use_semantic = use_semantic

  @property
  def observation_space(self):
    if not self.use_semantic:
      return BoxSpace(0, 255, tuple(self._size) + (3,), np.uint8)
    else:
      # 7*9=63 for the image + 16 for the inventory
      return BoxSpace(0, 1, (1,35,7,9), np.float64)

  @property
  def action_space(self):
    return DiscreteSpace(len(constants.actions))

  @property
  def action_names(self):
    return constants.actions

  def reset(self, init_pos = "center"):
    center = (self._world.area[0] // 2, self._world.area[1] // 2)
    self._episode += 1
    self._step = 0
    #self._world.reset(seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
    self._world.reset(seed=hash((self._seed, 0)) % (2 ** 31 - 1))
    self._update_time()
    self._player = objects.Player(self._world, center)
    self._last_health = self._player.health
    self._world.add(self._player)
    self._unlocked = set()
    worldgen.generate_world(self._world, self._player)

    # to start each episode at a random position (useful for collecting trajectories)
    if init_pos == "random":
      correct_pos = False
      while not correct_pos:
        random_x = np.random.randint(self._world.area[0])
        random_y = np.random.randint(self._world.area[1])
        center = (random_x, random_y)

        if self._world[center[0], center[1]][0] in constants.walkable and self._world._obj_map[center] == 0:
          correct_pos = True
    
      # set the player to the chosen positions. if it is not random, then its just the center of the map
      self._world.move(self._player, center)

    if not self.use_semantic:
      return self._obs()
    else:
      return self._semantic_view()
    
    #return None

  def step(self, action):
    self._step += 1
    self._update_time()
    self._player.action = constants.actions[action]
    for obj in self._world.objects:
      if self._player.distance(obj) < 2 * max(self._view):
        obj.update()
    if self._step % 10 == 0:
      for chunk, objs in self._world.chunks.items():
        # xmin, xmax, ymin, ymax = chunk
        # center = (xmax - xmin) // 2, (ymax - ymin) // 2
        # if self._player.distance(center) < 4 * max(self._view):
        self._balance_chunk(chunk, objs)
    
    
    if not self.use_semantic:
      obs = self._obs()
    else:
      obs = self._semantic_view()
    
    reward = (self._player.health - self._last_health) / 10
    self._last_health = self._player.health
    unlocked = {
        name for name, count in self._player.achievements.items()
        if count > 0 and name not in self._unlocked}
    if unlocked:
      self._unlocked |= unlocked
      reward += 1.0
    dead = self._player.health <= 0
    over = self._length and self._step >= self._length
    done = dead or over
    info = {
        'inventory': self._player.inventory.copy(),
        'achievements': self._player.achievements.copy(),
        'discount': 1 - float(dead),
        'semantic': self._sem_view(),
        'player_pos': self._player.pos,
        'reward': reward,
    }
    if not self._reward:
      reward = 0.0
    return obs, reward, done, info

  def render(self, size=None):
    size = size or self._size
    unit = size // self._view
    canvas = np.zeros(tuple(size) + (3,), np.uint8)
    local_view = self._local_view(self._player, unit)
    item_view = self._item_view(self._player.inventory, unit)
    view = np.concatenate([local_view, item_view], 1)
    border = (size - (size // self._view) * self._view) // 2
    (x, y), (w, h) = border, view.shape[:2]
    canvas[x: x + w, y: y + h] = view
    return canvas.transpose((1, 0, 2))


  def _semantic_view(self, one_hot_encoding=True):
    x, y = self._player.pos[0], self._player.pos[1]
    
    canvas = np.zeros((7,9))
    for i in range((x-3) - 1, (x-3) + 9 - 1):
      for j in range((y-4) - 1, (y-4) + 7 - 1):
        mat, obj = self._world[(i,j)]
        if mat is None and obj is None:
          continue
        #elif obj is not None:
        #  canvas[i - x + 3, j - y + 4] = self._obj_ids[obj.name]
        #else:
        canvas[j - y + 4, i - x + 3] = self._mat_ids[mat]

    plot_local_mask_without_inventory(canvas)    
    
    """
    #mat_map = self._world.return_mat_mask((x - 3), (x - 3) + 7, (y - 4),  (y - 4) + 9)
    #obj_map = self._world.return_obj_mask((x - 3), (x - 3) + 7, (y - 4),  (y - 4) + 9)

    for i in range(obj_map.shape[0]):
      for j in range(obj_map.shape[1]):

    #sem_view = self._sem_view()
    #sem_view = sem_view_pad[]

    if one_hot_encoding:

      one_hot = F.one_hot(torch.tensor(sem_view.astype(int))).permute(2,0,1)

      if one_hot.size(0) != 19:
        one_hot = torch.vstack([one_hot, torch.tensor(np.zeros((19 - one_hot.size(0), 7, 9 )))])
      
      inventory = self._player.inventory.copy()
      inventory = np.fromiter(inventory.values(), dtype=int)

      inventory_tiles = []
      for val in inventory:
        tile = np.tile(val / 9, (1,7,9))
        inventory_tiles.append(tile)

      inventory_tiles = np.concatenate(inventory_tiles)
      inventory_tiles = torch.tensor(inventory_tiles)

      obs = torch.vstack([one_hot, inventory_tiles])
      obs = np.expand_dims(obs, axis=0)

    else:
    """
    sem_view_flatten = canvas.flatten()

    inventory = self._player.inventory.copy()
    inventory = np.fromiter(inventory.values(), dtype=int)

    obs = np.concatenate([sem_view_flatten, inventory])
    obs = np.pad(obs,[0,2], mode="constant", constant_values=-1).reshape(9,9)

    #plot_local_mask(obs)

    obs = np.expand_dims(obs, axis=0)

    return obs

  def _obs(self):
    return self.render()

  def _update_time(self):
    # https://www.desmos.com/calculator/grfbc6rs3h
    progress = (self._step / 300) % 1 + 0.3
    daylight = 1 - np.abs(np.cos(np.pi * progress)) ** 3
    self._world.daylight = daylight

  def _balance_chunk(self, chunk, objs):
    light = self._world.daylight
    self._balance_object(
        chunk, objs, objects.Zombie, 'grass', 6, 0, 0.3, 0.4,
        lambda pos: objects.Zombie(self._world, pos, self._player),
        lambda num, space: (
            0 if space < 50 else 3.5 - 3 * light, 3.5 - 3 * light))
    self._balance_object(
        chunk, objs, objects.Skeleton, 'path', 7, 7, 0.1, 0.1,
        lambda pos: objects.Skeleton(self._world, pos, self._player),
        lambda num, space: (0 if space < 6 else 1, 2))
    self._balance_object(
        chunk, objs, objects.Cow, 'grass', 5, 5, 0.01, 0.1,
        lambda pos: objects.Cow(self._world, pos),
        lambda num, space: (0 if space < 30 else 1, 1.5 + light))

  def _balance_object(
      self, chunk, objs, cls, material, span_dist, despan_dist,
      spawn_prob, despawn_prob, ctor, target_fn):
    xmin, xmax, ymin, ymax = chunk
    random = self._world.random
    creatures = [obj for obj in objs if isinstance(obj, cls)]
    mask = self._world.mask(*chunk, material)
    target_min, target_max = target_fn(len(creatures), mask.sum())
    if len(creatures) < int(target_min) and random.uniform() < spawn_prob:
      xs = np.tile(np.arange(xmin, xmax)[:, None], [1, ymax - ymin])
      ys = np.tile(np.arange(ymin, ymax)[None, :], [xmax - xmin, 1])
      xs, ys = xs[mask], ys[mask]
      i = random.randint(0, len(xs))
      pos = np.array((xs[i], ys[i]))
      empty = self._world[pos][1] is None
      away = self._player.distance(pos) >= span_dist
      if empty and away:
        self._world.add(ctor(pos))
    elif len(creatures) > int(target_max) and random.uniform() < despawn_prob:
      obj = creatures[random.randint(0, len(creatures))]
      away = self._player.distance(obj.pos) >= despan_dist
      if away:
        self._world.remove(obj)

  def render_world(self, size=None):
    size = size or self._size
    global_view = self._local_view(self._player, size)
    return global_view / 255.