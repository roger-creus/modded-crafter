import gym
import numpy as np
from gym import spaces
import crafter
import cv2
from crafter.env import Env
cv2.ocl.setUseOpenCL(False)

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        Return only every `skip`-th frame (frameskipping)
        :param env: (Gym Environment) the environment
        :param skip: (int) number of `skip`-th frame
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,)+env.observation_space.shape,
            dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.
        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward,
                 done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=env.observation_space.shape,
            dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class WarpFramePyTorch(gym.ObservationWrapper):
    def __init__(self, env, gray_scale=False, image_size=64):
        """
        Warp frames to (image_size, image_size) as done in the Nature paper
        and later work.
        :param env: (Gym Environment) the environment
        :param image_size: (int) the size of the image
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = image_size
        self.height = image_size
        self.gray_scale = gray_scale
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(1 if gray_scale else 3, self.height, self.width),
            dtype=env.observation_space.dtype)

    def observation(self, frame):
        """
        returns the current observation from a frame
        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        if self.gray_scale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

            return frame[None,:, :]
        else:
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            frame = np.transpose(frame, (2, 0, 1))
            return frame

def make_semantic_crafter():
    env = Env(use_semantic = True)
    return env
    

def make_crafter(env_id, seed = 1, scale=False, gray_scale=False, frame_stack = 1, image_size=64, capture_video = False):
    env = gym.make(env_id)
    
    env = WarpFramePyTorch(env, gray_scale, image_size)
    
    #env = MaxAndSkipEnv(env, skip=4)
    
    if frame_stack > 1:
        env = gym.wrappers.FrameStack(env, frame_stack)
    
    if scale:
        env = ScaledFloatFrame(env)

    env = crafter.Recorder(
        env, './crafter_logs',
        save_stats=True,
        save_video=capture_video,
        save_episode=False,
    )
    return env

def make_env(env_id, seed, idx, capture_video, run_name, use_semantic = False):
    def thunk():
        if not use_semantic:
            env = make_crafter(env_id=env_id, seed=seed, scale=True, gray_scale=False, frame_stack = 1, image_size=64, capture_video = capture_video)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        else:
            env = make_semantic_crafter()
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
            
    return thunk