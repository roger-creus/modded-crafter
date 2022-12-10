import gym
import numpy as np
import crafter
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)

def make_crafter(env_id, scale=False, gray_scale=False, frame_stack = 1, image_size=64, capture_video = False):
    env = gym.make(env_id)
    
    env = WarpFramePyTorch(env, gray_scale, image_size)
    
    if frame_stack > 1:
        env = gym.wrappers.FrameStack(env, frame_stack)
    
    if scale:
        env = ScaledFloatFrame(env)

    env = crafter.Recorder(
        env, './crafter_logs',
        save_stats=False,
        save_video=capture_video,
        save_episode=False,
    )
    return env

class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat=1):
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self.repeat and not done:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info

class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super(TimeLimit, self).__init__(env)
        self._duration = duration
        self._step = 0
    
    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            info['time_limit_reached'] = True
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()

class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete), "This wrapper only works with discrete action space"
        shape = (env.action_space.n,)
        env.action_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        env.action_space.sample = self._sample_action
        super(OneHotAction, self).__init__(env)
    
    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        return self.env.step(index)

    def reset(self):
        return self.env.reset()
    
    def _sample_action(self):
        actions = self.env.action_space.shape[0]
        index = np.random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference

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
