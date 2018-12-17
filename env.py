import numpy as np
import cv2
import gym

from gym.spaces import Discrete
from atari_constants import ENV_LIST, PONG_ENV_LIST
from collections import deque


def make_env(env_name):
    if env_name in PONG_ENV_LIST:
        return gym.make('PongDeterministic-v4')
    else:
        return gym.make(env_name)

class Env:
    def __init__(self, num_stack=4):
        envs = [make_env(ENV_LIST[i]) for i in range(len(ENV_LIST))]
        self.observation_space = envs[0].observation_space
        self.envs = envs
        num_actions = max(envs, key=lambda e: e.action_space.n).action_space.n
        self.action_space = Discrete(num_actions)
        self.num_stack = num_stack
        self.queue = deque(maxlen=num_stack)
        self.current_index = 0
        self.raw_reward = 0
        self.local_steps = [0 for _ in range(len(ENV_LIST))]

    def state_preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        if ENV_LIST[self.current_index] == 'White':
            gray = 255 - gray
        elif ENV_LIST[self.current_index] == 'Scale':
            small = cv2.resize(gray, (42, 42))
            gray = np.zeros((84, 84), dtype=np.uint8)
            gray[5:47, 5:47] = small
        return cv2.resize(gray, (84, 84))

    def reward_preprocess(self, reward):
        return np.clip(reward, -1.0, 1.0)

    def step(self, action):
        # change out of range action to noop
        if action >= self.envs[self.current_index].action_space.n:
            action = 0
        obs, reward, done, info = self.envs[self.current_index].step(action)
        obs = self.state_preprocess(obs)
        self.queue.append(obs)
        self.raw_reward += reward
        reward = self.reward_preprocess(reward)
        self.local_steps[self.current_index] += 1
        return obs, reward, done, info

    def reset(self):
        self.raw_reward = 0
        self.current_index = np.random.randint(len(self.envs))
        obs = self.envs[self.current_index].reset()
        obs = self.state_preprocess(obs)
        empty_obs = np.zeros_like(obs)
        for i in range(self.num_stack - 1):
            self.queue.append(empty_obs)
        self.queue.append(obs)
        return obs

    def get_detail(self):
        return dict(reward=self.raw_reward)

    def current_env_name(self):
        return ENV_LIST[self.current_index]

    def get_num_actions(self):
        return self.num_actions

    def get_local_step(self, env_name):
        index = ENV_LIST.index(env_name)
        return self.local_steps[index]
