import argparse
import cv2
import gym
import copy
import os
import numpy as np

from dqn import DQN
from actions import get_action_space

from chainer import functions as F
from chainer import links as L
from chainer import optimizers

from chainerrl.action_value import DiscreteActionValue
from chainerrl import agents
from chainerrl import explorers
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl import replay_buffer


def phi(states):
    return np.asarray(states, dtype=np.float32) / 255.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Breakout-v0')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--update-interval', type=int, default=4)
    args = parser.parse_args()

    env = gym.make(args.env)

    actions = get_action_space(args.env)
    n_actions = len(actions)

    q_func = links.Sequence(
            links.NatureDQNHead(n_input_channels=args.update_interval),
            L.Linear(512, n_actions),
            DiscreteActionValue)

    opt = optimizers.RMSpropGraves(
        lr=2.5e-4, alpha=0.95, momentum=0.0, eps=1e-2)
    opt.setup(q_func)

    rbuf = replay_buffer.ReplayBuffer(10 ** 5)

    explorer = explorers.ConstantEpsilonGreedy(0, None)

    agent = DQN(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                  update_interval=args.update_interval,
                  explorer=explorer, phi=phi)

    if args.load:
        agent.load(args.load)

    global_step = 0
    episode = 0

    while True:
        states = np.zeros((args.update_interval, 84, 84), dtype=np.uint8)
        reward = 0
        done = False
        sum_of_rewards = 0
        step = 0
        state = env.reset()

        while True:
            if args.render:
                env.render()

            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            state = cv2.resize(state, (84, 84))
            states = np.roll(states, 1, axis=0)
            states[0] = state

            action = actions[agent.act(states)]

            if done:
                break

            state, reward, done, info = env.step(action)

            sum_of_rewards += reward
            step += 1
            global_step += 1

        episode += 1

        print('Episode: {}, Step: {}: Reward: {}'.format(
                episode, global_step, sum_of_rewards))

if __name__ == '__main__':
    main()
