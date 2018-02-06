import argparse
import cv2
import gym
import copy
import os
import numpy as np
import tensorflow as tf

from lightsaber.tensorflow.util import initialize
from lightsaber.tensorflow.log import TfBoardLogger
from lightsaber.rl.explorer import LinearDecayExplorer
from lightsaber.rl.replay_buffer import ReplayBuffer
from lightsaber.rl.trainer import Trainer
from actions import get_action_space
from network import make_cnn
from agent import Agent
from datetime import datetime


def main():
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongDeterministic-v4')
    parser.add_argument('--outdir', type=str, default=date)
    parser.add_argument('--logdir', type=str, default=date)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--final-exploration-frames', type=int, default=10 ** 6)
    parser.add_argument('--final-step', type=int, default=10 ** 7)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), args.outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logdir = os.path.join(os.path.dirname(__file__), 'logs/' + args.logdir)

    env = gym.make(args.env)

    actions = get_action_space(args.env)
    n_actions = len(actions)

    model = make_cnn(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[512]
    )
    replay_buffer = ReplayBuffer(10 ** 5)
    explorer = LinearDecayExplorer(
        final_exploration_step=args.final_exploration_frames
    )

    sess = tf.Session()
    sess.__enter__()

    agent = Agent(
        model,
        n_actions,
        replay_buffer,
        explorer,
        learning_starts=10000
    )

    initialize()

    saver = tf.train.Saver()
    if args.load is not None:
        saver.restore(sess, args.load)

    train_writer = tf.summary.FileWriter(logdir, sess.graph)
    logger = TfBoardLogger(train_writer)
    logger.register('reward', dtype=tf.int32)
    end_episode = lambda r, t, e: logger.plot('reward', r, t)

    def preprocess(state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (84, 84))
        return state

    def after_action(state, reward, global_step, local_step):
        if global_step % 10 ** 6 == 0:
            path = os.path.join(outdir, '{}/model.ckpt'.format(global_step))
            saver.save(sess, path)

    trainer = Trainer(
        env=env,
        agent=agent,
        render=args.render,
        state_shape=[84, 84],
        state_window=4,
        final_step=args.final_step,
        preprocess=preprocess,
        after_action=after_action,
        end_episode=end_episode,
        training=not args.demo
    )
    trainer.start()

if __name__ == '__main__':
    main()
