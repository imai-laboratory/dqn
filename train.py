import argparse
import cv2
import gym
import copy
import os
import box_constants
import atari_constants
import numpy as np
import tensorflow as tf

from lightsaber.tensorflow.util import initialize
from lightsaber.tensorflow.log import TfBoardLogger, JsonLogger, dump_constants
from lightsaber.rl.explorer import LinearDecayExplorer, ConstantExplorer
from lightsaber.rl.replay_buffer import ReplayBuffer
from lightsaber.rl.trainer import Trainer
from lightsaber.rl.env_wrapper import EnvWrapper
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
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), 'results/' + args.outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logdir = os.path.join(os.path.dirname(__file__), 'logs/' + args.logdir)

    env = gym.make(args.env)

    # box environment
    if len(env.observation_space.shape) == 1:
        constants = box_constants
        actions = range(env.action_space.n)
        state_shape = [env.observation_space.shape[0], constants.STATE_WINDOW]
        state_preprocess = lambda state: state
        # (window_size, dim) -> (dim, window_size)
        phi = lambda state: np.transpose(state, [1, 0])
    # atari environment
    else:
        constants = atari_constants
        actions = get_action_space(args.env)
        state_shape = [84, 84, constants.STATE_WINDOW]
        def state_preprocess(state):
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            state = cv2.resize(state, (84, 84))
            return np.array(state, dtype=np.float32) / 255.0
        # (window_size, H, W) -> (H, W, window_size)
        phi = lambda state: np.transpose(state, [1, 2, 0])

    # save constant variables
    dump_constants(constants, os.path.join(outdir, 'constants.json'))

    # exploration
    if constants.EXPLORATION_TYPE == 'linear':
        duration = constants.EXPLORATION_DURATION
        explorer = LinearDecayExplorer(final_exploration_step=duration)
    else:
        explorer = ConstantExplorer(constants.EXPLORATION_EPSILON)

    # wrap gym environment
    env = EnvWrapper(
        env,
        s_preprocess=state_preprocess,
        r_preprocess=lambda r: np.clip(r, -1, 1)
    )

    replay_buffer = ReplayBuffer(constants.REPLAY_BUFFER_SIZE)

    sess = tf.Session()
    sess.__enter__()

    model = make_cnn(convs=constants.CONVS, hiddens=constants.FCS)

    agent = Agent(
        model,
        actions,
        state_shape,
        replay_buffer,
        explorer,
        constants,
        phi=phi,
        learning_starts=constants.LEARNING_START_STEP,
        batch_size=constants.BATCH_SIZE,
        train_freq=constants.UPDATE_INTERVAL,
        target_network_update_freq=constants.TARGET_UPDATE_INTERVAL
    )

    initialize()

    saver = tf.train.Saver()
    if args.load is not None:
        saver.restore(sess, args.load)

    train_writer = tf.summary.FileWriter(logdir, sess.graph)
    tflogger = TfBoardLogger(train_writer)
    tflogger.register('reward', dtype=tf.int32)
    jsonlogger = JsonLogger(os.path.join(outdir, 'reward.json'))

    # callback on the end of episode
    def end_episode(reward, step, episode):
        tflogger.plot('reward', reward, step)
        jsonlogger.plot(reward=reward, step=step, episode=episode)

    def after_action(state, reward, global_step, local_step):
        if global_step > 0 and global_step % constants.MODEL_SAVE_INTERVAL == 0:
            path = os.path.join(outdir, '/model.ckpt')
            saver.save(sess, path, global_step=global_step)

    trainer = Trainer(
        env=env,
        agent=agent,
        render=args.render,
        state_shape=state_shape[:-1],
        state_window=constants.STATE_WINDOW,
        final_step=constants.FINAL_STEP,
        after_action=after_action,
        end_episode=end_episode,
        training=not args.demo
    )
    trainer.start()

if __name__ == '__main__':
    main()
