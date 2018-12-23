import argparse
import cv2
import gym
import copy
import os
import box_constants
import atari_constants
import numpy as np
import tensorflow as tf

from rlsaber.log import TfBoardLogger, JsonLogger, dump_constants
from rlsaber.explorer import LinearDecayExplorer, ConstantExplorer
from rlsaber.replay_buffer import ReplayBuffer
from rlsaber.trainer import Trainer, Evaluator, Recorder
from rlsaber.env import EnvWrapper
from actions import get_action_space
from network import make_cnn
from agent import Agent
from datetime import datetime
from env import Env

def main():
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongDeterministic-v4')
    parser.add_argument('--outdir', type=str, default=date)
    parser.add_argument('--logdir', type=str, default=date)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), 'results/' + args.logdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logdir = os.path.join(os.path.dirname(__file__), 'logs/' + args.logdir)

    env = Env()

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

    # optimizer
    if constants.OPTIMIZER == 'adam':
        optimizer = tf.train.AdamOptimizer(constants.LR)
    else:
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=constants.LR, momentum=constants.MOMENTUM,
            epsilon=constants.EPSILON)

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
        optimizer,
        gamma=constants.GAMMA,
        grad_norm_clipping=constants.GRAD_CLIPPING,
        phi=phi,
        learning_starts=constants.LEARNING_START_STEP,
        batch_size=constants.BATCH_SIZE,
        train_freq=constants.UPDATE_INTERVAL,
        target_network_update_freq=constants.TARGET_UPDATE_INTERVAL
    )

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    if args.load is not None:
        saver.restore(sess, args.load)

    train_writer = tf.summary.FileWriter(logdir, sess.graph)
    tflogger = TfBoardLogger(train_writer)
    tflogger.register('reward', dtype=tf.float32)
    tflogger.register('eval_reward', dtype=tf.float32)
    jsonloggers = {}
    for env_name in constants.ENV_LIST:
        jsonloggers[env_name] = JsonLogger(
            os.path.join(outdir, env_name + '_reward.json'))

    # callback on the end of episode
    def end_episode(reward, step, episode):
        tflogger.plot('reward', reward, step)
        env_name = env.env.current_env_name()
        step = env.env.get_local_step(env_name)
        jsonloggers[env_name].plot(reward=reward, step=step)

    def after_action(state, reward, global_step, local_step):
        if global_step > 0 and global_step % constants.MODEL_SAVE_INTERVAL == 0:
            path = os.path.join(outdir, 'model.ckpt')
            saver.save(sess, path, global_step=global_step)

    evaluator = Evaluator(
        env=copy.deepcopy(env),
        state_shape=state_shape[:-1],
        state_window=constants.STATE_WINDOW,
        eval_episodes=constants.EVAL_EPISODES,
        recorder=Recorder(outdir) if args.record else None,
        record_episodes=constants.RECORD_EPISODES
    )
    should_eval = lambda step, episode: step > 0 and step % constants.EVAL_INTERVAL == 0
    end_eval = lambda s, e, r: tflogger.plot('eval_reward', np.mean(r), s)

    trainer = Trainer(
        env=env,
        agent=agent,
        render=args.render,
        state_shape=state_shape[:-1],
        state_window=constants.STATE_WINDOW,
        final_step=constants.FINAL_STEP,
        after_action=after_action,
        end_episode=end_episode,
        training=not args.demo,
        evaluator=evaluator,
        should_eval=should_eval,
        end_eval=end_eval
    )
    trainer.start()

if __name__ == '__main__':
    main()
