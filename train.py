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
from abam import Abam


def main():
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongDeterministic-v4')
    parser.add_argument('--log', type=str, default=date)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--discount', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--abam', action='store_true')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), 'results/' + args.log)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logdir = os.path.join(os.path.dirname(__file__), 'logs/' + args.log)

    env = gym.make(args.env)

    #recorder = Recorder('.')

    # box environment
    if len(env.observation_space.shape) == 1:
        constants = box_constants
        actions = range(env.action_space.n)
        state_shape = [env.observation_space.shape[0], constants.STATE_WINDOW]
        state_preprocess = lambda state: state
        # (window_size, dim) -> (dim, window_size)
        def phi(state):
            noise = np.random.normal(0, args.sigma, state_shape[:-1])
            noise[0] *= 2.4
            noise[1] *= 2.0
            noise[2] *= 0.4
            noise[3] *= 3.5
            return np.transpose(state + noise, [1, 0])
    # atari environment
    else:
        constants = atari_constants
        actions = get_action_space(args.env)
        state_shape = [84, 84, constants.STATE_WINDOW]
        def state_preprocess(state):
            #noise = np.array(255.0 * ((np.asarray(state, dtype=np.float32) / 255.0) + np.random.normal(0, args.sigma, (210, 160, 3))), dtype=np.uint8)
            #recorder.append(noise)
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            state = cv2.resize(state, (210, 160))
            state = cv2.resize(state, (84, 110))
            state = state[18:102, :]
            #cv2.imshow('preview', np.array(255.0 * ((np.asarray(state, dtype=np.float32) / 255.0) + np.random.normal(0, args.sigma, (84, 84))), dtype=np.uint8))
            #if cv2.waitKey(10) < 0:
            #    pass
            return np.array(state, dtype=np.float32) / 255.0
        # (window_size, H, W) -> (H, W, window_size)
        phi = lambda state: np.transpose(state, [1, 2, 0]) + np.random.normal(0, args.sigma, state_shape)
 
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

    abam = Abam(1, len(actions), args.threshold, args.discount) if args.abam else None
    agent = Agent(
        model,
        actions,
        state_shape,
        replay_buffer,
        explorer,
        constants,
        abam,
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
    jsonlogger = JsonLogger(os.path.join(outdir, 'reward.json'))
    resultlogger = JsonLogger(os.path.join(outdir, 'result.json'))
    rewards = []
    percentages = []

    # callback on the end of episode
    def end_episode(reward, step, episode):
        tflogger.plot('reward', reward, step)
        percentage = float(agent.count) / agent.local_t
        jsonlogger.plot(reward=reward, step=step, episode=episode, percentage=percentage)
        rewards.append(reward)
        percentages.append(percentage)
        print(percentage)
        if episode == 30:
            resultlogger.plot(
                reward=np.mean(rewards),
                percentage=np.mean(percentages),
                threshold=args.threshold,
                discount=args.discount
            )
        #recorder.save_mp4('pong_noise_without_abam.mp4')

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
