from lightsaber.rl.trainer import AgentInterface
import network
import build_graph
import lightsaber.tensorflow.util as util
import numpy as np
import tensorflow as tf


# another preprocess to store integer matrix into replay memory
def preprocess(state):
    state = np.array(state, dtype=np.float32)
    return state / 255.0

class Agent(AgentInterface):
    def __init__(self, q_func, actions, replay_buffer,
            exploration, lr=2.5e-4, batch_size=32, train_freq=4,
            learning_starts=1e4, gamma=0.99, target_network_update_freq=1e4):
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.actions = actions
        self.learning_starts = learning_starts
        self.gamma = gamma
        self.target_network_update_freq = target_network_update_freq
        self.last_obs = None
        self.t = 0
        self.exploration = exploration
        self.replay_buffer = replay_buffer

        act, train, update_target, q_values = build_graph.build_train(
            q_func=q_func,
            num_actions=len(actions),
            optimizer=tf.train.RMSPropOptimizer(
                learning_rate=lr,
                momentum=0.95,
                epsilon=1e-2
            ),
            gamma=gamma,
            grad_norm_clipping=10.0
        )
        self._act = act
        self._train = train
        self._update_target = update_target
        self._q_values = q_values

    def act(self, obs, reward, training):
        # transpose state shape to WHC
        obs = np.transpose(obs, [1, 2, 0])
        # take the best action
        action = self._act(preprocess(obs).reshape([1, 84, 84, 4]))[0]
        # epsilon greedy exploration
        action = self.exploration.select_action(
            self.t, 
            action,
            len(self.actions)
        )

        if training:
            if self.t % self.target_network_update_freq == 0:
                self._update_target()

            if self.t > self.learning_starts and self.t % self.train_freq == 0:
                obs_t,\
                actions,\
                rewards,\
                obs_tp1,\
                dones = self.replay_buffer.sample(self.batch_size)
                td_errors = self._train(
                    preprocess(obs_t),
                    actions,
                    rewards,
                    preprocess(obs_tp1),
                    dones
                )

            if self.last_obs is not None:
                self.replay_buffer.append(
                    obs_t=self.last_obs,
                    action=self.last_action,
                    reward=reward,
                    obs_tp1=obs,
                    done=False
                )

        self.t += 1
        self.last_obs = obs
        self.last_action = action
        return self.actions[action]

    def stop_episode(self, obs, reward, done=False, training=True):
        if training:
            # transpose state shape to WHC
            obs = np.transpose(obs, [1, 2, 0])
            self.replay_buffer.append(
                obs_t=self.last_obs,
                action=self.last_action,
                reward=reward,
                obs_tp1=obs,
                done=done
            )
        self.last_obs = None
        self.last_action = 0
