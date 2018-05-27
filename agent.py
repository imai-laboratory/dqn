import network
import build_graph
import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self,
                q_func,
                actions,
                state_shape,
                replay_buffer,
                exploration,
                constants,
                abam,
                phi=lambda s: s,
                batch_size=32,
                train_freq=4,
                learning_starts=1e4,
                target_network_update_freq=1e4):
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.actions = actions
        self.learning_starts = learning_starts
        self.target_network_update_freq = target_network_update_freq
        self.exploration = exploration
        self.abam = abam
        self.replay_buffer = replay_buffer
        self.phi = phi
        self.gamma = constants.GAMMA

        if constants.OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(constants.LR)
        else:
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=constants.LR,
                momentum=constants.MOMENTUM,
                epsilon=constants.EPSILON
            )

        self._act,\
        self._train,\
        self._update_target,\
        self._q_values = build_graph.build_train(
            q_func=q_func,
            num_actions=len(actions),
            state_shape=state_shape,
            optimizer=optimizer,
            constants=constants,
            gamma=constants.GAMMA,
            grad_norm_clipping=constants.GRAD_CLIPPING
        )

        self.last_obs = None
        self.last_action = 0
        self.t = 0
        self.count = 0
        self.local_t = 0
        self.last_value = 0

    def act(self, obs, reward, training):
        # transpose state shape to WHC
        obs = self.phi(obs)
        # take the best action
        q_values = self._q_values([obs])[0]
        td = (reward + self.gamma * q_values - self.last_value) ** 2
        if self.abam is not None:
            self.abam.accumulate(0, q_values)
            evidence = self.abam.evidences[0]
            error = td[np.argmax(evidence)]
            if np.max(evidence) > self.abam.threshold:
                action = np.argmax(evidence)
                #self.abam.flush()
                self.count += 1
            else:
                action = 0
        else:
            action = np.argmax(q_values)

        self.local_t += 1
        value = q_values[action]

        # epsilon greedy exploration
        action = self.exploration.select_action(
            self.t, action, len(self.actions))

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
                    obs_t,
                    actions,
                    rewards,
                    obs_tp1,
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
        self.last_value = value
        return self.actions[action]

    def stop_episode(self, obs, reward, training=True):
        if training:
            # transpose state shape to WHC
            obs = self.phi(obs)
            self.replay_buffer.append(
                obs_t=self.last_obs,
                action=self.last_action,
                reward=reward,
                obs_tp1=obs,
                done=True
            )
        self.last_obs = None
        self.last_action = 0
        if self.abam is not None:
            self.abam.flush()
        self.local_t = 0
        self.count = 0
        self.last_value = 0
