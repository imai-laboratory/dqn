import copy

import chainer
from chainer import cuda
import chainer.functions as F

from chainerrl import agent
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.replay_buffer import batch_experiences
from chainerrl.replay_buffer import ReplayUpdater


class DQN(agent.AttributeSavingMixin, agent.Agent):

    saved_attributes = ('model', 'target_model', 'optimizer')

    def __init__(self, q_function, optimizer, replay_buffer, gamma,
                 explorer, gpu=None, replay_start_size=50000,
                 minibatch_size=32, update_interval=1,
                 target_update_interval=10000,
                 phi=lambda x: x,
                 n_times_update=1, average_q_decay=0.999,
                 average_loss_decay=0.99,
                 batch_states=batch_states):
        self.model = q_function

        if gpu is not None and gpu >= 0:
            cuda.get_device(gpu).use()
            self.model.to_gpu(device=gpu)

        self.xp = self.model.xp
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.gamma = gamma
        self.explorer = explorer
        self.gpu = gpu
        self.target_update_interval = target_update_interval
        self.update_interval = update_interval
        self.phi = phi
        self.batch_states = batch_states
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=self.update,
            batchsize=minibatch_size,
            episodic_update=False,
            episodic_update_len=0,
            n_times_update=n_times_update,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
        )

        self.t = 0
        self.last_state = None
        self.last_action = 0
        self.target_model = None
        self.sync_target_network()

        self.average_q = 0
        self.average_q_decay = average_q_decay
        self.average_loss = 0
        self.average_loss_decay = average_loss_decay

    def sync_target_network(self):
        if self.target_model is None:
            self.target_model = copy.deepcopy(self.model)
            call_orig = self.target_model.__call__

            def call_test(self_, x):
                with chainer.using_config('train', False):
                    return call_orig(self_, x)

            self.target_model.__call__ = call_test
        else:
            synchronize_parameters(
                src=self.model,
                dst=self.target_model,
                method='hard',
                tau=1e-2)

    def update(self, experiences):
        exp_batch = batch_experiences(experiences, xp=self.xp, phi=self.phi,
                                      batch_states=self.batch_states)
        loss = self._compute_loss(exp_batch, self.gamma)

        self.average_loss *= self.average_loss_decay
        self.average_loss += (1 - self.average_loss_decay) * float(loss.data)

        self.model.cleargrads()
        loss.backward()
        self.optimizer.update()

    def input_initial_batch_to_target_model(self, batch):
        self.target_model(batch['state'])

    def _compute_target_values(self, exp_batch, gamma):
        target_next_qout = self.target_model(exp_batch['next_state'])
        next_q_max = target_next_qout.max

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']

        return batch_rewards + self.gamma * (1.0 - batch_terminal) * next_q_max

    def _compute_y_and_t(self, exp_batch, gamma):
        batch_size = exp_batch['reward'].shape[0]

        qout = self.model(exp_batch['state'])

        batch_actions = exp_batch['action']
        batch_q = F.reshape(qout.evaluate_actions(
            batch_actions), (batch_size, 1))

        with chainer.no_backprop_mode():
            batch_q_target = F.reshape(
                self._compute_target_values(exp_batch, gamma),
                (batch_size, 1))

        return batch_q, batch_q_target

    def _compute_loss(self, exp_batch, gamma):
        y, t = self._compute_y_and_t(exp_batch, gamma)
        y = F.reshape(y, (-1, 1))
        t = F.reshape(t, (-1, 1))
        return F.sum(F.huber_loss(y, t, delta=1.0))

    def compute_q_values(self, states):
        with chainer.using_config('train', False):
            if not states:
                return []
            batch_x = self.batch_states(states, self.xp, self.phi)
            q_values = list(cuda.to_cpu(
                self.model(batch_x).q_values))
            return q_values

    def _to_my_device(self, model):
        if self.gpu >= 0:
            model.to_gpu(self.gpu)
        else:
            model.to_cpu()

    def act(self, state):
        if self.t % self.update_interval == 0:
            with chainer.using_config('train', False):
                with chainer.no_backprop_mode():
                    action_value = self.model(
                        self.batch_states([state], self.xp, self.phi))
                    q = float(action_value.max.data)
                    action = cuda.to_cpu(action_value.greedy_actions.data)[0]
        else:
            action = self.last_action

        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q
        self.last_action = action

        return action

    def act_and_train(self, state, reward):
        if self.t % self.update_interval == 0:
            with chainer.using_config('train', False):
                with chainer.no_backprop_mode():
                    action_value = self.model(
                        self.batch_states([state], self.xp, self.phi))
                    q = float(action_value.max.data)
                    greedy_action = cuda.to_cpu(action_value.greedy_actions.data)[0]
                    self.average_q *= self.average_q_decay
                    self.average_q += (1 - self.average_q_decay) * q

            action = self.explorer.select_action(
                self.t, lambda: greedy_action, action_value=action_value)
        else:
            action = self.last_action

        if self.t % self.target_update_interval == 0:
            self.sync_target_network()

        if self.last_state is not None:
            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=state,
                next_action=action,
                is_state_terminal=False)

        self.replay_updater.update_if_necessary(self.t)

        self.last_state = state
        self.last_action = action
        self.t += 1

        return self.last_action

    def stop_episode_and_train(self, state, reward, done=False):
        self.replay_buffer.append(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=state,
            next_action=self.last_action,
            is_state_terminal=done)

        self.stop_episode()

    def stop_episode(self):
        self.last_state = None
        self.last_action = 0
        self.replay_buffer.stop_current_episode()

    def get_statistics(self):
        return [
            ('average_q', self.average_q),
            ('average_loss', self.average_loss),
        ]
