import tenforflow as tf
import util

def build_act(observations_ph, q_func, num_actions, scope='deepq', resue=None):
    with tf.variable_scope(scope, reuse=reuse):
        stochastic_ph = tf.placeholder(tf.bool, (), name='stochastic')
        update_eps_ph = tf.placeholder(tf.float32, (), name='update_eps')

        eps = tf.get_variable('eps', (), initializer=tf.constant_initializer(0))

        q_values = q_func(observations_ph.get(), num_actions, scope='q_func')
        deterministic_actions = tf.argmax(q_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        act = util.function(inputs=[observation_ph, stochastic_ph, update_eps_ph],
                            outputs=output_actions,
                            givens={update_eps_ph: -1.0, stochastic_ph: True},
                            updates=[update_eps_expr])

def build_train(observations_ph, q_func, num_actions, optimizer, batch_size=32,
                grad_norm_clipping=10.0, gamma=1.0, scope='deepq', reuse=None):
    act_f = build_act(observations_ph, q_func, num_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        obs_t_input = tf.placeholder(tf.float32, [batch_size, 4, 84, 84], name='obs_t')
        act_t_ph = tf.placeholder(tf.int32, [None], name='action')
        rew_t_ph = tf.placeholder(tf.float32, [None], name='reward')
        obs_tp1_input = tf.placeholder(tf.float32, [batch_size, 4, 84, 84], name='obs_tp1')
        done_mask_ph = tf.placeholder(tf.float32, [None], name='done')

        q_t = q_func(obs_t_input.get(), num_actions, scope='q_func', reuse=True)
        q_func_vars = util.scope_vars(util.absolute_scope_name('q_func'))

        q_tp1 = q_func(obs_t_input.get(), num_actions, scope='target_q_func', reuse=True)
        target_q_func_vars = util.scope_vars(util.absolute_scope_name('target_q_func'))

        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)
        q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        q_t_selected_target = rew_t + gamma * q_tp1_best_masked
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = util.huber_loss(td_error)

        gradients = optimizer.compute_gradients(errors, var_list=q_func_vars)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
        optimize_expr = optimizer.apply_gradients(gradients)

        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                    sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        train = util.function(
            inputs=[
                obs_t_input, act_t_ph, rew_t_ph, obs_t1_input, done_mask_ph
            ],
            outputs=td_error,
            update=[optimize_expr]
        )
        update_target = util.function([], [], updates=[update_target_expr])

        q_values = util.function([obs_t_input], q_t)

        return act_f, train, update_target, q_values
