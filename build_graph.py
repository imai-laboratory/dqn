import tensorflow as tf


def huber_loss(loss, delta=1.0):
    return tf.where(
        tf.abs(loss) < delta,
        tf.square(loss) * 0.5,
        delta * (tf.abs(loss) - 0.5 * delta)
    )

def build_train(q_func,
                num_actions,
                state_shape,
                optimizer,
                constants,
                batch_size=32,
                grad_norm_clipping=10.0,
                gamma=1.0,
                scope='deepq',
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        obs_t_ph = tf.placeholder(tf.float32, [None] + state_shape, name='obs_t')
        act_t_ph = tf.placeholder(tf.int32, [None], name='action')
        rew_t_ph = tf.placeholder(tf.float32, [None], name='reward')
        obs_tp1_ph = tf.placeholder(tf.float32, [None] + state_shape, name='obs_tp1')
        done_mask_ph = tf.placeholder(tf.float32, [None], name='done')

        # q network
        q_t = q_func(obs_t_ph, num_actions, scope='q_func')
        q_func_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{}/q_func'.format(scope))

        # target q network
        q_tp1 = q_func(obs_tp1_ph, num_actions, scope='target_q_func')
        target_q_func_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{}/target_q_func'.format(scope))

        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)
        q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = tf.reduce_mean(huber_loss(td_error))

        # update parameters
        gradients = optimizer.compute_gradients(errors, var_list=q_func_vars)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
        optimize_expr = optimizer.apply_gradients(gradients)

        # update target network
        update_target_expr = []
        sorted_vars = sorted(q_func_vars, key=lambda v: v.name)
        sorted_target_vars = sorted(target_q_func_vars, key=lambda v: v.name)
        for var, var_target in zip(sorted_vars, sorted_target_vars):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # taking best action
        actions = tf.argmax(q_t, axis=1)
        def act(obs, sess=None):
            if sess is None:
                sess = tf.get_default_session()
            return sess.run(actions, feed_dict={obs_t_ph: obs})

        # update network
        def train(obs_t, act_t, rew_t, obs_tp1, done, sess=None):
            if sess is None:
                sess = tf.get_default_session()
            feed_dict = {
                obs_t_ph: obs_t,
                act_t_ph: act_t,
                rew_t_ph: rew_t,
                obs_tp1_ph: obs_tp1,
                done_mask_ph: done
            }
            td_error_val, _ = sess.run(
                [td_error, optimize_expr], feed_dict=feed_dict)
            return td_error_val

        # synchronize target network
        def update_target(sess=None):
            if sess is None:
                sess = tf.get_default_session()
            sess.run(update_target_expr)

        # estimate q value
        def q_values(obs, sess=None):
            if sess is None:
                sess = tf.get_default_session()
            return sess.run(q_t, feed_dict={obs_t_ph: obs})

        return act, train, update_target, q_values
