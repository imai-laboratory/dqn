import tenforflow as tf

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
        

