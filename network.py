import tensorflow as tf
import tensorflow.contrib.layers as layers

def _make_cnn(convs, hiddens, inpt, num_actions, scope, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope('convnet'):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(
                    out,
                    num_outputs=num_outputs,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding='VALID',
                    activation_fn=tf.nn.relu
                )
        conv_out = layers.flatten(out)
        with tf.variable_scope('action_value'):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(
                    action_out, num_outputs=hidden, activation_fn=tf.nn.relu)

            # final output layer
            action_scores = layers.fully_connected(
                action_out, num_outputs=num_actions, activation_fn=None)
        return action_scores

def make_cnn(convs, hiddens):
    return lambda *args, **kwargs: _make_cnn(convs, hiddens, *args, **kwargs)
