import network
import build_graph
import util
import tensorflow as tf


class Agent(object):
    def __init__(self, q_func, num_actions, lr=2.5e-4, train_freq=4,
            learning_starts=10000, gamma=0.99, target_network_update_freq=10000):
        act, train, update_target, debug = build_graph.build_train(
            q_func=q_func,
            num_actions=
        )
