import numpy as np


class Abam:
    def __init__(self, num_modules, num_actions, threshold, discount_factor):
        self.num_modules = num_modules
        self.num_actions = num_actions
        self.evidences = np.zeros((num_modules, num_actions), dtype=np.float32)
        self.threshold = threshold
        self.discount_factor = discount_factor

    def accumulate(self, index, q_values):
        probs = np.exp(q_values) / np.sum(np.exp(q_values + 1e-10))
        max_index = np.argmax(probs)
        self.evidences[index] *= self.discount_factor
        self.evidences[index][max_index] += probs[max_index]

    def select_action(self, td):
        selected_module = 0
        selected_action = np.argmax(self.evidences[0])
        for i, evidence in enumerate(self.evidences[1:]):
            index = i + 1
            max_index = np.argmax(evidence)
            error = td[max_index]
            print(error)
            if self.threshold + error <= evidence[max_index]:
                selected_module = index
                selected_action = max_index
        return selected_module, selected_action

    def flush(self):
        self.evidences = np.zeros(
            (self.num_modules, self.num_actions), dtype=np.float32)
