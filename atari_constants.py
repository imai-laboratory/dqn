REPLAY_BUFFER_SIZE = 10 ** 5
BATCH_SIZE = 32
LEARNING_START_STEP = 10 ** 4
FINAL_STEP = 10 ** 7
GAMMA = 0.99
UPDATE_INTERVAL = 4
TARGET_UPDATE_INTERVAL = 10 ** 4
STATE_WINDOW = 4
EXPLORATION_TYPE = 'linear'
EXPLORATION_EPSILON = 0.1
EXPLORATION_DURATION = 10 ** 6
CONVS = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
FCS = [512]

LR = 2.5e-4
OPTIMIZER = 'rmsprop'
MOMENTUM = 0.95
EPSILON = 1e-2
GRAD_CLIPPING = 10.0

DEVICE = '/gpu:0'
MODEL_SAVE_INTERVAL = 10 ** 6
EVAL_INTERVAL = 10 ** 5
EVAL_EPISODES = 10
RECORD_EPISODES = 3
