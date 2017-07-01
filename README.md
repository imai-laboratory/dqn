## DQN
DQN implementation for checking baselines.

## requirements
- Python3

## dependencies
- chainer==2.0.0
- gym[atari]
- chainerrl=0.2.0
- opencv-python

## usage
### training
```
$ python train.py --gpu {0 or -1} --render --final-steps 10000000
```

### playing
```
$ python play.py --gpu {0 or -1} --render --load {path of models}
```
