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

### cautions
- Use `***Deterministic-v4` instead of `***-v0` for environemts because in default, environments sample {2, 3, 4} frames uniformaly. This affects negatively performance because of non-deterministic frame skipping.

### TODO
- Deep Learning framework will be changed from Chainer to Sony's NNabla.
- Chainerrl will be removed because of its unflexiblity,
