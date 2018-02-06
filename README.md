## DQN
DQN implementation for checking baselines.

## requirements
- Python3

## dependencies
- tensorflow
- gym[atari]
- opencv-python
- git+https://github.com/imai-laboratory/lightsaber

## usage
### training
```
$ python train.py --render --final-steps 10000000
```

### playing
```
$ python train.py --render --load {path of models} --demo
```

### cautions
- Use `***Deterministic-v4` instead of `***-v0` for environemts because in default, environments sample {2, 3, 4} frames uniformaly. This affects negatively performance because of non-deterministic frame skipping.

### implementation
This baseline is inspired by following projects.

- [OpenAI Baselines](https://github.com/openai/baselines)
- [ChainerRL](https://github.com/chainer/chainerrl)
