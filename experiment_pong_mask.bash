python train.py --env PongDeterministic-v4 --load pretrained_pong/model.ckpt-10000000 --demo --noise mask --mask-size 3 --log without_abam

python train.py --env PongDeterministic-v4 --load pretrained_pong/model.ckpt-10000000 --demo --discount 0.3 --threshold 0.34 --noise mask --mask-size 3 --log with_abam_discount_03_threshold_035 --abam
python train.py --env PongDeterministic-v4 --load pretrained_pong/model.ckpt-10000000 --demo --discount 0.3 --threshold 0.35 --noise mask --mask-size 3 --log with_abam_discount_03_threshold_036 --abam
python train.py --env PongDeterministic-v4 --load pretrained_pong/model.ckpt-10000000 --demo --discount 0.3 --threshold 0.36 --noise mask --mask-size 3 --log with_abam_discount_03_threshold_037 --abam

python train.py --env PongDeterministic-v4 --load pretrained_pong/model.ckpt-10000000 --demo --discount 0.4 --threshold 0.34 --noise mask --mask-size 3 --log with_abam_discount_04_threshold_035 --abam
python train.py --env PongDeterministic-v4 --load pretrained_pong/model.ckpt-10000000 --demo --discount 0.4 --threshold 0.35 --noise mask --mask-size 3 --log with_abam_discount_04_threshold_036 --abam
python train.py --env PongDeterministic-v4 --load pretrained_pong/model.ckpt-10000000 --demo --discount 0.4 --threshold 0.36 --noise mask --mask-size 3 --log with_abam_discount_04_threshold_037 --abam

python train.py --env PongDeterministic-v4 --load pretrained_pong/model.ckpt-10000000 --demo --discount 0.2 --threshold 0.34 --noise mask --mask-size 3 --log with_abam_discount_02_threshold_035 --abam
python train.py --env PongDeterministic-v4 --load pretrained_pong/model.ckpt-10000000 --demo --discount 0.2 --threshold 0.35 --noise mask --mask-size 3 --log with_abam_discount_02_threshold_036 --abam
python train.py --env PongDeterministic-v4 --load pretrained_pong/model.ckpt-10000000 --demo --discount 0.2 --threshold 0.36 --noise mask --mask-size 3 --log with_abam_discount_02_threshold_037 --abam

python train.py --env PongDeterministic-v4 --load pretrained_pong/model.ckpt-10000000 --demo --discount 0.1 --threshold 0.33 --noise mask --mask-size 3 --log with_abam_discount_01_threshold_034 --abam
python train.py --env PongDeterministic-v4 --load pretrained_pong/model.ckpt-10000000 --demo --discount 0.1 --threshold 0.34 --noise mask --mask-size 3 --log with_abam_discount_01_threshold_035 --abam
python train.py --env PongDeterministic-v4 --load pretrained_pong/model.ckpt-10000000 --demo --discount 0.1 --threshold 0.35 --noise mask --mask-size 3 --log with_abam_discount_01_threshold_036 --abam
python train.py --env PongDeterministic-v4 --load pretrained_pong/model.ckpt-10000000 --demo --discount 0.1 --threshold 0.36 --noise mask --mask-size 3 --log with_abam_discount_01_threshold_037 --abam
