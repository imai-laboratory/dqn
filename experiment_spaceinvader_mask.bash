python train.py --env SpaceInvadersDeterministic-v4 --load pretrained_spaceinvader/model.ckpt-40000000 --demo --noise mask --mask-size 6 --log without_abam

python train.py --env SpaceInvadersDeterministic-v4 --load pretrained_spaceinvader/model.ckpt-40000000 --demo --discount 0.3 --threshold 0.26 --noise mask --mask-size 6 --log with_abam_discount_03_threshold_035 --abam
python train.py --env SpaceInvadersDeterministic-v4 --load pretrained_spaceinvader/model.ckpt-40000000 --demo --discount 0.3 --threshold 0.27 --noise mask --mask-size 6 --log with_abam_discount_03_threshold_036 --abam
python train.py --env SpaceInvadersDeterministic-v4 --load pretrained_spaceinvader/model.ckpt-40000000 --demo --discount 0.3 --threshold 0.28 --noise mask --mask-size 6 --log with_abam_discount_03_threshold_037 --abam

python train.py --env SpaceInvadersDeterministic-v4 --load pretrained_spaceinvader/model.ckpt-40000000 --demo --discount 0.4 --threshold 0.26 --noise mask --mask-size 6 --log with_abam_discount_04_threshold_035 --abam
python train.py --env SpaceInvadersDeterministic-v4 --load pretrained_spaceinvader/model.ckpt-40000000 --demo --discount 0.4 --threshold 0.27 --noise mask --mask-size 6 --log with_abam_discount_04_threshold_036 --abam
python train.py --env SpaceInvadersDeterministic-v4 --load pretrained_spaceinvader/model.ckpt-40000000 --demo --discount 0.4 --threshold 0.28 --noise mask --mask-size 6 --log with_abam_discount_04_threshold_037 --abam

python train.py --env SpaceInvadersDeterministic-v4 --load pretrained_spaceinvader/model.ckpt-40000000 --demo --discount 0.2 --threshold 0.26 --noise mask --mask-size 6 --log with_abam_discount_02_threshold_035 --abam
python train.py --env SpaceInvadersDeterministic-v4 --load pretrained_spaceinvader/model.ckpt-40000000 --demo --discount 0.2 --threshold 0.27 --noise mask --mask-size 6 --log with_abam_discount_02_threshold_036 --abam
python train.py --env SpaceInvadersDeterministic-v4 --load pretrained_spaceinvader/model.ckpt-40000000 --demo --discount 0.2 --threshold 0.28 --noise mask --mask-size 6 --log with_abam_discount_02_threshold_037 --abam

python train.py --env SpaceInvadersDeterministic-v4 --load pretrained_spaceinvader/model.ckpt-40000000 --demo --discount 0.1 --threshold 0.26 --noise mask --mask-size 6 --log with_abam_discount_01_threshold_034 --abam
python train.py --env SpaceInvadersDeterministic-v4 --load pretrained_spaceinvader/model.ckpt-40000000 --demo --discount 0.1 --threshold 0.27 --noise mask --mask-size 6 --log with_abam_discount_01_threshold_035 --abam
python train.py --env SpaceInvadersDeterministic-v4 --load pretrained_spaceinvader/model.ckpt-40000000 --demo --discount 0.1 --threshold 0.28 --noise mask --mask-size 6 --log with_abam_discount_01_threshold_036 --abam
python train.py --env SpaceInvadersDeterministic-v4 --load pretrained_spaceinvader/model.ckpt-40000000 --demo --discount 0.1 --threshold 0.255 --noise mask --mask-size 6 --log with_abam_discount_01_threshold_037 --abam
