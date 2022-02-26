CHECKPOINT=~/robustness/imagenet_l2_eps_30/109_checkpoint.pt
IMAGENET=~/imagenet
OUT=~/robustness/imagenet_l2_eps_30_resume

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m robustness.main --dataset imagenet \
    --data $IMAGENET --adv-train 1 --arch resnet50 \
    --eps 3.0 --attack-lr 0.5 \
    --attack-steps 7 --constraint 2 \
    --resume $CHECKPOINT \
    --step-lr 75 --epochs 250 \
    --save-ckpt-iters 1 --log-iters 4 \
    --out-dir $OUT
