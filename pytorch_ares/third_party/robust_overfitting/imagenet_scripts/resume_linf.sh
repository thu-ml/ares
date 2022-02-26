CHECKPOINT=~/robustness/imagenet_linf_eps_4/checkpoint.pt.best
IMAGENET=~/imagenet
OUT=~/robustness/imagenet_linf_eps_4_resume/

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m robustness.main --dataset imagenet \
    --data $IMAGENET --adv-train 1 --arch resnet50 \
    --eps 0.0156862745 --attack-lr 0.004 \
    --attack-steps 5 --constraint inf \
    --resume $CHECKPOINT \
    --lr 0.001 --epochs 150 \
    --save-ckpt-iters 1 --log-iters 1 \
    --out-dir $OUT
