dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt

# Note: --is-semisup use additional labelled data for CIFAR-10 released by Carmon et al. Do not use this flag with SVHN. 

pretrain_prune_finetune() {
    # Order: exp_name ($1), arch ($2), trainer ($3), val_method ($4), gpu ($5), k ($6), pruning_epochs ($7)

    # pre-training
    python train.py --exp-name $1 --arch $2 --exp-mode pretrain --configs configs/configs_crown-ibp.yml \
    --trainer $3 --val_method $4 --gpu $5 --k 1.0 --save-dense --dataset SVHN --batch-size 128 --epochs 200 --schedule_length 120;

    pruning
    python train.py --exp-name $1 --arch $2 --exp-mode prune --configs configs/configs_crown-ibp.yml \
    --trainer $3 --val_method $4 --gpu $5 --k $6 --save-dense --scaled-score-init \
    --source-net ./trained_models/$1/pretrain/latest_exp/checkpoint/checkpoint.pth.tar --epochs $7 \
    --schedule_length 1 --lr 0.00001 --dataset SVHN --batch-size 128;

    finetuning
    python train.py --exp-name $1 --arch $2 --exp-mode finetune --configs configs/configs_crown-ibp.yml \
    --trainer $3 --val_method $4 --gpu $5 --k $6 --save-dense \
    --source-net ./trained_models/$1/prune/latest_exp/checkpoint/checkpoint.pth.tar --lr 0.00001 \
    --schedule_length 1 --dataset SVHN --batch-size 128;

    weight base pruning
    python train.py --exp-name $1"_weight_based_pruning" --arch $2 --exp-mode finetune --configs configs/configs_crown-ibp.yml \
    --trainer $3 --val_method $4 --gpu $5 --k $6 --save-dense --scaled-score-init \
    --source-net ./trained_models/$1/pretrain/latest_exp/checkpoint/checkpoint.pth.tar --lr 0.0005\
    --schedule_length 1 --dataset SVHN --batch-size 128;
}


arch="cifar_model_large"

(
    pretrain_prune_finetune  "svhn_large_model-trainer_crown-ibp-k_0.1-prunepochs_20"  $arch  "crown-ibp"   "ibp"  "0"   0.1  20 &
    pretrain_prune_finetune  "svhn_large_model-trainer_crown-ibp-k_0.05-prunepochs_20"  $arch  "crown-ibp"   "ibp"  "1"   0.05  20 &
    pretrain_prune_finetune  "svhn_large_model-trainer_crown-ibp-k_0.01-prunepochs_20"  $arch  "crown-ibp"   "ibp"  "2"   0.01  20 ;
); 


arch="cifar_model"

(
    pretrain_prune_finetune  "svhn_model-trainer_crown-ibp_new-k_0.1-prunepochs_20"  $arch  "crown-ibp"   "ibp"  "0"   0.1  20 &
    pretrain_prune_finetune  "svhn_model-trainer_crown-ibp_new-k_0.05-prunepochs_20"  $arch  "crown-ibp"   "ibp"  "1"   0.05  20 &
    pretrain_prune_finetune  "svhn_model-trainer_crown-ibp_new-k_0.01-prunepochs_20"  $arch  "crown-ibp"   "ibp"  "2"   0.01  20 ;
); 