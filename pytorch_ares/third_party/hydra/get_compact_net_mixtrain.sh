dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt

# Note: --is-semisup use additional labelled data for CIFAR-10 released by Carmon et al. Do not use this flag with SVHN. 

pretrain_prune_finetune() {
    # Order: exp_name ($1), arch ($2), trainer ($3), val_method ($4), gpu ($5), k ($6), pruning_epochs ($7), mixtraink($8)

    # pre-training
    python train.py --exp-name $1 --arch $2 --exp-mode pretrain --configs configs/configs_mixtrain.yml \
    --trainer $3 --val_method $4 --gpu $5 --k 1.0 --save-dense --dataset SVHN --schedule_length 15 \
    --mixtraink $8 --batch-size 50;

    # pruning
    python train.py --exp-name $1 --arch $2 --exp-mode prune --configs configs/configs_mixtrain.yml \
    --trainer $3 --val_method $4 --gpu $5 --k $6 --save-dense --scaled-score-init \
    --source-net ./trained_models/$1/pretrain/latest_exp/checkpoint/checkpoint.pth.tar --epochs $7 \
    --schedule_length 0 --lr 0.00001 --dataset SVHN  --mixtraink $8 --batch-size 50;

    # finetuning
    python train.py --exp-name $1 --arch $2 --exp-mode finetune --configs configs/configs_mixtrain.yml \
    --trainer $3 --val_method $4 --gpu $5 --k $6 --save-dense \
    --source-net ./trained_models/$1/prune/latest_exp/checkpoint/checkpoint.pth.tar --lr 0.0005 \
    --schedule_length 0 --dataset SVHN  --mixtraink $8 --batch-size 50;

    # weight base pruning
    python train.py --exp-name $1"_weight_based_pruning" --arch $2 --exp-mode finetune --configs configs/configs_mixtrain.yml \
    --trainer $3 --val_method $4 --gpu $5 --k $6 --save-dense --scaled-score-init \
    --source-net ./trained_models/$1/pretrain/latest_exp/checkpoint/checkpoint.pth.tar --lr 0.0005 \
    --schedule_length 0 --dataset SVHN --mixtraink $8 --batch-size 50;
}

arch="cifar_model_large"

(
    pretrain_prune_finetune  "svhn_model_large-trainer_mixtraink1-k_0.1-prunepochs_20"  $arch  "mixtrain"   "mixtrain"  "0"   0.1  20 1 &
    pretrain_prune_finetune  "svhn_model_large-trainer_mixtraink1-k_0.05-prunepochs_20"  $arch  "mixtrain"   "mixtrain"  "1"   0.05  20 1 &
    pretrain_prune_finetune  "svhn_model_large-trainer_mixtraink1-k_0.01-prunepochs_20"  $arch  "mixtrain"   "mixtrain"  "2"   0.01  20 1 ;
); 


arch="cifar_model"

(
    pretrain_prune_finetune  "svhn_model-trainer_mixtraink5-k_0.1-prunepochs_20"  $arch  "mixtrain"   "mixtrain"  "0"   0.1  20 5 &
    pretrain_prune_finetune  "svhn_model-trainer_mixtraink5-k_0.05-prunepochs_20"  $arch  "mixtrain"   "mixtrain"  "1"   0.05  20 5 &
    pretrain_prune_finetune  "svhn_model-trainer_mixtraink5-k_0.01-prunepochs_20"  $arch "mixtrain"   "mixtrain"  "2"   0.01  20 5 ;
); 