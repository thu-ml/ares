dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt

# Note: --is-semisup use additional labelled data for CIFAR-10 released by Carmon et al. Do not use this flag with SVHN. 

pretrain_prune_finetune_semisup() {
    # Order: exp_name ($1), arch ($2), trainer ($3), val_method ($4), gpu ($5), k ($6), pruning_epochs ($7)

    # pre-training
    python train.py --is-semisup --exp-name $1 --arch $2 --exp-mode pretrain --configs configs/configs.yml \
    --trainer $3 --val_method $4 --gpu $5 --k 1.0 --save-dense ;

    # pruning
    python train.py --is-semisup --exp-name $1 --arch $2 --exp-mode prune --configs configs/configs.yml \
    --trainer $3 --val_method $4 --gpu $5 --k $6 --save-dense --scaled-score-init \
    --source-net ./trained_models/$1/pretrain/latest_exp/checkpoint/checkpoint.pth.tar --epochs $7;

    # finetuning
    python train.py --is-semisup --exp-name $1 --arch $2 --exp-mode finetune --configs configs/configs.yml \
    --trainer $3 --val_method $4 --gpu $5 --k $6 --save-dense \
    --source-net ./trained_models/$1/prune/latest_exp/checkpoint/checkpoint.pth.tar --lr 0.01 ;

    # weight base pruning
    python train.py --is-semisup --exp-name $1"_weight_based_pruning" --arch $2 --exp-mode finetune --configs configs/configs.yml \
    --trainer $3 --val_method $4 --gpu $5 --k $6 --save-dense --scaled-score-init \
    --source-net ./trained_models/$1/pretrain/latest_exp/checkpoint/checkpoint.pth.tar --lr 0.01 ;

}


arch="wrn_28_4"


# Iterative adv training
(   
    pretrain_prune_finetune_semisup  "semisup-$arch-trainer_adv-k_0.1-prunepochs_20"  $arch  "adv"   "adv"  "0"   0.1 20 &
    pretrain_prune_finetune_semisup  "semisup-$arch-trainer_adv-k_0.05-prunepochs_20"  $arch  "adv"   "adv"  "1"   0.05  20 &
    pretrain_prune_finetune_semisup  "semisup-$arch-trainer_adv-k_0.01-prunepochs_20"  $arch  "adv"   "adv"  "2"   0.01  20 ;
);

#Natural training 
(   
    pretrain_prune_finetune_semisup  "semisup-$arch-trainer_base-k_0.1-prunepochs_20"  $arch "base"   "base"  "0"   0.1  20 &
    pretrain_prune_finetune_semisup  "semisup-$arch-trainer_base-k_0.05-prunepochs_20"  $arch "base"   "base"  "1"   0.05  20 &
    pretrain_prune_finetune_semisup  "semisup-$arch-trainer_base-k_0.01-prunepochs_20"  $arch  "base"   "base"  "2"   0.01  20 ;
); 

