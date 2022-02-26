python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 \
    main.py --configs=$1