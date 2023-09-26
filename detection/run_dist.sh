CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port 21245 --nproc_per_node 2 run.py \
--cfg configs/global_demo.py