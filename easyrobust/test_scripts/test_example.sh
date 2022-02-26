python robustness_validation.py \
--model=resnet50 --interpolation=3 \
--imagenet_val_path=/path/to/ILSVRC/Data/CLS-LOC/val \
--imagenet_a_path=/path/to/Data/imagenet-a \
--imagenet_r_path=/path/to/Data/imagenet-r \
--imagenet_sketch_path=/path/to/Data/imagenet-sketch/imagenet-sketch/sketch \
--imagenet_v2_path=/path/to/Data/imagenetv2 \
--stylized_imagenet_path=/path/to/Data/imagenet_style/val \
--imagenet_c_path=/path/to/Data/imagenet-c \
--objectnet_path=/path/to/Data/ObjectNet/images