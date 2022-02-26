source env_py27/bin/activate
cd AT_HE_imagenet

mkdir saved_images
mkdir saved_images/original_images
mkdir saved_images/original_images_HE
mkdir saved_images/original_images_pretrained
mkdir saved_images/adv_images
mkdir saved_images/adv_images_HE
mkdir saved_images/adv_images_pretrained
mkdir saved_images/scaled_perturbation
mkdir saved_images/scaled_perturbation_HE
mkdir saved_images/scaled_perturbation_pretrained


# ##### Old version commands #####
# # wide_resnet50_2; bs=512; gpu workers=8
# nohup python -u main_free.py /data/LargeData/ImageNet > LOGS/wide_resnet50_2_free_adv_step4_eps4_repeat4_bs512.out 2>&1 &
# nohup python -u main_free.py /data/LargeData/ImageNet --HE True > LOGS/wide_resnet50_2_free_adv_step4_eps4_repeat4_bs512_HE.out 2>&1 &  

# # wide_resnet50_2; bs=256; gpu workers=4
# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main_free.py /data/LargeData/ImageNet > LOGS/wide_resnet50_2_free_adv_step4_eps4_repeat4_bs256.out 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u main_free.py /data/LargeData/ImageNet --HE True > LOGS/wide_resnet50_2_free_adv_step4_eps4_repeat4_bs256_HE.out 2>&1 &  

# # resnet50; bs=256; gpu workers=4
# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main_free.py /data/LargeData/ImageNet > LOGS/resnet50_free_adv_step4_eps4_repeat4_bs256.out 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u main_free.py /data/LargeData/ImageNet --HE True > LOGS/resnet50_free_adv_step4_eps4_repeat4_bs256_HE.out 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u main_free.py /data/LargeData/ImageNet --HE True --s_HE 20.0 > LOGS/resnet50_free_adv_step4_eps4_repeat4_bs256_HE_s20p0.out 2>&1 &


# # resnet152; bs=256; gpu workers=4
# nohup python -u main_free.py /data/LargeData/ImageNet > LOGS/resnet152_free_adv_step4_eps4_repeat4_bs256.out 2>&1 &
# nohup python -u main_free.py /data/LargeData/ImageNet --HE True > LOGS/resnet152_free_adv_step4_eps4_repeat4_bs256_HE.out 2>&1 &

# # resnet152; bs=256; gpu workers=4
# nohup python -u main_free.py /data/LargeData/ImageNet > LOGS/wide_resnet101_2_free_adv_step4_eps4_repeat4_bs256.out 2>&1 &
# nohup python -u main_free.py /data/LargeData/ImageNet --HE True > LOGS/wide_resnet101_2_free_adv_step4_eps4_repeat4_bs256_HE.out 2>&1 &



# python -u main_free.py /data/LargeData/ImageNet --HE True



##### New version commands #####
# resnet50; bs=256; gpu workers=4
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main_free.py /data/LargeData/ImageNet --FN True --WN True --s_HE 10.0 --angular_margin_HE 0.0 > LOGS/resnet50_free_adv_step4_eps4_repeat4_bs256_FN_WN_s10p0_margin0p0.out 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u main_free.py /data/LargeData/ImageNet --FN True --s_HE 10.0 --angular_margin_HE 0.0 > LOGS/resnet50_free_adv_step4_eps4_repeat4_bs256_FN_s10p0_margin0p0.out 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u main_free.py /data/LargeData/ImageNet --WN True --s_HE 10.0 --angular_margin_HE 0.0 > LOGS/resnet50_free_adv_step4_eps4_repeat4_bs256_WN_s10p0_margin0p0.out 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main_free.py /data/LargeData/ImageNet --FN True --WN True --s_HE 10.0 --angular_margin_HE 0.2 > LOGS/resnet50_free_adv_step4_eps4_repeat4_bs256_FN_WN_s10p0_margin0p2.out 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u main_free.py /data/LargeData/ImageNet --FN True --s_HE 1.0 --angular_margin_HE 0.0 > LOGS/resnet50_free_adv_step4_eps4_repeat4_bs256_FN_s1p0_margin0p0.out 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main_free.py /data/LargeData/ImageNet --WN True --s_HE 1.0 --angular_margin_HE 0.0 > LOGS/resnet50_free_adv_step4_eps4_repeat4_bs256_WN_s1p0_margin0p0.out 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main_free.py /data/LargeData/ImageNet --FN True --WN True --s_HE 1.0 --angular_margin_HE 0.0 > LOGS/resnet50_free_adv_step4_eps4_repeat4_bs256_FN_WN_s1p0_margin0p0.out 2>&1 &



# print gradient norm for FreeAT on ImageNet
nohup python -u main_free.py /data/LargeData/ImageNet --print_gradients True > LOGS/resnet18_free_adv_step4_eps4_repeat4_bs256_print_gradients.out 2>&1 &
nohup python -u main_free.py /data/LargeData/ImageNet --FN True --WN True --s_HE 10.0 --angular_margin_HE 0.2 --print_gradients True > LOGS/resnet18_free_adv_step4_eps4_repeat4_bs256_FN_WN_s10p0_margin0p2_print_gradients.out 2>&1 &



# evaluation and save the image and perturbation
CUDA_VISIBLE_DEVICES=0 python evaluate.py /data/LargeData/ImageNet -e --pretrained
CUDA_VISIBLE_DEVICES=0 python evaluate.py /data/LargeData/ImageNet -e --resume /home/tianyu/AT_HE_imagenet/trained_models/resnet50_free_adv_step4_eps4_repeat4_bs256/model_best.pth.tar
CUDA_VISIBLE_DEVICES=0 python evaluate.py /data/LargeData/ImageNet -e --FN True --WN True --s_HE 10.0 --angular_margin_HE 0.2 --resume /home/tianyu/AT_HE_imagenet/trained_models/resnet50_free_adv_step4_eps4_repeat4_bs256_FN_WN_s10.0_margin0.2/model_best.pth.tar


# evaluation on Imaget-C
nohup python -u evaluate.py /home/tianyu/datasets/ImageNet-C --eva_on_imagenet_c --pretrained > ImageNet-C_resnet50_standard.out 2>&1 &
nohup python -u evaluate.py /home/tianyu/datasets/ImageNet-C --eva_on_imagenet_c --resume /home/tianyu/AT_HE_imagenet/trained_models/resnet50_free_adv_step4_eps4_repeat4_bs256/model_best.pth.tar > ImageNet-C_resnet50_free_adv_step4_eps4_repeat4_bs256.out 2>&1 &
nohup python -u evaluate.py /home/tianyu/datasets/ImageNet-C --eva_on_imagenet_c --FN True --WN True --s_HE 10.0 --angular_margin_HE 0.2 --resume /home/tianyu/AT_HE_imagenet/trained_models/resnet50_free_adv_step4_eps4_repeat4_bs256_FN_WN_s10.0_margin0.2/model_best.pth.tar > ImageNet-C_resnet50_free_adv_step4_eps4_repeat4_bs256_FN_WN_s10p0_margin0p2.out 2>&1 &


















