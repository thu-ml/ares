_base_ = './patch/base.py'
batch_size = 2
detector = dict(
    cfg_file='work_dirs/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py',
    weight_file='work_dirs/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth'
    )
lr_scheduler = dict(type='MultiStepLR',
                    kwargs=dict(milestones=[40, 70])
                    )

patch = dict(size=200,
        resume_path='work_dirs/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco/best-patches.pth')
adv_image = dict(save=False, save_folder='adv_images', with_bboxes=True)
#attacked_classes = ['person']