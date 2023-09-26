# attack_mode: str. 'global' or 'patch'
attack_mode = 'global'
# object_vanish_only: bool. If True, the attack goal is to make objects vanish only.
# Otherwise, the attack goal is to make objects vanish, misclassify objects and generate false objects in the background.
object_vanish_only = False
# amp: bool. Whether to use 'Automatic Mixed Precision'
amp = False
# detector: a dict with following attributes:
# cfg_file: str. Path to your mmdet-style detector config file.
# weight_file: str. Path to your detector weight file.
detector = dict(cfg_file='path_to_your_detector_config_file',
                weight_file='path_to_your_detector_weight_file')
# batch_size: int. Batch size for each GPU
batch_size = 4
# attack_method: a dict with following attributes:
## type: str. Name of the method used to attack detectors. 'fgsm', 'bim', 'mim', 'di_fgsm', 'si_ni_fgsm', 'vmi_fgsm' and 'pgd' are supported now.
## kwargs: dict. Corresponding kwargs. See ares.attack for details.
attack_method = dict(type='pgd',
                     kwargs=dict(eps=3/255,norm='inf'))
# loss_fn: a dict with following attributes:
# excluded_losses: list. It specifies detector losses not be used to attack.
loss_fn = dict(excluded_losses=['loss_bbox', 'loss_iou'])
# adv_image: a dict with following attributes:
# save: bool. Whether to save adversarial images. Suggested value is False to avoid frequently saving amount of images at each evaluation
# save_folder: str.
# with_bboxes: bool. Whether to save adversarial images with bboxes drawn.
adv_image = dict(save=False, save_folder='adv_images', with_bboxes=True)
# clean_image: a dict with followinng attributes:
# save: bool. Whether to save the original clean images.
# save_folder: str.
# with_bboxes: bool. Whether to save adversarial images with bboxes drawn.
clean_image = dict(save=False, save_folder='clean_images', with_bboxes=True)

