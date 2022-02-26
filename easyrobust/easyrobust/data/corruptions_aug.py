'''

Benchmarking Neural Network Robustness to Common Corruptions and Perturbations (ICLR2019)
Paper link: https://arxiv.org/abs/1903.12261

Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming
Paper link: https://arxiv.org/abs/1907.07484

Modified by Xiaofeng Mao 
2021.8.30
'''

from imagecorruptions import corrupt
import random

class Corruption_Gaussian_Noise(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='gaussian_noise', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Shot_Noise(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='shot_noise', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Impulse_Noise(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='impulse_noise', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Defocus_Blur(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='defocus_blur', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Glass_Blur(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='glass_blur', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Motion_Blur(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='motion_blur', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Zoom_Blur(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='zoom_blur', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Snow(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='snow', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Frost(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='frost', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Fog(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='fog', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Brightness(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='brightness', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Contrast(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='contrast', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Elastic_Transform(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='elastic_transform', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Pixelate(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='pixelate', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Jpeg_Compression(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='jpeg_compression', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Speckle_Noise(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='speckle_noise', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Gaussian_Blur(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='gaussian_blur', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Spatter(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='spatter', severity=random.choice([1,2,3,4,5]))
        return corrupted_image

class Corruption_Saturate(object):

    def __call__(self, img):
        corrupted_image = corrupt(img, corruption_name='saturate', severity=random.choice([1,2,3,4,5]))
        return corrupted_image
