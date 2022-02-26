data_loaders_names = {
            'Brightness': 'brightness',
            'Contrast': 'contrast',
            'Defocus Blur': 'defocus_blur',
            'Elastic Transform': 'elastic_transform',
            'Fog': 'fog',
            'Frost': 'frost',
            'Gaussian Noise': 'gaussian_noise',
            'Glass Blur': 'glass_blur',
            'Impulse Noise': 'impulse_noise',
            'JPEG Compression': 'jpeg_compression',
            'Motion Blur': 'motion_blur',
            'Pixelate': 'pixelate',
            'Shot Noise': 'shot_noise',
            'Snow': 'snow',
            'Zoom Blur': 'zoom_blur'
        }

def get_ce_alexnet():
    """Returns Corruption Error values for AlexNet"""

    ce_alexnet = dict()
    ce_alexnet['Gaussian Noise'] = 0.886428
    ce_alexnet['Shot Noise'] = 0.894468
    ce_alexnet['Impulse Noise'] = 0.922640
    ce_alexnet['Defocus Blur'] = 0.819880
    ce_alexnet['Glass Blur'] = 0.826268
    ce_alexnet['Motion Blur'] = 0.785948
    ce_alexnet['Zoom Blur'] = 0.798360
    ce_alexnet['Snow'] = 0.866816
    ce_alexnet['Frost'] = 0.826572
    ce_alexnet['Fog'] = 0.819324
    ce_alexnet['Brightness'] = 0.564592
    ce_alexnet['Contrast'] = 0.853204
    ce_alexnet['Elastic Transform'] = 0.646056
    ce_alexnet['Pixelate'] = 0.717840
    ce_alexnet['JPEG Compression'] = 0.606500

    return ce_alexnet

def get_mce_from_accuracy(accuracy, error_alexnet):
    """Computes mean Corruption Error from accuracy"""
    error = 100. - accuracy
    ce = error / (error_alexnet * 100.)

    return ce