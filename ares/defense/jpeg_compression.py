import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
from ares.utils.registry import registry

_to_pil_image = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()


@registry.register_model('jpeg_compression')
class Jpeg_compression(object):
    '''JPEG compression defense method.'''
    def __init__(self, device='cuda', quality=75):
        '''
        Args:
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            quality (int): The compressed image quality.
        '''
        self.quality = quality
        self.device = device
    
    def __call__(self, images):
        '''The function to perform JPEG compression on the input images.'''
        images = self.jpegcompression(images)
        
        return images


    def jpegcompression(self, x):
        lst_img = []
        for img in x:
            img = _to_pil_image(img.detach().clone().cpu())
            virtualpath = BytesIO()
            img.save(virtualpath, 'JPEG', quality=self.quality)
            lst_img.append(_to_tensor(Image.open(virtualpath)))
        return x.new_tensor(torch.stack(lst_img))
