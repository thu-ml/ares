from ares.utils.registry import registry


@registry.register_model('bit_depth_reduction')
class BitDepthReduction(object):
    '''Bit depth reduction defense method.'''
    def __init__(self, device='cuda', compressed_bit=4):
        '''
        Args:
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            compressed_bit (int): The compressed bit.
        '''
        self.compressed_bit = compressed_bit
        self.device = device
    
    def __call__(self, images):
        '''The function to perform bit depth reduction on the input images.'''
        images = self.bit_depth_reduction(images)
        
        return images

    def bit_depth_reduction(self, xs):
        bits = 2 ** self.compressed_bit    #2**i
        xs_compress = (xs.detach() * bits).int()
        xs_255 = (xs_compress * (255 / bits))
        xs_compress = (xs_255 / 255).to(self.device)

        return xs_compress
