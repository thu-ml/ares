from timm.models.xcit import _create_xcit
from timm.models.registry import register_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj.0.0', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'xcit_medium_12_p16_224': _cfg(url=''),
    'xcit_large_12_p16_224': _cfg(url='')
}

@register_model
def xcit_medium_12_p16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=512,
                        depth=12,
                        num_heads=8,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = _create_xcit('xcit_medium_12_p16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def xcit_large_12_p16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=16,
                        eta=1.0,
                        tokens_norm=True,
                        **kwargs)
    model = _create_xcit('xcit_large_12_p16_224', pretrained=pretrained, **model_kwargs)
    return model