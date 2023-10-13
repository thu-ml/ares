import torch
from transformers import ViTModel
from .Base import BaseFeatureExtractor
from torchvision import transforms

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


class VisionTransformerFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(VisionTransformerFeatureExtractor, self).__init__()
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.normalizer = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
            ]
        )

    def forward(self, x):
        x = torch.clamp(x, min=0, max=1)
        # inputs = self.processor(images=x, return_tensors="pt")
        inputs = dict(pixel_values=self.normalizer(x))
        # print(inputs['pixel_values'].shape)
        inputs["pixel_values"] = inputs["pixel_values"].cuda()
        outputs = self.model(**inputs)
        pooled_output = outputs.pooler_output
        # print(f'Vit {pooled_output.shape}')
        return pooled_output
