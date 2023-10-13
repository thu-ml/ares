import torch
from transformers import CLIPVisionModel, AutoProcessor, CLIPModel
from .Base import BaseFeatureExtractor
from torchvision import transforms

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


class ClipFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(ClipFeatureExtractor, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.normalizer = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        )

    def forward(self, x):
        x = torch.clamp(x, min=0, max=1)
        # inputs = self.processor(images=x, return_tensors="pt")
        inputs = dict(pixel_values=self.normalizer(x))
        # print(inputs['pixel_values'].shape)
        inputs["pixel_values"] = inputs["pixel_values"].cuda()
        outputs = self.model.get_image_features(**inputs)
        pooled_output = outputs
        # print(f"Clip {pooled_output.shape}")
        return pooled_output
