import torch
from transformers import (
    Blip2VisionModel,
    Blip2VisionConfig,
    Blip2Processor,
    Blip2Model,
    BlipImageProcessor,
)
from torchvision import transforms
from .Base import BaseFeatureExtractor

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


class BlipFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(BlipFeatureExtractor, self).__init__()
        # self.model = Blip2VisionModel(Blip2VisionConfig())
        self.normalizer = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        )
        self.processor = BlipImageProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.device = torch.device("cuda")
        self.eval().requires_grad_(False)

    def forward(self, x):
        x = torch.clamp(x, min=0, max=1)
        # inputs = self.processor(images=show_image(x), return_tensors="pt").to(self.device)
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"].cuda()
        # print(torch.mean((inputs.pixel_values - self.normalizer(x)) ** 2),
        #       inputs.pixel_values.shape, self.normalizer(x).shape)
        # outputs = self.model.get_image_features(**inputs, output_attentions=True)
        outputs = self.model.get_image_features(**inputs)
        # print(inputs.pixel_values, self.normalizer(x))
        # assert False
        # pooler_output = outputs.pooler_output
        # attention = torch.stack(list(outputs.attentions))
        # print(f'Blip {outputs.pooler_output.shape}')
        return outputs.pooler_output
