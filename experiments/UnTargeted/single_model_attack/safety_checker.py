from transformers import AutoFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from PIL import Image
import numpy as np
from utils.ImageHandling import get_image, save_image


safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("html/NSFW_replace.jpg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception as e:
        return x


def check_safety(x_image: str = "./resources/hbk/knife.jpg"):
    safety_checker_input = safety_feature_extractor(Image.open(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(
        images=get_image(x_image).unsqueeze(0).numpy(), clip_input=safety_checker_input.pixel_values
    )
    print(x_checked_image.shape, len(has_nsfw_concept))
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


print(check_safety())
