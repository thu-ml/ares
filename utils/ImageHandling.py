import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from torch import Tensor
import cv2
from typing import List, Callable
from tqdm import tqdm


@torch.no_grad()
def show_image(x: Tensor) -> Image.Image:
    if len(x.shape) == 4:
        x = x.squeeze(0)
    x = x.permute(1, 2, 0) * 255
    x = x.cpu().numpy()
    x = Image.fromarray(np.uint8(x))
    return x


@torch.no_grad()
def save_image(x: Tensor, path="./0.png") -> Image.Image:
    if len(x.shape) == 4:
        x = x.squeeze(0)
    x = x.permute(1, 2, 0) * 255
    x = x.cpu().numpy()
    if x.shape[2] == 1:
        cv2.imwrite(path, x.squeeze())
        return x
    x = Image.fromarray(np.uint8(x))
    x.save(path)
    return x


@torch.no_grad()
def save_list_images(xs: List, folder_path="./debug/", begin_id: int = 0):
    for i, x in enumerate(xs, begin_id):
        save_image(x, os.path.join(folder_path, f"{i}.png"))


@torch.no_grad()
def save_multi_images(xs: Tensor, folder_path="./debug/", begin_id: int = 0):
    assert len(xs.shape) == 4, "Input should be (N, C, H, D) Tensor"
    save_list_images(xs.split(1, dim=0), folder_path=folder_path, begin_id=begin_id)


@torch.no_grad()
def scale_and_show_tensor(x: Tensor):
    x = x.cpu()
    x += torch.min(x)
    x /= torch.max(x)
    return show_image(x)


def get_image(path: str = "image.jpg") -> Tensor:
    image = Image.open(path)
    image = image.convert("RGB")
    transform = transforms.ToTensor()
    return transform(image)


def get_list_image(path: str) -> List[Tensor]:
    result = []
    images = os.listdir(path)
    for image in images:
        result.append(get_image(os.path.join(path, image)))
    return result


def concatenate_image(
    img_path: str = "./generated",
    padding=1,
    img_shape=(32, 32, 3),
    row=10,
    col=10,
    save_path="./concatenated.png",
    sort_key=None,
) -> Image.Image:
    imgs = os.listdir(img_path)
    imgs.sort(key=sort_key)
    assert len(imgs) >= row * col, "images should be enough for demonstration"
    alls = []
    for img in imgs:
        img = Image.open(os.path.join(img_path, img))
        x = np.array(img)
        x = np.pad(x, ((padding, padding), (padding, padding), (0, 0)))
        alls.append(x)
    alls = alls[: row * col]
    x = np.stack(alls)
    x = x.reshape((row, col, img_shape[0] + padding * 2, img_shape[1] + padding * 2, img_shape[2]))
    x = torch.from_numpy(x)
    x = (
        x.permute(0, 2, 1, 3, 4)
        .reshape(
            row * (img_shape[0] + padding * 2),
            col * (img_shape[1] + padding * 2),
            img_shape[2],
        )
        .numpy()
    )
    x = Image.fromarray(x)
    x.save(save_path)
    return x


def total_variation(x):
    adv_patch = x
    if len(x.shape) == 3:
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
    elif len(x.shape) == 4:
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, :, 1:] - adv_patch[:, :, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(tvcomp1)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, :, 1:, :] - adv_patch[:, :, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(tvcomp2)
        tv = tvcomp1 + tvcomp2
    else:
        raise ValueError
    return tv / torch.numel(adv_patch)


@torch.no_grad()
def synthesize_images_and_show(
    generator: Callable,
    path: str = "./temp/",
    total_generation_iter: int = 1,
    reserve_temporary_directory=False,
    **kwargs,
) -> Image.Image:
    to_img = transforms.ToPILImage()
    if not os.path.exists(path):
        os.mkdir(path)

    # extract images from generator
    imgs = []
    for _ in tqdm(range(total_generation_iter)):
        imgs += list(torch.split(generator(), split_size_or_sections=1, dim=0))
    for i, img in enumerate(imgs):
        img = to_img(img.squeeze())
        img.save(os.path.join(path, f"{i}.png"))

    width = int(len(imgs) ** 0.5)
    result = concatenate_image(img_path=path, row=width, col=width, **kwargs)
    if not reserve_temporary_directory:
        os.rmdir(path)
    return result
