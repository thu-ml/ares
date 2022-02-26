import numpy as np
import pickle

import torch
import torchvision
import os

from utils.misc import CustomDatasetFromNumpy


def get_semisup_dataloader(args, transform):
    if args.semisup_data == "splitgan":
        print(f"Loading {args.semisup_data} generated data")
        img, label = (
            np.load(
                "/data/scsi/home/vvikash/research/mini_projects/trades_minimal/filter_gan_generate_images_c_99.npy"
            ),
            np.load(
                "/data/scsi/home/vvikash/research/mini_projects/trades_minimal/filter_gan_generate_labels_c_99.npy"
            ).astype(np.int64),
        )
    if args.semisup_data == "tinyimages":
        print(f"Loading {args.semisup_data} dataset")
        with open(
            os.path.join(args.data_dir, "tiny_images/ti_top_50000_pred_v3.1.pickle"),
            "rb",
        ) as f:
            data = pickle.load(f)
        img, label = data["data"], data["extrapolated_targets"]

    # select random subset
    index = np.random.permutation(np.arange(len(label)))[
        0 : int(args.semisup_fraction * len(label))
    ]

    sm_loader = torch.utils.data.DataLoader(
        CustomDatasetFromNumpy(img[index], label[index], transform),
        batch_size=args.batch_size,
        shuffle=True,
    )
    print(f"Semisup dataset: {len(sm_loader.dataset)} images.")
    return sm_loader
