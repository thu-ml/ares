import time

import torch
import torch.nn as nn
import torchvision

from utils.logging import AverageMeter, ProgressMeter
from utils.eval import accuracy

# TODO: support sm_loader when len(sm_loader.dataset) < len(train_loader.dataset)
def train(
    model, device, train_loader, sm_loader, criterion, optimizer, epoch, args, writer
):
    print(" ->->->->->->->->->-> One epoch with Natural training <-<-<-<-<-<-<-<-<-<-")

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    dataloader = train_loader if sm_loader is None else zip(train_loader, sm_loader)

    for i, data in enumerate(dataloader):
        if sm_loader:
            images, target = (
                torch.cat([d[0] for d in data], 0).to(device),
                torch.cat([d[1] for d in data], 0).to(device),
            )
        else:
            images, target = data[0].to(device), data[1].to(device)

        # basic properties of training
        if i == 0:
            print(
                images.shape,
                target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [{}, {}]".format(
                    torch.min(images).data.cpu().numpy(),
                    torch.max(images).data.cpu().numpy(),
                )
            )

        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            progress.write_to_tensorboard(
                writer, "train", epoch * len(train_loader) + i
            )

        # write a sample of training images to tensorboard (helpful for debugging)
        if i == 0:
            writer.add_image(
                "training-images",
                torchvision.utils.make_grid(images[0 : len(images) // 4]),
            )

