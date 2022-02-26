import time

import torch
import torch.nn as nn
import torchvision

from utils.logging import AverageMeter, ProgressMeter
from utils.eval import accuracy
from utils.adv import trades_loss

import numpy as np

from symbolic_interval.symbolic_network import sym_interval_analyze, naive_interval_analyze, mix_interval_analyze


def set_epsilon(args, epoch):
    if epoch<args.schedule_length:
        epsilon = epoch*(args.epsilon - args.starting_epsilon)/\
                args.schedule_length + args.starting_epsilon
    else:
        epsilon = args.epsilon
    return epsilon

def set_interval_weight(args, epoch):
    interval_weight = args.interval_weight * (2.5 **\
                ((max((epoch-args.schedule_length), 0) // 5)))
    interval_weight = min(interval_weight, 50)
    return interval_weight


# TODO: add adversarial accuracy.
def train(
    model, device, train_loader, sm_loader, criterion, optimizer, epoch, args, writer
):
    epsilon = set_epsilon(args, epoch)
    k = args.mixtraink
    alpha = 0.8
    iw = set_interval_weight(args, epoch)

    print(
        " ->->->->->->->->->-> One epoch with MixTrain{} (SYM {:.3f})"
        " <-<-<-<-<-<-<-<-<-<-".format(k, epsilon)
    )

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    sym_losses = AverageMeter("Sym_Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    sym1 = AverageMeter("Sym1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, sym_losses, top1, sym1],
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

        # basic properties of training data
        if i == 0:
            print(
                images.shape,
                target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(f"Training images range: {[torch.min(images), torch.max(images)]}")

        output = model(images)
        ce = nn.CrossEntropyLoss()(output, target)

        if(np.random.uniform()<=alpha):
            r = np.random.randint(low=0, high=images.shape[0], size=k)
            rce, rerr = sym_interval_analyze(model, epsilon, 
                            images[r], target[r],
                            use_cuda=torch.cuda.is_available(),
                            parallel=False)

            #print("sym:", rce.item(), ce.item())
            loss = iw * rce + ce
            sym_losses.update(rce.item(), k)
            sym1.update((1-rerr)*100., images.size(0))
        else:
            loss = ce

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        losses.update(ce.item(), images.size(0))
        
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
