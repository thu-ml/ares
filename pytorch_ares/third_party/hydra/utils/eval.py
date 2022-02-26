import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F

from utils.logging import AverageMeter, ProgressMeter
from utils.adv import pgd_whitebox, fgsm
from symbolic_interval.symbolic_network import (
    sym_interval_analyze,
    naive_interval_analyze,
    mix_interval_analyze,
)
from crown.bound_layers import (
    BoundSequential,
    BoundLinear,
    BoundConv2d,
    BoundDataParallel,
    Flatten,
)

from scipy.stats import norm
import numpy as np
import time


def get_output_for_batch(model, img, temp=1):
    """
        model(x) is expected to return logits (instead of softmax probas)
    """
    with torch.no_grad():
        out = nn.Softmax(dim=-1)(model(img) / temp)
        p, index = torch.max(out, dim=-1)
    return p.data.cpu().numpy(), index.data.cpu().numpy()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def base(model, device, val_loader, criterion, args, writer, epoch=0):
    """
        Evaluating on unmodified validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

            if writer:
                progress.write_to_tensorboard(
                    writer, "test", epoch * len(val_loader) + i
                )

            # write a sample of test images to tensorboard (helpful for debugging)
            if i == 0 and writer:
                writer.add_image(
                    "test-images",
                    torchvision.utils.make_grid(images[0 : len(images) // 4]),
                )
        progress.display(i)  # print final results

    return top1.avg, top5.avg


def adv(model, device, val_loader, criterion, args, writer, epoch=0):
    """
        Evaluate on adversarial validation set inputs.
    """

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    adv_losses = AverageMeter("Adv_Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    adv_top1 = AverageMeter("Adv-Acc_1", ":6.2f")
    adv_top5 = AverageMeter("Adv-Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, adv_losses, top1, top5, adv_top1, adv_top5],
        prefix="Test: ",
    )

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # clean images
            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # adversarial images
            images = pgd_whitebox(
                model,
                images,
                target,
                device,
                args.epsilon,
                args.num_steps,
                args.step_size,
                args.clip_min,
                args.clip_max,
                is_random=not args.const_init,
            )

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            adv_losses.update(loss.item(), images.size(0))
            adv_top1.update(acc1[0], images.size(0))
            adv_top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

            if writer:
                progress.write_to_tensorboard(
                    writer, "test", epoch * len(val_loader) + i
                )

            # write a sample of test images to tensorboard (helpful for debugging)
            if i == 0 and writer:
                writer.add_image(
                    "Adv-test-images",
                    torchvision.utils.make_grid(images[0 : len(images) // 4]),
                )
        progress.display(i)  # print final results

    return adv_top1.avg, adv_top5.avg


def mixtrain(model, device, val_loader, criterion, args, writer, epoch=0):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    sym_losses = AverageMeter("Sym_Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    sym_top1 = AverageMeter("Sym-Acc_1", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, sym_losses, top1, top5, sym_top1],
        prefix="Test: ",
    )

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # clean images
            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            rce_avg = 0
            rerr_avg = 0
            for r in range(images.shape[0]):

                rce, rerr = sym_interval_analyze(
                    model,
                    args.epsilon,
                    images[r : r + 1],
                    target[r : r + 1],
                    use_cuda=torch.cuda.is_available(),
                    parallel=False,
                )
                rce_avg = rce_avg + rce.item()
                rerr_avg = rerr_avg + rerr

            rce_avg = rce_avg / float(images.shape[0])
            rerr_avg = rerr_avg / float(images.shape[0])

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            sym_losses.update(rce_avg, images.size(0))
            sym_top1.update((1 - rerr_avg) * 100.0, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

            if writer:
                progress.write_to_tensorboard(
                    writer, "test", epoch * len(val_loader) + i
                )

            # write a sample of test images to tensorboard (helpful for debugging)
            if i == 0 and writer:
                writer.add_image(
                    "Adv-test-images",
                    torchvision.utils.make_grid(images[0 : len(images) // 4]),
                )
        progress.display(i)  # print final results

    return sym_top1.avg, sym_top1.avg


def ibp(model, device, val_loader, criterion, args, writer, epoch=0):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ibp_losses = AverageMeter("IBP_Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    ibp_top1 = AverageMeter("IBP-Acc_1", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, ibp_losses, top1, top5, ibp_top1],
        prefix="Test: ",
    )

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # clean images

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            rce, rerr = naive_interval_analyze(
                model,
                args.epsilon,
                images,
                target,
                use_cuda=torch.cuda.is_available(),
                parallel=False,
            )

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            ibp_losses.update(rce.item(), images.size(0))
            ibp_top1.update((1 - rerr) * 100.0, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

            if writer:
                progress.write_to_tensorboard(
                    writer, "test", epoch * len(val_loader) + i
                )

            # write a sample of test images to tensorboard (helpful for debugging)
            if i == 0 and writer:
                writer.add_image(
                    "Adv-test-images",
                    torchvision.utils.make_grid(images[0 : len(images) // 4]),
                )
        progress.display(i)  # print final results

    return ibp_top1.avg, ibp_top1.avg


def smooth(model, device, val_loader, criterion, args, writer, epoch=0):
    """
        Evaluating on unmodified validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    rad = AverageMeter("rad", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, top1, top5, rad], prefix="Smooth (eval): "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # Defult: evaluate on 10 random samples of additive gaussian noise.
            output = []
            for _ in range(10):
                # add noise
                if args.dataset == "imagenet":
                    std = (
                        torch.tensor([0.229, 0.224, 0.225])
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    ).to(device)
                    noise = (torch.randn_like(images) / std).to(device) * args.noise_std
                else:
                    noise = torch.randn_like(images).to(device) * args.noise_std

                output.append(F.softmax(model(images + noise), -1))

            output = torch.sum(torch.stack(output), axis=0)

            p_max, _ = output.max(dim=-1)
            radii = (args.noise_std + 1e-16) * norm.ppf(p_max.data.cpu().numpy())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            rad.update(np.mean(radii))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

            if writer:
                progress.write_to_tensorboard(
                    writer, "test", epoch * len(val_loader) + i
                )

            # write a sample of test images to tensorboard (helpful for debugging)
            if i == 0 and writer:
                writer.add_image(
                    "Adv-test-images",
                    torchvision.utils.make_grid(images[0 : len(images) // 4]),
                )

        progress.display(i)  # print final results

    return top1.avg, rad.avg


def freeadv(model, device, val_loader, criterion, args, writer, epoch=0):

    assert (
        not args.normalize
    ), "Explicit normalization is done in the training loop, Dataset should have [0, 1] dynamic range."

    # Mean/Std for normalization
    mean = torch.Tensor(np.array(args.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, args.image_dim, args.image_dim).to(device)
    std = torch.Tensor(np.array(args.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, args.image_dim, args.image_dim).to(device)

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: ",
    )

    eps = args.epsilon
    K = args.num_steps
    step = args.step_size
    model.eval()
    end = time.time()
    print(" PGD eps: {}, num-steps: {}, step-size: {} ".format(eps, K, step))
    for i, (input, target) in enumerate(val_loader):

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        orig_input = input.clone()
        randn = torch.FloatTensor(input.size()).uniform_(-eps, eps).to(device)
        input += randn
        input.clamp_(0, 1.0)
        for _ in range(K):
            invar = Variable(input, requires_grad=True)
            in1 = invar - mean
            in1.div_(std)
            output = model(in1)
            ascend_loss = criterion(output, target)
            ascend_grad = torch.autograd.grad(ascend_loss, invar)[0]
            pert = fgsm(ascend_grad, step)
            # Apply purturbation
            input += pert.data
            input = torch.max(orig_input - eps, input)
            input = torch.min(orig_input + eps, input)
            input.clamp_(0, 1.0)

        input.sub_(mean).div_(std)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

        if writer:
            progress.write_to_tensorboard(writer, "test", epoch * len(val_loader) + i)

        # write a sample of test images to tensorboard (helpful for debugging)
        if i == 0 and writer:
            writer.add_image(
                "Adv-test-images",
                torchvision.utils.make_grid(input[0 : len(input) // 4]),
            )

    progress.display(i)  # print final results

    return top1.avg, top5.avg
