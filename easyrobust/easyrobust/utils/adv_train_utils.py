import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from apex import amp
    has_apex = False
except ImportError:
    has_apex = False

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss

class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.
        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set
        Args:
            ch.tensor x : the input to project back into the feasible set.
        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).
        Parameters:
            g (ch.tensor): the raw gradient
        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x

# L-infinity threat model
class LinfStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = torch.clamp(diff, -self.eps, self.eps)
        return torch.clamp(diff + self.orig_input, 0, 1)

    def step(self, x, g):
        """
        """
        step = torch.sign(g) * self.step_size
        return x + step

    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (torch.rand_like(x) - 0.5) * self.eps
        return torch.clamp(new_x, 0, 1)

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def replace_best(loss, bloss, x, bx):
    if bloss is None:
        bx = x.clone().detach()
        bloss = loss.clone().detach()
    else:
        replace = bloss < loss
        bx[replace] = x[replace].clone().detach()
        bloss[replace] = loss[replace]

    return bloss, bx

def replace_best_reverse(loss, bloss, x, bx):
    if bloss is None:
        bx = x.clone().detach()
        bloss = loss.clone().detach()
    else:
        replace = bloss > loss
        bx[replace] = x[replace].clone().detach()
        bloss[replace] = loss[replace]

    return bloss, bx

def adv_generator_random_target(images, target, model, eps, attack_steps, attack_lr, random_start, gpu, use_best=True):
    # generate adversarial examples
    orig_input = images.detach().cuda(gpu, non_blocking=True)
    step = LinfStep(eps=eps, orig_input=orig_input, step_size=attack_lr)

    attack_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    hard_label = target.topk(1)[1].squeeze()
    while True:
        attack_label = torch.randint(0,1000,(target.size(0),)).cuda(gpu, non_blocking=True)
        if torch.sum(attack_label == hard_label).item() == 0:
            break

    best_loss = None
    best_x = None
    if random_start:
        images = step.random_perturb(images) 
    for _ in range(attack_steps):
        images = images.clone().detach().requires_grad_(True)
        adv_losses = -1 * attack_criterion(model(images), attack_label.long())

        if has_apex:
            with amp.scale_loss(torch.mean(adv_losses), []) as sl:
                sl.backward()
        else:
            torch.mean(adv_losses).backward()
        grad = images.grad.detach()

        with torch.no_grad():
            varlist = [adv_losses, best_loss, images, best_x]
            best_loss, best_x = replace_best_reverse(*varlist) if use_best else (adv_losses, images)

            images = step.step(images, grad)
            images = step.project(images)

    adv_losses = attack_criterion(model(images), attack_label.long())
    varlist = [adv_losses, best_loss, images, best_x]
    best_loss, best_x = replace_best_reverse(*varlist) if use_best else (adv_losses, images)
    
    return best_x

def adv_generator(images, target, model, eps, attack_steps, attack_lr, random_start, gpu, attack_criterion='regular', use_best=True):
    # generate adversarial examples
    prev_training = bool(model.training)
    model.eval()
    orig_input = images.detach().cuda(gpu, non_blocking=True)
    step = LinfStep(eps=eps, orig_input=orig_input, step_size=attack_lr)

    if attack_criterion == 'regular':
        attack_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    elif attack_criterion == 'smooth':
        attack_criterion = LabelSmoothingCrossEntropy()
    elif attack_criterion == 'mixup':
        attack_criterion = SoftTargetCrossEntropy()
    best_loss = None
    best_x = None
    if random_start:
        images = step.random_perturb(images) 
    for _ in range(attack_steps):
        images = images.clone().detach().requires_grad_(True)
        adv_losses = attack_criterion(model(images), target)

        if has_apex:
            with amp.scale_loss(torch.mean(adv_losses), []) as sl:
                sl.backward()
        else:
            torch.mean(adv_losses).backward()
        grad = images.grad.detach()

        with torch.no_grad():
            varlist = [adv_losses, best_loss, images, best_x]
            best_loss, best_x = replace_best(*varlist) if use_best else (adv_losses, images)

            images = step.step(images, grad)
            images = step.project(images)

    adv_losses = attack_criterion(model(images), target)
    varlist = [adv_losses, best_loss, images, best_x]
    best_loss, best_x = replace_best(*varlist) if use_best else (adv_losses, images)
    if prev_training:
        model.train()
    
    return best_x