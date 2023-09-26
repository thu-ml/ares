from contextlib import suppress
import torch
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


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
            orig_input (torch.Tensor): the original input
        '''

        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set

        Args:
            x (torch.Tensor): the input to project back into the feasible set.
        Returns:
            A torch.Tensor that is the input projected back into
            the feasible set, that is, .. math:: \min_{x' \in S} \|x' - x\|_2
        '''

        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p` norms).

        :param x (torch.Tensor): The input to project back into the feasible set.
        :param g (torch.Tensor): The raw gradient
        :return: The new input, a torch.Tensor for the next step.
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

    def random_uniform(self, x):
        noise=torch.rand_like(x)
        noise.uniform_(-self.eps, self.eps)

        return torch.clamp(x+noise, 0, 1)


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

def adv_generator(args, images, target, model, eps, attack_steps, attack_lr, random_start, attack_criterion='regular', use_best=True):
    # denorm images to 0-1
    std_tensor=torch.Tensor(args.std).cuda(non_blocking=True)[None, :, None, None]
    mean_tensor=torch.Tensor(args.mean).cuda(non_blocking=True)[None, :, None, None]
    images=images*std_tensor+mean_tensor

    # define perturbation range
    prev_training = bool(model.training)
    model.eval()
    orig_input = images.detach().cuda(non_blocking=True)
    step = LinfStep(eps=eps, orig_input=orig_input, step_size=attack_lr)

    # define loss function
    if attack_criterion == 'regular':
        attack_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    elif attack_criterion == 'smooth':
        attack_criterion = LabelSmoothingCrossEntropy()
    elif attack_criterion == 'mixup':
        attack_criterion = SoftTargetCrossEntropy()

    # amp_autocast
    amp_autocast=suppress
    if args.amp_version == 'native':
        amp_autocast = torch.cuda.amp.autocast

    # adv generation
    best_loss = None
    best_x = None
    if random_start:
        images = step.random_perturb(images)
    else:
        images = step.random_uniform(images)
    
    for _ in range(attack_steps):
        images = images.clone().detach().requires_grad_(True)

        # forward
        adv_losses=None
        with amp_autocast():
            adv_losses = attack_criterion(model((images-mean_tensor)/std_tensor), target)

        # backward
        if args.amp_version == 'apex':
            from apex import amp
            with amp.scale_loss(torch.mean(adv_losses), []) as sl:
                sl.backward()
        else:
            torch.mean(adv_losses).backward()

        # update gradient
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
    
    best_x=(best_x-mean_tensor)/std_tensor
    return best_x

class LinfStepEps(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """

    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = torch.minimum(torch.maximum(diff, -self.eps.repeat(x.size()[1],x.size()[2],x.size()[3],1).permute(3,0,1,2)), self.eps.repeat(x.size()[1],x.size()[2],x.size()[3],1).permute(3,0,1,2))
        return torch.clamp(diff + self.orig_input, 0, 1)

    def step(self, x, g):
        """
        """
        step = torch.sign(g) * self.step_size.repeat(x.size()[1],x.size()[2],x.size()[3],1).permute(3,0,1,2)
        return x + step

    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (torch.rand_like(x) - 0.5) * self.eps.repeat(x.size()[1],x.size()[2],x.size()[3],1).permute(3,0,1,2)
        return torch.clamp(new_x, 0, 1)

class PgdAttackEps(object):
    def __init__(self, model, batch_size, goal, distance_metric, magnitude=4/255, alpha=1/255, iteration=100, num_restart=1, random_start=True, use_best=False, device=None, _logger=None):
        self.model=model
        self.batch_size=batch_size
        self.goal=goal
        self.distance_metric=distance_metric
        self.magnitude=magnitude
        self.alpha=alpha
        self.iteration=iteration
        self.num_restart=num_restart
        self.random_start=random_start
        self.use_best=use_best
        self.device=device
        self._logger=_logger

    def config(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def batch_attack(self, input_image, label, target_label):
        # generate adversarial examples
        prev_training = bool(self.model.training)
        self.model.eval()
        orig_input = input_image.clone().detach().cuda(non_blocking=True)

        attack_criterion = torch.nn.CrossEntropyLoss(reduction='none')

        best_loss = None
        best_x = None

        for _ in range(self.num_restart):
            step = LinfStepEps(eps=self.magnitude, orig_input=orig_input, step_size=self.alpha)
            images = orig_input.clone().detach()

            if self.random_start:
                images = step.random_perturb(images) 
            for _ in range(self.iteration):
                images = images.clone().detach().requires_grad_(True)
                adv_losses = attack_criterion(self.model(images), label)
                torch.mean(adv_losses).backward()
                grad = images.grad.detach()

                with torch.no_grad():
                    varlist = [adv_losses, best_loss, images, best_x]
                    best_loss, best_x = replace_best(*varlist) if self.use_best else (adv_losses, images)

                    images = step.step(images, grad)
                    images = step.project(images)

            adv_losses = attack_criterion(self.model(images), label)
            varlist = [adv_losses, best_loss, images, best_x]
            best_loss, best_x = replace_best(*varlist) if self.use_best else (adv_losses, images)
        if prev_training:
            self.model.train()
    
        return best_x