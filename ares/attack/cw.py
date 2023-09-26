import numpy as np
import torch
from torch.autograd import Variable
from ares.utils.registry import registry

@registry.register_attack('cw')
class CW(object):
    ''' Carlini & Wagner Attack (C&W). A white-box iterative optimization-based method. Require a differentiable logits.
    
    Example:
        >>> from ares.utils.registry import registry
        >>> attacker_cls = registry.get_attack('cw')
        >>> attacker = attacker_cls(model)
        >>> adv_images = attacker(images, labels, target_labels)
    
    - Supported distance metric: 2.
    - References: References: https://arxiv.org/pdf/1608.04644.pdf.
    '''

    def __init__(self, model, device='cuda', norm=2, kappa=0, lr=0.2, init_const=0.01,
                 max_iter=200, binary_search_steps=4, num_classes=1000, target=False):
        '''
        Args:
            model (torch.nn.Module): The target model to be attacked.
            device  (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            norm (float): The norm of distance calculation for adversarial constraint. Defaults to 2.
            kappa (float): Defaults to 0.
            lr (float): The learning rate for attack process.
            init_const (float): The initialized constant.
            max_iter (int): The maximum iteration.
            binary_search_steps (int): The steps for binary search.
            num_classes (int): The number of classes of all the labels.
            target (bool): Conduct target/untarget attack. Defaults to False.
        '''

        self.net = model
        self.device = device
        self.IsTargeted = target
        self.kappa = kappa
        self.learning_rate = lr
        self.init_const = init_const
        self.lower_bound = 0.0
        self.upper_bound = 1.0
        self.max_iter = max_iter
        self.norm = norm
        self.binary_search_steps = binary_search_steps
        self.class_type_number = num_classes
        assert self.norm == 2, 'curreent cw only support l_2'

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))
    
    def __call__(self, images=None, labels=None, target_labels=None):
        '''This function perform attack on target images with corresponding labels 
        and target labels for target attack.

        Args:
            images (torch.Tensor): The images to be attacked. The images should be torch.Tensor with shape [N, C, H, W] and range [0, 1].
            labels (torch.Tensor): The corresponding labels of the images. The labels should be torch.Tensor with shape [N, ]
            target_labels (torch.Tensor): The target labels for target attack. The labels should be torch.Tensor with shape [N, ]

        Returns:
            torch.Tensor: Adversarial images with value range [0,1].

        '''        
        device = self.device
        targeted = self.IsTargeted
        copy_images = images.clone()
        copy_labels = labels.clone()
        if target_labels is not None:
            copy_target_labels = target_labels.clone()
        else:
            copy_target_labels = copy_labels

        batch_size = images.shape[0]

        mid_point = (self.upper_bound + self.lower_bound) * 0.5
        half_range = (self.upper_bound - self.lower_bound) * 0.5
        arctanh_images = self.atanh((copy_images - mid_point) / half_range * 0.9999)
        var_images=arctanh_images.clone()
        var_images.requires_grad=True

        const_origin = torch.ones(batch_size, device=self.device) * self.init_const
        c_upper_bound = [1e10] * batch_size
        c_lower_bound = torch.zeros(batch_size, device=self.device)
        targets_in_one_hot = []
        targeteg_class_in_one_hot = []

        temp_one_hot_matrix = torch.eye(int(self.class_type_number), device=self.device)
        if targeted:
            for i in range(batch_size):
                current_target1 = temp_one_hot_matrix[copy_target_labels[i]]
                targeteg_class_in_one_hot.append(current_target1)
            targeteg_class_in_one_hot = torch.stack(targeteg_class_in_one_hot).clone().type_as(images).to(self.device) #torch.Size([10, 10])
        else:
            for i in range(batch_size):
                current_target = temp_one_hot_matrix[copy_labels[i]]
                targets_in_one_hot.append(current_target)
            targets_in_one_hot = torch.stack(targets_in_one_hot).clone().type_as(images).to(self.device) #torch.Size([10, 10])

        best_l2 = [1e10] * batch_size
        best_perturbation = torch.zeros(var_images.size())
        current_prediction_class = [-1] * batch_size

        def attack_achieved(pre_softmax, true_class, target_class):
            targeted = self.IsTargeted
            if targeted:
                pre_softmax[target_class] -= self.kappa
                return torch.argmax(pre_softmax).item() == target_class
            else:
                pre_softmax[true_class] -= self.kappa
                return torch.argmax(pre_softmax).item() != true_class

        for search_for_c in range(self.binary_search_steps):
            modifier = torch.zeros(var_images.shape).float()
            modifier = Variable(modifier.to(device), requires_grad=True)
            optimizer = torch.optim.Adam([modifier], lr=self.learning_rate)
            var_const = const_origin.clone().to(device)
            # print("\tbinary search step {}:".format(search_for_c))

            for iteration_times in range(self.max_iter):
                # inverse the transform tanh -> [0, 1]
                perturbed_images = (torch.tanh(var_images + modifier) * half_range + mid_point)
                prediction = self.net(perturbed_images)

                l2dist = torch.sum(
                    (perturbed_images - (torch.tanh(var_images) * half_range + mid_point))
                    ** 2,
                    [1, 2, 3],
                )

                if targeted:
                    constraint_loss = torch.max((prediction - 1e10 * targeteg_class_in_one_hot).max(1)[0] - (prediction * targeteg_class_in_one_hot).sum(1),
                        torch.ones(batch_size, device=device) * self.kappa * -1,
                    )
                else:
                    constraint_loss = torch.max((prediction * targets_in_one_hot).sum(1)
                    - (prediction - 1e10 * targets_in_one_hot).max(1)[0],
                    torch.ones(batch_size, device=device) * self.kappa * -1,
                )

                loss_f = var_const * constraint_loss
                loss = l2dist.sum() + loss_f.sum()  # minimize |r| + c * loss_f(x+r,l)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                for i in range(prediction.shape[0]):
                    dist=l2dist[i]
                    score=prediction[i]
                    img=perturbed_images[i]
                    if dist.item() < best_l2[i] and attack_achieved(score, copy_labels[i], copy_target_labels[i]):
                        best_l2[i] = dist
                        current_prediction_class[i] = torch.argmax(score)
                        best_perturbation[i] = img

            # update the best constant c for each sample in the batch
            for i in range(batch_size):
                if (
                    current_prediction_class[i] == copy_labels[i].item()
                    and current_prediction_class[i] != -1
                ):
                    c_upper_bound[i] = min(c_upper_bound[i], const_origin[i].item())
                    if c_upper_bound[i] < 1e10:
                        const_origin[i] = (c_lower_bound[i].item() + c_upper_bound[i]) / 2.0
                else:
                    c_lower_bound[i] = max(c_lower_bound[i].item(), const_origin[i].item())
                    if c_upper_bound[i] < 1e10:
                        const_origin = (c_lower_bound[i].item() + c_upper_bound[i]) / 2.0
                    else:
                        const_origin[i] *= 10

        adv_images = best_perturbation.to(device)
        return adv_images
