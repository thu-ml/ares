import torch
from collections import deque
import numpy as np

class BoundaryAttack(object):
    def __init__(self, net, spherical_step_eps, p,orth_step_factor,perp_step_factor, orthogonal_step_eps, max_iters, data_name, device,target):
        self.net = net
        self.spherical_step = spherical_step_eps
        self.p = p
        self.target = target
        self.orthogonal_step = orthogonal_step_eps#1e-2 
        self.max_iters = max_iters
        self.data_name = data_name
        self.min_value = 0
        self.max_value = 1
        self.orth_step_factor = orth_step_factor
        self.perp_step_factor = perp_step_factor
        self.device = device
        self.orth_step_stats = deque(maxlen=30)
        self.perp_step_stats = deque(maxlen=100)
        if self.data_name=="cifar10" and self.target:
            raise AssertionError('cifar10 dont support targeted attack')
        if self.p !=2:
            raise AssertionError('boundary attck only support L2 bound')
    
    def _is_adversarial(self,x, y, y_target):    
        output = torch.argmax(self.net(x), dim=1)
        if self.target:
            return output == y_target
        else:
            return output != y
    
    def get_init_noise(self, x_target, y, ytarget):
        while True:
            x_init = torch.rand(x_target.size()).to(self.device)
            x_init = torch.clamp(x_init, min=self.min_value, max=self.max_value)
            if self._is_adversarial(x_init, y, ytarget):
                print("Success getting init noise",end=' ')
                return x_init


    def perturbation(self, x, x_adv,y,ytarget):
        unnormalized_source_direction = x - x_adv
        source_norm = torch.norm(unnormalized_source_direction)
        source_direction = unnormalized_source_direction / source_norm
        perturbation = torch.normal(torch.zeros_like(x_adv), torch.ones_like(x_adv)).to(self.device)
        dot = torch.matmul(perturbation, source_direction)
        perturbation -= dot * source_direction
        perturbation *= self.perp_step_factor * source_norm / torch.norm(perturbation)

        D = (1 / torch.sqrt(torch.tensor(self.perp_step_factor ** 2.0 + 1))).to(self.device)
        direction = perturbation - unnormalized_source_direction
        spherical_candidate = torch.clamp(x + D * direction, self.min_value, self.max_value)

        new_source_direction = x - spherical_candidate
        new_source_direction_norm = torch.norm(new_source_direction)
        length = self.orthogonal_step * source_norm
        deviation = new_source_direction_norm - source_norm
        length = max(0, length + deviation) / new_source_direction_norm
        candidate = torch.clamp(spherical_candidate + length * new_source_direction, self.min_value, self.max_value)
        return candidate
   
  
    def boundary(self, x, y, ytarget): 
        if self._is_adversarial(x, y, ytarget):
            print("The original image is already adversarial")
            return x
        x_adv = self.get_init_noise(x, y, ytarget)
        
        for i in range(self.max_iters):
            pertubed = self.perturbation(x, x_adv,y,ytarget)
            if self._is_adversarial(pertubed,y, ytarget):
                x_adv = pertubed
            if len(self.perp_step_stats) == self.perp_step_stats.maxlen:
                if torch.Tensor(self.perp_step_stats).mean() > 0.5:
                    # print('Boundary too linear, increasing steps')
                    self.spherical_step /= self.perp_step_factor
                    self.orthogonal_step /= self.orth_step_factor
                elif torch.Tensor(self.perp_step_stats).mean() < 0.2:
                    # print('Boundary too non-linear, decreasing steps')
                    self.spherical_step *= self.perp_step_factor
                    self.orthogonal_step *= self.orth_step_factor
                self.perp_step_stats.clear()

            if len(self.orth_step_stats) == self.orth_step_stats.maxlen:
                if torch.Tensor(self.orth_step_stats).mean() > 0.5:
                    # print('Success rate too high, increasing source step')
                    self.orthogonal_step /= self.orth_step_factor
                elif torch.Tensor(self.orth_step_stats).mean() < 0.2:
                    # print('Success rate too low, decreasing source step')
                    self.orthogonal_step *= self.orth_step_factor
                self.orth_step_stats.clear()
        return x_adv
        
    def forward(self, xs, ys, ys_target):
        adv_xs = []
        for i in range(len(xs)):
            print(i + 1, end=' ')
            if self.data_name=='cifar10':
                adv_x = self.boundary(xs[i].unsqueeze(0), ys[i].unsqueeze(0), None)
            else:
                adv_x = self.boundary(xs[i].unsqueeze(0), ys[i].unsqueeze(0), ys_target[i].unsqueeze(0))
            distortion = torch.mean((adv_x - xs[i].unsqueeze(0))**2) / ((1-0)**2)
            print(distortion.item())
            adv_xs.append(adv_x)
        adv_xs = torch.cat(adv_xs, 0)
        return adv_xs
        