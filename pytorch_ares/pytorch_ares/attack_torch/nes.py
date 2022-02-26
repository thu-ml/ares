import torch
import numpy as np


class NES(object):
    def __init__(self, model, nes_samples, sample_per_draw, p, max_queries, epsilon, step_size,
                device, data_name, search_sigma=0.02, decay=0.00, random_perturb_start=False, 
                target=False):
        self.model = model
        self.p = p
        self.epsilon = epsilon
        self.step_size = step_size
        self.data_name = data_name
        self.max_queries = max_queries
        self.device = device
        self.search_sigma = search_sigma
        nes_samples = nes_samples if nes_samples else sample_per_draw
        self.nes_samples = (nes_samples // 2) *2
        self.sample_per_draw = (sample_per_draw // self.nes_samples) * self.nes_samples
        self.nes_iters = self.sample_per_draw // self.nes_samples #2
        # self.n_samples = n_samples
        self.decay = decay
        self.random_perturb_start = random_perturb_start 
        self.target = target
        
        self.min_value = 0
        self.max_value = 1
        if self.data_name=="cifar10" and self.target:
            raise AssertionError('cifar10 dont support targeted attack')

    
    def _is_adversarial(self,x, y, y_target):    
        output = torch.argmax(self.model(x), dim=1)
        if self.target:
            return output == y_target
        else:
            return output != y
    

    def _margin_logit_loss(self, x, labels, target_labels):
        
        #x [10,3,32,32]
        outputs = self.model(x)
        if self.target:
            one_hot_labels = torch.eye(len(outputs[0]))[target_labels].to(self.device)
            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            cost = -torch.clamp((i-j), min=0)  # -self.kappa=0
        else:
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)
            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            cost = -torch.clamp((j-i), min=0)  # -self.kappa=0
        return cost
        
    def clip_eta(self, batchsize, eta, norm, eps):
        if norm == np.inf:
            eta = torch.clamp(eta, -eps, eps)
        elif norm == 2:
            normVal = torch.norm(eta.view(batchsize, -1), self.p, 1)#求范数
            mask = normVal<=eps
            scaling = eps/normVal
            scaling[mask] = 1
            eta = eta*scaling.view(batchsize, 1, 1, 1)
        else:
            raise NotImplementedError
        return eta

    # It does not call get_loss()
    def nes_gradient(self, x, y, ytarget):
        x_shape = x.size()
        g = torch.zeros(x_shape).to(self.device)
        mean = torch.zeros(x_shape).to(self.device)
        std = torch.ones(x_shape).to(self.device)

        for i in range(self.nes_iters):
            u = torch.normal(mean, std).to(self.device)
            pred = self._margin_logit_loss(torch.clamp(x+self.search_sigma*u,self.min_value,self.max_value),y,ytarget)
            g = g + pred*u
            pred = self._margin_logit_loss(torch.clamp(x-self.search_sigma*u,self.min_value,self.max_value), y,ytarget)
            g = g - pred*u

        return g/(2*self.nes_iters*self.search_sigma)
    

    def nes(self, x_victim, y_victim, y_target):
        batchsize = x_victim.shape[0]
        with torch.no_grad():
            self.model.eval()
            x_victim = x_victim.to(self.device)
            y_victim = y_victim.to(self.device)
            if y_target is not None:
                y_target = y_target.to(self.device)
            self.model.to(self.device)
           
            if self._is_adversarial(x_victim, y_victim, y_target):
                self.detail['queries'] = 0
                self.detail['success'] = True
                return x_victim
            
            self.detail['success'] = False
            queries = 0
            
            x_adv = x_victim.clone().to(self.device)

            if self.random_perturb_start:
                noise = torch.rand(x_adv.size()).to(self.device)
                normalized_noise = self.clip_eta(batchsize, noise, self.p, self.epsilon)
                x_adv += normalized_noise

            momentum = torch.zeros_like(x_adv)
            self.model.eval()

            while queries+self.sample_per_draw <=  self.max_queries:
                queries += self.sample_per_draw
                x_adv.requires_grad = True
                self.model.zero_grad()
                grad = self.nes_gradient(x_adv, y_victim, y_target) if self.data_name =='imagenet' else self.nes_gradient(x_adv, y_victim, None)
                grad = grad + momentum * self.decay
                momentum = grad
                if self.p==np.inf:
                    updates = grad.sign()
                else:
                    normVal = torch.norm(grad.view(batchsize, -1), self.p, 1)
                    updates = grad/normVal.view(batchsize, 1, 1, 1)
                updates = updates*self.step_size
                
                x_adv = x_adv + updates
                delta = x_adv-x_victim
                delta = self.clip_eta(batchsize, delta, self.p, self.epsilon)
                x_adv = torch.clamp(x_victim + delta, min=self.min_value, max=self.max_value).detach()
                
                if self._is_adversarial(x_adv, y_victim, y_target):
                    self.detail['success'] = True
                    break
            self.detail['queries'] = queries
            return x_adv

    def forward(self, xs, ys, ys_target): 
        adv_xs = []
        self.detail = {}
        for i in range(len(xs)):
            print(i + 1, end=' ')
            if self.data_name=='cifar10':
                adv_x = self.nes(xs[i].unsqueeze(0), ys[i].unsqueeze(0), None)
            else:
                adv_x = self.nes(xs[i].unsqueeze(0), ys[i].unsqueeze(0), ys_target[i].unsqueeze(0))
            if self.p==np.inf:
                distortion = torch.mean((adv_x - xs[i].unsqueeze(0))**2) / ((1-0)**2)
            else:
                distortion = torch.mean((adv_x - xs[i].unsqueeze(0))**2) / ((1-0)**2)
            print(distortion.item(), end=' ')
            print(self.detail)
            adv_xs.append(adv_x)
        adv_xs = torch.cat(adv_xs, 0)
        return adv_xs
            
            