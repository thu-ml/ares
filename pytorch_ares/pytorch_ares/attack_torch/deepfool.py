"""adversary.py"""
import torch
import numpy as np

class DeepFool(object):
    def __init__(self, net, overshoot, max_iter, norm, target, device):
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.net = net
        self.norm = norm
        self.device = device
        self.target = target
        self.min_value = 0
        self.max_value = 1
        if self.target:
            raise AssertionError('DeepFool dont support targeted attack')

    def deepfool(self, x, y, y_target=None):
        with torch.no_grad():
            logits = self.net(x)
            outputs = torch.argmax(logits, dim=1)
            if outputs!=y:
                return x
        self.nb_classes = logits.size(-1)

        adv_x = x.clone().detach().requires_grad_()
        
        iteration = 0
        logits = self.net(adv_x)
        current = logits.max(1)[1].item()
        original = logits.max(1)[1].item()
        noise = torch.zeros(x.size()).to(self.device)
        w = torch.zeros(x.size()).to(self.device)
        
        while (current == original and iteration < self.max_iter):
            gradients_0 = torch.autograd.grad(logits[0, current], [adv_x],retain_graph=True)[0].detach()
            
            for k in range(self.nb_classes):
                pert = np.inf
                if k==current:
                    continue
                gradients_1 = torch.autograd.grad(logits[0, k], [adv_x],retain_graph=True)[0].detach()
                w_k = gradients_1 - gradients_0
                f_k = logits[0, k] - logits[0, current]
                if self.norm == np.inf:
                    pert_k = (torch.abs(f_k) + 0.00001)/ torch.norm(w_k.flatten(1),1, -1)
                elif self.norm == 2:
                    pert_k = (torch.abs(f_k) + 0.00001) / torch.norm(w_k.flatten(1),2,-1)
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
                if self.norm == np.inf:
                    r_i = (pert + 1e-4) * w.sign()
                elif self.norm==2:
                    r_i = (pert + 1e-4) * w / torch.norm(w.flatten(1),2,-1)
            noise += r_i.clone()
            adv_x = torch.clamp(adv_x + noise, self.min_value, self.max_value).requires_grad_()
            logits = self.net(adv_x + noise)
            current = logits.max(1)[1].item()
            iteration = iteration + 1


            

        
        adv_x = torch.clamp((1 + self.overshoot) * noise + x, self.min_value, self.max_value)
        
        return adv_x

    def forward(self, xs, ys, ys_target): 
        adv_xs = []
        for i in range(len(xs)):
            print(i + 1, end=' ')
            adv_x = self.deepfool(xs[i].unsqueeze(0), ys[i].unsqueeze(0), None)  
            distortion = torch.mean((adv_x - xs[i].unsqueeze(0))**2) / ((1-0)**2)
            print(distortion.item())
            adv_xs.append(adv_x)
        adv_xs = torch.cat(adv_xs, 0)
        return adv_xs