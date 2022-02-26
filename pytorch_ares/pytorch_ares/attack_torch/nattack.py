import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def nattack_loss(inputs, targets,target_lables, device,targeted):
    batch_size = inputs.shape[0] #100
    losses = torch.zeros(batch_size).to(device)
    if targeted:
        for i in range(batch_size):
            target_lable = target_lables[i]
            correct_logit = inputs[i][target_lable]
            tem_tensor = torch.zeros(inputs.shape[-1]).to(device)
            tem_tensor[target_lable] = -10000
            wrong_logit = inputs[i][torch.argmax(inputs[i] + tem_tensor)]
            losses[i] = wrong_logit - correct_logit
        return -losses
    else:
        for i in range(batch_size):
            target = targets[i]
            correct_logit = inputs[i][target]
            tem_tensor = torch.zeros(inputs.shape[-1]).to(device)
            tem_tensor[target] = -10000
            wrong_logit = inputs[i][torch.argmax(inputs[i] + tem_tensor)]
            losses[i] = wrong_logit - correct_logit
        return losses



def scale(x, dst_min, dst_max, src_min, src_max):
    k = (dst_max - dst_min) / (src_max - src_min)
    b = dst_min - k * src_min
    return k * x + b


class Nattack(object):
    def __init__(self,model, eps, max_queries, device,data_name, distance_metric,target, sample_size=100, lr=0.02, sigma=0.1):
        
        self.max_queries = max_queries
        self.sample_size = sample_size
        self.distance_metric = distance_metric
        self.lr = lr
        self.target = target
        self.sigma = sigma
        self.loss_func = nattack_loss
        self.clip_max = 1
        self.model = model
        self.clip_min = 0
        self.device = device
        self.data_name = data_name
        self.eps = eps
        if self.data_name=="cifar10" and self.target:
            raise AssertionError('cifar10 dont support targeted attack')


    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))
    
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
    
    def is_adversarial(self, x, y,ytarget):
        out = self.model(x)
        pred = torch.argmax(out)
        if self.target:
            return pred == ytarget
        else:
            return pred != y
    
    def scale_to_tanh(self,x):
        bound = 1e-6 - 1
        return scale(x, bound, -bound, self.clip_min, self.clip_max)


    def nattack(self ,x, y, y_target):
        self.model.eval()
        batchsize = x.shape[0]
        nx = x.to(self.device)#torch.Size([1, 3, 32, 32])
        ny = y.to(self.device)#torch.Size([1])
        if y_target is not None:
            vy_target = y_target.to(self.device)
        else:
            vy_target = None
        shape = nx.shape
        model = self.model.to(self.device)
        if self.is_adversarial(nx, ny, vy_target):
            self.detail['queries'] = 0
            self.detail['success'] = True
            return nx
            
        self.detail['success'] = False

        with torch.no_grad():
            y = torch.tensor([y] * self.sample_size)#torch.Size([100])
            y = y.to(self.device)
            if vy_target is not None:
                y_target = torch.tensor([y_target] * self.sample_size)
                y_target = y_target.to(self.device)
                
            #正态分布,随机初始化mean
            mu = torch.randn(1, x.size(1), x.size(2), x.size(3)).to(self.device) * 0.001 #torch.Size([1, 3, 32, 32])

            self.detail['success'] = False
            q = 0
            while q < self.max_queries:
                pert = torch.randn(self.sample_size, x.size(1), x.size(2), x.size(3)).to(self.device) #torch.Size([100, 3, 32, 32])
                seed_z = mu + self.sigma * pert #torch.Size([100, 3, 32, 32])
                #s双线性插值
                g0_z = F.interpolate(seed_z, shape[-2:], mode='bilinear', align_corners=False) #torch.Size([100, 3, 32, 32])
               
                arctanh_xs = self.atanh(self.scale_to_tanh(nx)) #torch.Size([1, 3, 32, 32])
                g_z = 0.5 * (torch.tanh(arctanh_xs + g0_z) + 1) #torch.Size([100, 3, 32, 32])
                noise = g_z - nx #扰动
                adv_image = nx + self.clip_eta(batchsize, noise, self.distance_metric, self.eps) #torch.Size([100, 3, 32, 32])

                outputs = model(adv_image) #torch.Size([100, 10])
                loss = self.loss_func(outputs, y, y_target, self.device, self.target)
           
                normalize_loss = (loss - torch.mean(loss)) / (torch.std(loss) + 1e-7)

                q += self.sample_size
                # z-score fi'*pert
                grad = normalize_loss.reshape(-1, 1, 1, 1) * pert 
                grad = torch.mean(grad, dim=0) / self.sigma
                # self.lr = setpsize/batchsize update mu
                mu = mu + self.lr * grad
                mu_test = F.interpolate(mu, shape[-2:], mode='bilinear', align_corners=False)

                adv_t = 0.5 * (torch.tanh(arctanh_xs + mu_test) + 1)
                adv_t = nx + self.clip_eta(batchsize, adv_t - nx, self.distance_metric, self.eps)

                if self.is_adversarial(adv_t, ny, vy_target):
                    self.detail['success'] = True
                    # print('image is adversarial, query', q)
                    break
            self.detail['queries'] = q
        return adv_t

    def forward(self, xs, ys, ytarget):
        adv_xs = []
        self.detail = {}
        for i in range(len(xs)):
            print(i + 1, end=' ')
            if self.data_name=='cifar10':
                adv_x = self.nattack(xs[i].unsqueeze(0), ys[i].unsqueeze(0), None)
            else:
                adv_x = self.nattack(xs[i].unsqueeze(0), ys[i].unsqueeze(0), ytarget[i].unsqueeze(0))
            
            if self.distance_metric==np.inf:
                distortion = torch.mean((adv_x - xs[i].unsqueeze(0))**2) / ((1-0)**2)
            else:
                distortion = torch.mean((adv_x - xs[i].unsqueeze(0))**2) / ((1-0)**2)
            print(distortion.item(), end=' ')
            print(self.detail)
            adv_xs.append(adv_x)
        adv_xs = torch.cat(adv_xs, 0)
        return adv_xs