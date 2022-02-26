import torch
import numpy as np


class Evolutionary(object):
    def __init__(self, model,data_name,targeted,device, ccov=0.001, decay_weight=0.99, max_queries=10000, mu=0.01, sigma=3e-2, maxlen=30):
        self.model = model
        self.ccov = ccov
        self.decay_weight = decay_weight
        self.max_queries = max_queries
        self.mu = mu
        self.device =device
        self.sigma = sigma
        self.maxlen = maxlen
        self.targeted = targeted
        self.min_value = 0
        self.max_value = 1
        self.data_name =data_name
        if self.targeted:
            raise AssertionError('dont support targeted attack')


    def _is_adversarial(self,x, y, ytarget):
        output = torch.argmax(self.model(x), dim=1)
        if self.targeted:
            return output == ytarget
        else:
            return output != y

    def get_init_noise(self, x_target, y, ytarget):
        while True:
            x_init = torch.rand(x_target.size()).to(self.device)
            x_init = torch.clamp(x_init, min=self.min_value, max=self.max_value)
            if self._is_adversarial(x_init, y, ytarget):
                print("Success getting init noise",end=' ')
                return x_init

   
    def evolutionary(self, x, y,ytarget):
        x = x.to(self.device)
        y = y.to(self.device)
        if ytarget is not None:
            ytarget = ytarget.to(self.device)
        pert_shape = (x.size(0),x.size(1),x.size(2),x.size(3))#这样设置形状的原因是因为后面用到的cv2.resize是需要对H,W进行修改的
        m = np.prod(pert_shape)#返回内部所有元素的积
        k = int(m / 20)
        evolutionary_path = np.zeros(pert_shape)
        decay_weight = self.decay_weight
        diagonal_covariance = np.ones(pert_shape)
        ccov = self.ccov
        if self._is_adversarial(x, y,ytarget):
            return x

        # find an starting point
        x_adv = self.get_init_noise(x , y, ytarget)
            
        mindist = 1e10
        stats_adversarial = []
        for _ in range(self.max_queries):
            unnormalized_source_direction = x - x_adv
            source_norm = torch.norm(unnormalized_source_direction)
            if mindist > source_norm:
                mindist = source_norm
                best_adv = x_adv

            selection_prob = diagonal_covariance.reshape(-1) / np.sum(diagonal_covariance)
            selection_indices = np.random.choice(m, k, replace=False, p=selection_prob)
            pert = np.random.normal(0.0, 1.0, pert_shape)
            factor = np.zeros([m])
            factor[selection_indices] = True
            pert *= factor.reshape(pert_shape) * np.sqrt(diagonal_covariance)
            pert_large = torch.Tensor(pert).to(self.device)

            biased = (x_adv + self.mu * unnormalized_source_direction).to(self.device)
            candidate = biased + self.sigma * source_norm * pert_large / torch.norm(pert_large)
            candidate = x - (x - candidate) / torch.norm(x - candidate) * torch.norm(x - biased)
            candidate = torch.clamp(candidate, self.min_value, self.max_value)
            
            if self._is_adversarial(candidate, y, ytarget):
                x_adv = candidate
                evolutionary_path = decay_weight * evolutionary_path + np.sqrt(1-decay_weight** 2) * pert
                diagonal_covariance = (1 - ccov) * diagonal_covariance + ccov * (evolutionary_path ** 2)
                stats_adversarial.append(1)
            else:
                stats_adversarial.append(0)
            if len(stats_adversarial) == self.maxlen:
                self.mu *= np.exp(np.mean(stats_adversarial) - 0.2)
                stats_adversarial = []
        return best_adv


    def forward(self, xs, ys, ys_target):
        adv_xs = []
        for i in range(len(xs)):
            print(i + 1, end=' ')
            if self.data_name=='cifar10':
                adv_x = self.evolutionary(xs[i].unsqueeze(0), ys[i].unsqueeze(0), None)
            else:
                adv_x = self.evolutionary(xs[i].unsqueeze(0), ys[i].unsqueeze(0), ys_target[i].unsqueeze(0))
            distortion = torch.mean((adv_x - xs[i].unsqueeze(0))**2) / ((1-0)**2)
            print(distortion.item())
            adv_xs.append(adv_x)
        adv_xs = torch.cat(adv_xs, 0)
        return adv_xs