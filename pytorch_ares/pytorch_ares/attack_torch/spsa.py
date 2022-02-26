import numpy as np
from scipy.fftpack import ss_diff
import torch
from torch import optim


class SPSA(object):
    def __init__(self, model,norm, device, eps, learning_rate, delta, spsa_samples, sample_per_draw, 
                 nb_iter, data_name, early_stop_loss_threshold=None, IsTargeted=None):
       
        self.model = model
        self.device = device
        self.IsTargeted = IsTargeted
        self.eps = eps #0.05
        self.learning_rate = learning_rate #0.01
        self.delta = delta #0.01
        spsa_samples = spsa_samples if spsa_samples else sample_per_draw
        self.spsa_samples = (spsa_samples // 2) *2
        self.sample_per_draw = (sample_per_draw // self.spsa_samples) * self.spsa_samples
        self.nb_iter = nb_iter #20
        self.norm = norm # np.inf
        self.data_name = data_name
        self.early_stop_loss_threshold = early_stop_loss_threshold
        self.clip_min = 0
        self.clip_max = 1
        if self.data_name=="cifar10" and self.IsTargeted:
            raise AssertionError('cifar10 dont support targeted attack')
    

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
   
   
    def _get_batch_sizes(self, n, max_batch_size):
        batches = [max_batch_size for _ in range(n // max_batch_size)]
        if n % max_batch_size > 0:
            batches.append(n % max_batch_size)
        return batches
    
    def _compute_spsa_gradient(self, loss_fn, x, delta, nb_sample, max_batch_size):

        grad = torch.zeros_like(x)
        x = x.unsqueeze(0)
        x = x.expand(max_batch_size, *x.shape[1:]).contiguous()
        v = torch.empty_like(x[:, :1, ...])

        for batch_size in self._get_batch_sizes(nb_sample, max_batch_size):
            x_ = x[:batch_size]
            vb = v[:batch_size]
            vb = vb.bernoulli_().mul_(2.0).sub_(1.0)
            v_ = vb.expand_as(x_).contiguous()
            x_shape = x_.shape
            x_ = x_.view(-1, *x.shape[2:])
            v_ = v_.view(-1, *v.shape[2:])
            df = loss_fn(delta * v_) - loss_fn(- delta * v_)
            df = df.view(-1, *[1 for _ in v_.shape[1:]])
            grad_ = df / (2. * delta * v_)
            grad_ = grad_.view(x_shape)
            grad_ = grad_.sum(dim=0, keepdim=False)
            grad += grad_
        grad /= nb_sample

        return grad
        
    
    def _is_adversarial(self,x, y, y_target):
        output = torch.argmax(self.model(x), dim=1)
        if self.IsTargeted:
            return output == y_target
        else:
            return output != y
    
    
    def _margin_logit_loss(self, logits, labels, target_label):
        if self.IsTargeted:
            correct_logits = logits.gather(1, target_label[:, None]).squeeze(1)

            logit_indices = torch.arange(logits.size()[1], dtype=target_label.dtype, device=target_label.device)[None, :].expand(target_label.size()[0], -1)
            incorrect_logits = torch.where(logit_indices == target_label[:, None], torch.full_like(logits, float("-inf")), logits)
            max_incorrect_logits, _ = torch.max(incorrect_logits, 1)

            return max_incorrect_logits -correct_logits
        else:
            correct_logits = logits.gather(1, labels[:, None]).squeeze(1)

            logit_indices = torch.arange(logits.size()[1], dtype=labels.dtype, device=labels.device)[None, :].expand(labels.size()[0], -1)
            incorrect_logits = torch.where(logit_indices == labels[:, None], torch.full_like(logits, float("-inf")), logits)
            max_incorrect_logits, _ = torch.max(incorrect_logits, 1)

            return -(max_incorrect_logits-correct_logits)

    def spsa(self,x, y, y_target):
        device = self.device
        eps = self.eps
        batchsize = x.shape[0]
        learning_rate = self.learning_rate
        delta = self.delta
        spsa_samples = self.spsa_samples
        nb_iter = self.nb_iter

        v_x = x.to(device)
        v_y = y.to(device)

        if self._is_adversarial(v_x, v_y, y_target):
                self.detail['queries'] = 0
                self.detail['success'] = True
                return v_x


        perturbation = (torch.rand_like(v_x) * 2 - 1) * eps
        optimizer = optim.Adam([perturbation], lr=learning_rate)
        
        self.detail['success'] = False
        queries = 0
        
        while queries+self.sample_per_draw <= nb_iter:
            queries += self.sample_per_draw
            def loss_fn(pert):

                input1 = v_x + pert
                input1 = torch.clamp(input1, self.clip_min, self.clip_max)
                logits = self.model(input1)
                return self._margin_logit_loss(logits, v_y.expand(len(pert)), y_target.expand(len(pert))) if self.IsTargeted else self._margin_logit_loss(logits, v_y.expand(len(pert)), None)
         
            spsa_grad = self._compute_spsa_gradient(loss_fn, v_x, delta=delta, nb_sample=spsa_samples, max_batch_size=self.sample_per_draw)
            perturbation.grad = spsa_grad
            optimizer.step()

            clip_perturbation = self.clip_eta(batchsize, perturbation, self.norm, eps)
            adv_image = torch.clamp(v_x + clip_perturbation, self.clip_min, self.clip_max)
            perturbation.add_((adv_image - v_x) - perturbation)
            loss = loss_fn(perturbation).item()
   
            if (self.early_stop_loss_threshold is not None and loss < self.early_stop_loss_threshold):
                break

            if self._is_adversarial(adv_image, v_y, y_target):
                self.detail['success'] = True
                break
        self.detail['queries'] = queries
        return adv_image

    def forward(self, xs, ys, ys_target):
     
        adv_xs = []
        self.detail = {}
        for i in range(len(xs)):
            print(i + 1, end=' ')
            if self.data_name=='cifar10':
                adv_x = self.spsa(xs[i].unsqueeze(0), ys[i].unsqueeze(0), None)
            else:
                adv_x = self.spsa(xs[i].unsqueeze(0), ys[i].unsqueeze(0), ys_target[i].unsqueeze(0))
            if self.norm==np.inf:
                distortion = torch.mean((adv_x - xs[i].unsqueeze(0))**2) / ((1-0)**2) #mean_square_distance
            else:
                distortion = torch.mean((adv_x - xs[i].unsqueeze(0))**2) / ((1-0)**2)
            print(distortion.item(), end=' ')
            print(self.detail)
            adv_xs.append(adv_x)
        adv_xs = torch.cat(adv_xs, 0)
        return adv_xs