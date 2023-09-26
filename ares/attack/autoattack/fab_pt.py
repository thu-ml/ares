# Copyright (c) 2019-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from ares.attack.autoattack.utils import zero_gradients
from ares.attack.autoattack.fab_base import FABAttack

class FABAttack_PT(FABAttack):
    """Fast Adaptive Boundary Attack (Linf, L2, L1)
    https://arxiv.org/abs/1907.02044

    Args:
            predict (): Forward pass function.
            norm (): Lp-norm to minimize ('Linf', 'L2', 'L1' supported).
            n_restarts (): Number of random restarts.
            n_iter (): Number of iterations
            eps (): Epsilon for the random restarts.
            alpha_max (): Alpha_max.
            eta (): Overshooting.
            beta (): Backward step.
            verbose (): Whether to print information.
            seed (): Random seed.
            device (): torch.device.
            n_target_classes (): Number of target classes.
    """

    def __init__(
            self,
            predict,
            norm='Linf',
            n_restarts=1,
            n_iter=100,
            eps=None,
            alpha_max=0.1,
            eta=1.05,
            beta=0.9,
            verbose=False,
            seed=0,
            device=None,
            n_target_classes=9):


        self.predict = predict
        super().__init__(norm, n_restarts, n_iter, eps, alpha_max, eta,
                         beta, verbose, seed, device, n_target_classes)

    def _predict_fn(self, x):
        return self.predict(x)

    def _get_predicted_label(self, x):
        with torch.no_grad():
            outputs = self._predict_fn(x)
        _, y = torch.max(outputs, dim=1)
        return y

    def get_diff_logits_grads_batch(self, imgs, la):
        im = imgs.clone().requires_grad_()
        with torch.enable_grad():
            y = self.predict(im)

        g2 = torch.zeros([y.shape[-1], *imgs.size()]).to(self.device)
        grad_mask = torch.zeros_like(y)
        for counter in range(y.shape[-1]):
            zero_gradients(im)
            grad_mask[:, counter] = 1.0
            y.backward(grad_mask, retain_graph=True)
            grad_mask[:, counter] = 0.0
            g2[counter] = im.grad.data

        g2 = torch.transpose(g2, 0, 1).detach()
        #y2 = self.predict(imgs).detach()
        y2 = y.detach()
        df = y2 - y2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
        dg = g2 - g2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
        df[torch.arange(imgs.shape[0]), la] = 1e10

        return df, dg

    def get_diff_logits_grads_batch_targeted(self, imgs, la, la_target):
        u = torch.arange(imgs.shape[0])
        im = imgs.clone().requires_grad_()
        with torch.enable_grad():
            y = self.predict(im)
            diffy = -(y[u, la] - y[u, la_target])
            sumdiffy = diffy.sum()

        zero_gradients(im)
        sumdiffy.backward()
        graddiffy = im.grad.data
        df = diffy.detach().unsqueeze(1)
        dg = graddiffy.unsqueeze(1)

        return df, dg
