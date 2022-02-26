# This is a simple implementation of AWP for Standard Adversarial Training (Madry)
import copy
import torch.nn as nn
import torch.optim as optim
import torch
EPS = 1E-20


def normalize(perturbations, weights):
    perturbations.mul_(weights.norm()/(perturbations.norm() + EPS))


def normalize_grad_by_weights(weights, ref_weights):
    for w, ref_w in zip(weights, ref_weights):
        if w.dim() <= 1:
            w.grad.data.fill_(0)  # ignore perturbations with 1 dimension (e.g. BN, bias)
        else:
            normalize(w.grad.data, ref_w)


class AdvWeightPerturb(object):
    """
    This is an implementation of AWP ONLY for Standard adversarial training
    """
    def __init__(self, model, eta, nb_iter=1):
        super(AdvWeightPerturb, self).__init__()
        self.eta = eta
        self.nb_iter = nb_iter
        self.model = model
        self.optim = optim.SGD(model.parameters(), lr=eta/nb_iter)
        self.criterion = nn.CrossEntropyLoss()
        self.diff = None

    def perturb(self, X_adv, y):
        # store the original weight
        old_w = copy.deepcopy([p.data for p in self.model.parameters()])

        # perturb the model
        for idx in range(self.nb_iter):
            self.optim.zero_grad()
            outputs = self.model(X_adv)
            loss = - self.criterion(outputs, y)
            loss.backward()

            # normalize the gradient
            normalize_grad_by_weights(self.model.parameters(), old_w)

            self.optim.step()

        # calculate the weight perturbation
        self.diff = [w1 - w2 for w1, w2 in zip(self.model.parameters(), old_w)]

    def restore(self):
        for w, v in zip(self.model.parameters(), self.diff):
            w.data.sub_(v.data)
