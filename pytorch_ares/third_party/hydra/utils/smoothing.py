import torch
import torch.nn.functional as F
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint


def eval_quick_smoothing(model, loader, device, sigma=0.25, nbatch=10):
    pred = []
    rad = []

    # model.eval()

    for index, (x, y) in enumerate(loader):
        q = quick_smoothing(
            model,
            x.to(device),
            y.to(device),
            device,
            sigma=sigma,
            eps=0.25,
            num_smooth=10,
            batch_size=100,
            softmax_temperature=1.0,
            detailed_output=True,
        )
        pred += list(q[0])
        rad += list(q[1])
        if index == nbatch:
            break
    # print(pred)
    print(f"Mean smooth accuracy (len = {len(pred)})= ", np.mean(pred))
    print(f"Mean rad (len={len(rad)})", np.mean(rad))


def quick_smoothing(
    model,
    x,
    y,
    device,
    sigma=1.0,
    eps=1.0,
    num_smooth=100,
    batch_size=1000,
    softmax_temperature=100.0,
    detailed_output=False,
):
    """Quick and dirty randomized smoothing 'certification', without proper
     confidence bounds. We use it only to monitor training."""

    x_noise = x.view(1, *x.shape) + sigma * torch.randn(num_smooth, *x.shape).to(device)
    x_noise = x_noise.view(-1, *x.shape[1:])
    # by setting a high softmax temperature, we are effectively using the
    # randomized smoothing approach as originally defined
    # it will be interesting to see if lower temperatures help

    preds = torch.cat(
        [
            F.softmax(softmax_temperature * model(batch), dim=-1)
            for batch in torch.split(x_noise, batch_size)
        ]
    )

    preds = preds.view(num_smooth, x.shape[0], -1).mean(dim=0)
    p_max, y_pred = preds.max(dim=-1)

    correct = (y_pred == y).data.cpu().numpy().astype("int64")
    radii = (sigma + 1e-16) * norm.ppf(p_max.data.cpu().numpy())

    err = (1 - correct).sum()
    robust_err = (1 - correct * (radii >= eps)).sum()

    if not detailed_output:
        return err, robust_err
    else:
        return correct, radii


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(
        self, base_classifier: torch.nn.Module, num_classes: int, sigma: float
    ):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(
        self,
        x: torch.tensor,
        n0: int,
        n: int,
        alpha: float,
        batch_size: int,
        device: str,
    ) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size, device)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size, device)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(
        self, x: torch.tensor, n: int, alpha: float, batch_size: int, device: str
    ) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size, device)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(
        self, x: torch.tensor, num: int, batch_size, device
    ) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device=device) * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(
                    predictions.data.cpu().numpy(), self.num_classes
                )
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
