import torch
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Any, Callable


class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class EnsembleFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Tensor:
        features = []
        for model in self.extractors:
            features.append(model(x).squeeze())
        features = torch.cat(features, dim=0)
        return features


class EnsembleFeatureLoss(nn.Module):
    def __init__(self,
                 extractors: List[BaseFeatureExtractor],
                 count_to_index: Callable,
                 feature_loss=nn.MSELoss()):
        super(EnsembleFeatureLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.count = 0
        self.ground_truth = []
        self.feature_loss = feature_loss
        self.count_to_index = count_to_index

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        for model in self.extractors:
            self.ground_truth.append(model(x))
        self.count = 0

    def __call__(self, feature: Tensor, y: Any = None) -> Tensor:
        index = self.count_to_index(self.count)
        gt = self.ground_truth[index]
        loss = self.feature_loss(feature, gt)
        self.count = self.count + 1
        return loss
