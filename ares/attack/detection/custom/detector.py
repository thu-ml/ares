import torch
import torch.nn as nn


class DataPreprocessor(nn.Module):
    """
    If you have finished all preprocess of your input images and annotations in your dataloader,
    the batch_data['inputs'] should be batched with shape [N, C, H, W], and batch_data['data_samples'] should be a list with
    length N. If so, just return the batch_data directly in the forward function. If not, you can realize your preprocess here.
    The mean and std are required. They will be used to denormalize image tensors as the input of adversarial attack method.
    If you do not normalize input images in your detection pipeline, just fill mean with [0,0,0] and std with [1,1,1].
    Args:
        mean (list or torch.Tensor): Image mean used to normalize images. Length: 3.
        std (list or torch.Tensor): Image std used to normalize images. Length: 3.
    """
    def __init__(self, mean, std, *args, **kwargs):

        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    def forward(self, batch_data):
        '''Just return the input if batch_data['inputs'] is batched image tensor.'''
        return batch_data

class CustomDetector(nn.Module):
    def __init__(self, detector, mean, std, *args, **kwargs):
        self.detector = detector
        self.data_preprocessor = DataPreprocessor(mean=mean, std=std)


    def loss(self, batch_data):
        """Loss function used to compute detection loss.

        Args:
            batch_data (dict): Dict with two keys: inputs and data_samples, where inputs are input images (batched image tensor or list of image tensor),
                data_samples are a list of each sample annotation. We use mmdet.structure.DetDataSample to represent sample
                annotation. Please make sure that your sample annotation uses mmdet.structure.DetDataSample.
                To obtain batch_data, you may need define a collate_fn and pass it to your dataloader (torch.utils.data.Dataloader).

        Returns:
            dict: A dict with loss names as keys, e.g., {'loss_bboxes':..., 'loss_cls':...}
        """
        batch_data = self.data_preprocessor(batch_data)
        return self.detector(batch_data) # or return self.detector(batch_data['inputs'], batch_data['data_samples'])


    def predict(self, batch_data):
        """Predict function used to predict bboxes on input images.

        Args:
            batch_data (dict): Dict with two keys: inputs and data_samples, where inputs are input images (batched image tensor or list of image tensor),
                data_samples are a list of each sample annotation. We use mmdet.structure.DetDataSample to represent sample
                annotation. Please make sure that your sample annotation uses mmdet.structure.DetDataSample.
                To obtain batch_data, you may need define a collate_fn and pass it to your dataloader (torch.utils.data.Dataloader).


        Returns:
            list: List of mmdet.structure.DetDataSample where each element should be added a pred_instances attribute.
            Like gt_instances in it, pred_instances should have keys: bboxes and labels. The predicted bboxes coornidates should be scaled to
            the orginal image size.
        """
        batch_data = self.data_preprocessor(batch_data)
        return self.detector(batch_data) # or return self.detector(batch_data['inputs'], batch_data['data_samples'])



