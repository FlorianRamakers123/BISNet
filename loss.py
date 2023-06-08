import os
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from monai.networks.layers import GaussianFilter
from monai.transforms import ScaleIntensity, GaussianSmooth
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class RoundSTE(torch.autograd.function.Function):
    """ Round input tensor in forward pass, straight-through estimator(STE) IN BACKWARD pass """

    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class GuidanceDistanceLoss(nn.Module):
    """
    Loss function to optimize guidance distance. Intuitively, the loss should be low whenever the guidance points is close
    to the border of the prediction. To obtain a differentiable value that is proportional to this distance the guidance channel
    is multiplied with the smoothed border of the prediction. This multiplication will be maximal when all guidance points
    go through the border. The gradient of the prediction is used as an approximation for the border. The distance is only
    calculated for the foreground channel. Since a smaller distance is better, the total loss will be exponentiated. Since
    The distance is expected to decrease, the sigma parameters is decreased over time according to the following formula:
    ``sigma_new = sigma * sigma_decay^t)`` where ``sigma`` and ``sigma_decay`` correspond to the intializer parameters
    and t is the number of batches processed by this loss functions.
    """
    def __init__(self, num_classes = 2, sigma : float = 15.0, sigma_decay : float = 0.9995, device : torch.device = None):
        """
        Initializer for ``GuidanceDistanceLoss``.
        :param num_classes: The number of classes. defaults to 2.
        :param sigma: The initial sigma to use for smoothing the gradient. Defaults to 15.0.
        :param sigma_decay: The decay factor for sigma.
        :param device: The device to calculate the loss on. Defaults to None which uses cuda:0 or cpu if that's not available.
        """
        super().__init__()
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define the Sobel operator for calculating the gradient
        kernel_v = [[[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]]]
        kernel_h = [[[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]]]
        kernel_d = [[[-1], [0], [1]],
                    [[-2], [0], [2]],
                    [[-1], [0], [1]]]

        # Multi-channel kernels for output
        kernel_h_output = torch.FloatTensor(kernel_h).tile((num_classes, num_classes, 1, 1, 1)).to(device)
        kernel_v_output = torch.FloatTensor(kernel_v).tile((num_classes, num_classes, 1, 1, 1)).to(device)
        kernel_d_output = torch.FloatTensor(kernel_d).tile((num_classes, num_classes, 1, 1, 1)).to(device)
        self.weight_h_output = nn.Parameter(data=kernel_h_output, requires_grad=False)
        self.weight_v_output = nn.Parameter(data=kernel_v_output, requires_grad=False)
        self.weight_d_output = nn.Parameter(data=kernel_d_output, requires_grad=False)
        self.sigma = sigma
        self.sigma_decay = sigma_decay

    def forward(self, outputs, guidance):
        """
        Calculate the loss. Input parameters are expected to be passed before processing and should have a channel and
        batch dimension.
        """
        outputs = RoundSTE.apply(outputs)              # Make prediction discrete in a differentiable way

        # Calculate the smoothed gradient
        x_v = F.conv3d(outputs, self.weight_v_output, padding='same')
        x_h = F.conv3d(outputs, self.weight_h_output, padding='same')
        x_d = F.conv3d(outputs, self.weight_d_output, padding='same')
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + torch.pow(x_d, 2) + 1e-6)
        smoother = GaussianFilter(3, self.sigma)
        x = smoother(x)
        guidance = guidance.clone()
        guidance[guidance != 0] = 1.0
        guidance = smoother(guidance)
        reduce_axes = list(range(1, x.ndim))
        reward : torch.Tensor = (x * guidance).sum(reduce_axes) / (guidance.sum(reduce_axes) + 1)

        loss = torch.exp(-reward)
        self.sigma *= self.sigma_decay
        return loss


def save_array_as_images(arr : Union[torch.Tensor, np.ndarray], directory : str, filename_callable : callable = lambda i: f"{i}.png"):
    """
    Save a 3D numpy array or pytorch tensor as a stack of images. The images are saved to the specified directory. The file
    name of each image is defined by the `filename_callable`. This callable should take an integer corresponding to the index
    of the image as parameter and should return the appropriate file name.
    :param arr: The numpy array or pytorch tensor to save.
    :param directory: The directory to store the images in.
    :param filename_callable: The callable that defines the filename.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if len(arr.shape) != 3:
        raise RuntimeError(f"Expected array to have 3 dimensions but got {len(arr.shape)}.")
    os.makedirs(directory, exist_ok=True)
    for i in range(arr.shape[0]):
        img = Image.fromarray(((arr[i] - arr.min()) / (arr.max() - arr.min()) * 255.0).astype(np.uint8))
        img.save(os.path.join(directory, filename_callable(i)))


class WeightedDiceCELoss(nn.Module):
    """
    DiceCELoss weighted with the guidance channel. The loss works the same way as DiceCELoss but only regions close the
    guidance (as defined by a Gaussian smoothed guidance channel) are contributing to the loss. This loss stimulates the model
    to pay specific attention to regions close to the guidance point(s).
    """

    def __init__(self, sigma : float = 10.0):
        """
        Initializer for ``WeightedDiceCELoss``.
        :param sigma: The sigma to use for Gaussian smoothing the guidance channel.
        """
        super().__init__()
        self.sigma = sigma
        self.ce = CrossEntropyLoss(reduction='none')
        self.normalizer = ScaleIntensity()


    def forward(self, output, label, guidance):
        """
        Calculate the loss.
        :param output: The raw output of the network, shaped as (B,C,H,W,D)
        :param label: The label in one-hot format, shaped as (B,C,H,W,D)
        :param guidance: The guidance points, as a list of lists.
        """
        output_s = torch.softmax(output, dim=1)         # Convert output to probabilities
        smoother = GaussianFilter(3, self.sigma)
        guidance = guidance.clone()
        guidance[guidance != 0] = 1.0
        weights = smoother(guidance)

        # Calculate the weighted dice
        reduce_axis = list(range(2, len(output_s.shape)))
        intersect = (label * output_s)
        dice_numerator = torch.sum(weights * intersect, dim=reduce_axis)
        dice_denominator = torch.sum(weights * label, dim=reduce_axis) + torch.sum(weights * output_s, dim=reduce_axis)
        dice = 1.0 - (2.0 * dice_numerator) / (dice_denominator + 1e-5)
        dice = dice.mean(1)     # average dice over all channels

        # Calculate the weighted CE
        ce = self.ce(output, label.float())
        ce_weights = self.normalizer(weights.sum(1))
        reduce_axis = list(range(1, len(ce_weights.shape)))
        weighted_ce = (ce_weights * ce).sum(dim=reduce_axis) / (ce_weights.sum() + 1)

        # reduce_axes = list(range(1, guidance.ndim))
        # active = guidance.sum(reduce_axes).bool().float()  # active[i] = 1.0 if the i-th sample has a non-zero guidance channel

        loss = weighted_ce + dice
        return loss
