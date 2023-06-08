import json
import os
from math import cos, pi
from typing import Union, Tuple, Mapping, Hashable, Any, Dict, Sequence, Optional, List, Callable

import FastGeodis
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, RandomizableTransform, GaussianSmooth, RandomizableTrait
from monai.transforms.utils import ndimage
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors


class AddRandomBoundaryClickSignald(RandomizableTransform, MapTransform):
    """
    Dictionary-based ``MapTransform`` for simulating boundary clicks based on the ground truth and the prediction of the model.
    Locations the minimal distance between the boundary of the prediction and ground truth is maximal have the highest probability
    of being chosen. When the prediction is not available, the boundary is uniformly sampled. The sampled point(s) will be
    stored under the specified key and as list of tuples. If the key already exists, the point will be added to the existing list.

    USAGE:
    - Be sure to apply any effect that alters the position, orientation, rotation or scale of image and/or label BEFORE this
      transform.
    - This transform must be followed by the ``EncodeBoundaryClickSignald``.
    - For full reproducibility initialize this class as ``AddRandomBoundaryClickSignald(...).set_random_state(seed=<seed>)``
      with <seed> a seed to your liking.
    - For multi-class labels, use ``SplitLabeld`` first and provide all resulting keys to this transform.
    """

    def __init__(self,
                 keys : KeysCollection,
                 guidance_key : Union[Sequence[str], str] = "guidance",
                 pred_key : str = "pred",
                 num_points : Union[int, Tuple[int, int]] = 1,
                 prob : float = 0.3,
                 sigma : float = .5,
                 error_influence : float = 0.1,
                 allow_missing_keys : bool = False):
        """
        Initializer for ``AddRandomBoundaryClickSignald``.
        :param keys: The key (collection) indicating the keys of the corresponding items to be transformed. This should contain
                     only label keys.
        :param guidance_key: The name of the key to store the guidance signal under. If this is applied to multiple keys this
                             must be a sequence matching the keys parameter.
        :param pred_key: The key that stores the current prediction of the model.
        :param num_points: The number of points to sample. Can also be a tuple (a,b) which results in a random number of
                           points (minimum a, maximum b) being sampled. Defaults to 1.
        :param prob: The probability of applying this transform. Defaults to 0.3.
        :param sigma: The sigma parameter to use for sampling the point.
        :param error_influence: The influence of the error of the prediction. The higher this value the more the error of the
                                prediction will be taken into account. A value close to zero will result in a nearly uniform
                                sampling of the border of the ground truth. Settings this value too high might result in
                                the same boundary point being sampled multiple times. Defaults to 0.1.
        :param allow_missing_keys: If set to False, an exception will be raised if one of the keys specified by the ``keys``
                                   parameter is not in the provided dictionary.
        """
        MapTransform.__init__(self,keys=keys, allow_missing_keys=allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.guidance_key = [guidance_key] * len(self.keys) if isinstance(guidance_key, str) else guidance_key
        self.pred_key = pred_key
        self.guidance_meta_key = f"{self.guidance_key}_meta_dict"
        self.sigma = sigma
        self.error_influence = max(0.00000001, error_influence)      # make sure error_influence > 0
        self.num_points = num_points

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, any]:
        """
        Apply the transform.
        """
        d: Dict = dict(data)
        self.randomize(d)          # sets self._do_transform which indicates whether to apply this transform

        num_points = self.num_points if isinstance(self.num_points, int) else self.R.randint(self.num_points[0], self.num_points[1] + 1)       # Use self.R for reproducibility

        if not self._do_transform or num_points == 0:
            for _, g_key in self.key_iterator(d, self.guidance_key):
                d[g_key] = []
            return d

        for key_label, (key, g_key) in enumerate(self.key_iterator(d, self.guidance_key), start=1):  # iterate through all label keys and guidance keys
            boundary_clicks = d.get(g_key, [])             # get the existing guidance points
            if len(boundary_clicks) > num_points:
                boundary_clicks = boundary_clicks[:num_points]
            elif len(boundary_clicks) < num_points:
                boundary_clicks = self._sample_points(d[key][0], boundary_clicks, num_points - len(boundary_clicks), d.get(self.pred_key, None), key_label)
            d[g_key] = boundary_clicks


        return d

    def _sample_points(self, ground_truth : torch.Tensor, curr_points : List[Tuple[int, int, int]], num_points: int, curr_prediction : Optional[torch.Tensor], key_label : Optional[int]) -> List[Tuple[int, int, int]]:
        """
        Sample the given amount of points, given the ground truth. If a previous prediction of the model is available
        the sampling will be based on this previous prediction. Otherwise boundary points are uniformly sampled.
        :param ground_truth: The ground truth with shape (C,H,W,D).
        :param num_points: The number of points to sample.
        :param curr_prediction: The prediction of the model. Specified as a segmentation mask in one-hot encoding of
                                shape (K,H,W,D) with K the number of classes.
        :param key_label: The class label to index the current prediction.
        :return: A list of tuples of integers representing a list of points. A tuple (x_0, x_1, ..., x_k) is ordered
        such that x_i is the index of the i-th dimension.
        """

        # Calculate the border of the ground truth
        label = ground_truth.numpy()
        struct = ndimage.generate_binary_structure(3, 1)
        eroded_label = ndimage.binary_erosion(label, structure=struct, iterations=1).astype(np.uint8)
        label_border = label - eroded_label

        # Set the density map to the label border for now
        density_map = label_border.astype(np.float)

        # If there is a prediction given, calculate for each border point the minimal distance to the prediction
        if curr_prediction is not None and key_label is not None and curr_prediction[key_label].max() > 0:
            # Calculate the border of the prediction
            pred = curr_prediction[key_label].detach().cpu().numpy()
            eroded_pred = ndimage.binary_erosion(pred, structure=struct, iterations=1).astype(np.uint8)
            pred_border = pred - eroded_pred

            # Fit a 1-Nearest-Neighbours algorithm to find the minimum distances
            z_pred, y_pred, x_pred = (pred_border != 0).nonzero()       # list of prediction border points
            data_pred = np.array(list(zip(x_pred, y_pred, z_pred)))
            nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', p=2).fit(data_pred)
            z_label, y_label, x_label = (label_border != 0).nonzero()     # list of ground truth border points
            data_label = np.array(list(zip(x_label, y_label, z_label)))
            min_dist = nn.kneighbors(data_label)[0]

            # Update the density map
            for i, t in enumerate(zip(z_label, y_label, x_label)):
                density_map[t] = min_dist[i][0]
            with np.errstate(over="raise"):
                try:
                    density_map = np.exp((self.error_influence * density_map).astype(np.float64)) - 1.0      # -1.0 to make sure min(density_map) = 0
                except FloatingPointError:
                    print("OVERFLOW ENCOUNTERED!!! max value =", density_map.max())


        # density_map = torch.from_numpy(gaussian_filter(density_map, sigma=self.sigma))      # Apply Gaussian smoothing to simulate user variability
        density_map = torch.from_numpy(density_map)
        points = self._sample(density_map, curr_points, num_points)
        for point in points:
            if density_map[point] == 0:
                raise RuntimeError(f"Chose point {point} but density was 0.")
        return points

    def _sample(self, density_map, points, num_points) -> List[Tuple[int, int, int]]:
        # Sample the points according to the density map
        c = (torch.flatten(density_map) / density_map.sum()).cumsum(0)
        old_len = len(points)
        while len(points) < old_len + num_points:
            ra = self.R.rand(1)
            i = torch.searchsorted(c, torch.from_numpy(ra))
            out = []
            for dim in reversed(density_map.shape):
                out.append((i % dim).item())
                i = i // dim
            point = tuple(reversed(out))
            if point not in points: # eliminate duplicates
                points.append(point)
        # print(points)
        return points

class EncodeBoundaryClickSignald(MapTransform):
    """
    Dictionary-based ``MapTransform`` for converting the explicit boundary clicks to implicit guidance tensors and appending
    it as a new channel to the input image. The implicit tensor can have different encodings and can mix in image features.

    USAGE:
    - Be sure to apply any effect that alters the position, orientation, rotation or scale of image and/or label BEFORE this
      transform.
    - This transform must come after ``AddRandomBoundaryClickSignald``.
    """

    def __init__(self, keys : KeysCollection, image_key : str, pred_key : str, encoding : Union[str, Sequence[str]], mix_image_features : bool = True, tau : float = 10.0, sigma : float = 5.0, scaling : float = 0.5, differ_inward_points : bool = False, allow_missing_keys : bool = False):
        """
        Initializer for ``EncodeBoundaryClickSignald``.
        :param keys: The key (collection) indicating the keys of the corresponding items to be transformed. This should contain
                     only guidance keys.
        :param image_key: The key that stores the image.
        :param pred_key: The key that stores the prediction.
        :param encoding: The type of encoding to apply. Can be 'gaussian' or 'inverse_euclidean'. If multiple encoding are to be
                         used, you need to specify a list of encodings matching the keys.
        :param mix_image_features: Bool indicating whether to mix in image features. This will result in a multiplication of the
                                   guidance tensor with the image.
        :param tau: The scaling factor for the geodesic distance.
        :param sigma: The standard deviation to use for Gaussian smoothing when ``encoding`` is set to 'gaussian'.
        :param scaling: The scaling to use when ``encoding`` is set to 'inverse_euclidean'
        :param differ_inward_points: Whether to use an additional channel for inward points. The additional channel will use the same
                                     encoding.
        :param allow_missing_keys: If set to False, an exception will be raised if one of the keys specified by the ``keys``
                                   parameter is not in the provided dictionary.
        """
        super().__init__(keys, allow_missing_keys)
        self.encodings = [encoding] * len(self.keys) if isinstance(encoding, str) else encoding
        self.image_key = image_key
        self.pred_key = pred_key
        self.mix_image_features = mix_image_features
        self.sigma = sigma
        self.tau = tau
        self.scaling = scaling
        self.gaussian_smoothing = GaussianSmooth(self.sigma)
        self.differ_inward_points = differ_inward_points

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, any]:
        d: Dict = dict(data)

        # Check if the image key is present
        if self.image_key not in d:
            raise KeyError(f"Image key '{self.image_key}' was not present in given data.")

        # Check if the prediction key is present
        # if self.pred_key not in d:
        #     raise KeyError(f"Prediction key '{self.pred_key}' was not present in given data.")

        num_guidance_channels = (2 if self.differ_inward_points else 1) * len(self.keys)
        guidance_tensor = torch.zeros(num_guidance_channels, *d[self.image_key].shape[-3:], dtype=torch.float)
        channel = 0
        for key, encoding in self.key_iterator(data, self.encodings):
            points = d[key]

            if len(points) > 0:
                # Gaussian smoothing
                if encoding == "gaussian":
                    if self.differ_inward_points:
                        for point in points:
                            if d[self.pred_key][1][point]:         # TODO: remove hardcoded 1 in index
                                guidance_tensor[channel+1][point] = 1.0
                            else:
                                guidance_tensor[channel][point] = 1.0
                        guidance_tensor[channel+1] = self.gaussian_smoothing(guidance_tensor[channel+1])
                    else:
                        for point in points:
                           guidance_tensor[channel][point] = 1.0
                    guidance_tensor[channel] = self.gaussian_smoothing(guidance_tensor[channel])

                # Inverse Euclidean
                elif encoding == "inverse_euclidean":
                    r1 = torch.arange(0, guidance_tensor.shape[-3])
                    r2 = torch.arange(0, guidance_tensor.shape[-2])
                    r3 = torch.arange(0, guidance_tensor.shape[-1])
                    x1, x2, x3 = torch.meshgrid(r1, r2, r3)
                    guidance_tensor[channel] = torch.tensor(torch.inf)
                    if self.differ_inward_points:
                        guidance_tensor[channel+1] = torch.tensor(torch.inf)
                    for point in points:
                        euclid_distance = torch.sqrt(torch.pow(x1 - point[0], 2) + torch.pow(x2 - point[1], 2) + torch.pow(x3 - point[2], 2))
                        if self.differ_inward_points and d[self.pred_key][1][point]:         # TODO: remove hardcoded 1 in index
                            guidance_tensor[channel+1] = torch.minimum(euclid_distance, guidance_tensor[channel+1])
                        else:
                            guidance_tensor[channel] = torch.minimum(euclid_distance, guidance_tensor[channel])
                    if (guidance_tensor[channel] != torch.inf).all():
                        guidance_tensor[channel] = 1.0 / (self.scaling * guidance_tensor[channel] + 1.1)
                        guidance_tensor[channel] = (guidance_tensor[channel] - guidance_tensor[channel].min()) / (guidance_tensor[channel].max() - guidance_tensor[channel].min())
                    else:
                        guidance_tensor[channel].zero_()
                    if self.differ_inward_points:
                        if (guidance_tensor[channel+1] != torch.inf).all():
                            guidance_tensor[channel+1] = 1.0 / (self.scaling * guidance_tensor[channel+1] + 1.1)
                            guidance_tensor[channel+1] = (guidance_tensor[channel+1] - guidance_tensor[channel+1].min()) / (guidance_tensor[channel+1].max() - guidance_tensor[channel+1].min())
                        else:
                            guidance_tensor[channel+1].zero_()
                elif encoding == "geodesic":
                    for point in points:
                        guidance_tensor[channel][point] = 1.0
                    geodesic = FastGeodis.generalised_geodesic3d(self.gaussian_smoothing(d[self.image_key].squeeze(0)).unsqueeze(0).unsqueeze(0), 1 - guidance_tensor[channel].unsqueeze(0).unsqueeze(0), spacing=[1.0,1.0,1.0], v=1e9, lamb=1.0, iter=4)
                    guidance_tensor[channel] = torch.exp(-geodesic / self.tau).squeeze(0).squeeze(0)
                    guidance_tensor[channel] = (guidance_tensor[channel] - guidance_tensor[channel].min()) / (guidance_tensor[channel].max() - guidance_tensor[channel].min())
                    plt.imshow(guidance_tensor[channel][points[0][0]].detach().cpu().numpy(), cmap="gray")
                    plt.title(str(points[0]))
                    plt.show()
                else:
                    raise ValueError(f"Unsupported encoding: {encoding}")


            # Mix in the image features if specified
            # if self.mix_image_features:
            #     guidance_tensor *= d[self.image_key]

            channel += 2 if self.differ_inward_points else 1
        d[self.image_key] = torch.cat([d[self.image_key], guidance_tensor])
        return d


class ToImplicitGuidanced(MapTransform):
    """
    Converts an list of explicit guidance points to an implicit guidance tensor. The tensor has the shame
    shape as the input volume and contains the point number on the coordinates of the guidance tensor that equals the
    """

    def __init__(self, keys: KeysCollection, shape : Tuple[int, int, int], new_key_name: str = "guidance", allow_missing_keys: bool = False):
        """
        Initializer for ``ToImplicitGuidanced``.
        :param keys: The key (collection) indicating the keys of the corresponding items to be transformed. This should contain
                     all the guidance keys that need to be combined.
        :param shape: The shape of the tensor. Only the volumetric shape should be specified. The channel dimensions is inferred
                      from the amount of keys.
        :param new_key_name: The name of the key to store the guidance tensor under. Defaults to "guidance".
        :param allow_missing_keys: If set to False, an exception will be raised if one of the keys specified by the ``keys``
                                   parameter is not in the provided dictionary.
        """
        super().__init__(keys, allow_missing_keys)
        self.shape = shape
        self.new_key_name = new_key_name

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, any]:
        d: Dict = dict(data)
        guidance_tensor = torch.zeros((len(self.keys), *self.shape))

        for key, channel in self.key_iterator(d, range(len(self.keys))):
            for i, point in enumerate(d[key], 1):
                guidance_tensor[channel][point] = float(i)
            d.pop(key)

        d[self.new_key_name] = guidance_tensor
        return d


class SaveGuidanced(MapTransform, RandomizableTrait):   # Must implement RandomizableTrait otherwise this might be skipped when CacheDataset is used
    """
    Saves part of the given mapping specified by the keys to a file. The resulting .gh file will contain different dictionaries
    separated by a ';' such that the i-th dictionary corresponds to the i-th iteration.

    Example:
         SaveGuidanced(keys="guidance", image_meta_dict_key=<image_meta_dict_key>, output_folder=<output_folder>)

         iteration 1: { "guidance": [(1,2,3)], <image_meta_dict_key>: {"filename_or_obj": "image1.nii.gz" }}
         iteration 2: { "guidance": [(4,5,6)], <image_meta_dict_key>: {"filename_or_obj": "image1.nii.gz" }}

         Results in '<output_folder>/image1.gh':
            { "guidance": [(1,2,3)] };
            { "guidance": [(4,5,6)] };

    USAGE:
    - This transform must be used BEFORE ``EncodeBoundaryClickSignald`` and AFTER ``AddRandomBoundaryClickSignald``.
    """

    def __init__(self, keys: KeysCollection, image_meta_dict_key : str, output_folder : str, allow_missing_keys : bool = False):
        """
        Initializer for ``SaveGuidanced``.
        :param keys: The key (collection) indicating the keys of the corresponding items to be transformed. This should contain
                     only guidance keys.
        :param image_meta_dict_key: The dictionary key containing the metadata of the image.
        :param output_folder: The output folder to store the guidance files in.
        :param allow_missing_keys: If set to False, an exception will be raised if one of the keys specified by the ``keys``
                                   parameter is not in the provided dictionary.
        """
        super().__init__(keys, allow_missing_keys)
        self.image_meta_dict_key = image_meta_dict_key
        self.output_folder = output_folder

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, any]:
        d: Dict = dict(data)
        file_name =  os.path.basename(os.path.normpath(d[self.image_meta_dict_key]["filename_or_obj"]))
        new_file_name = file_name.replace(".nii.gz", f".gh")

        guidance_dict = {}
        for key in self.key_iterator(d):
            guidance_dict[key] = d[key]
        with open(os.path.join(self.output_folder, new_file_name), 'a+') as f:
            f.write(json.dumps(guidance_dict) + ";\n")
        return d


class LoadGuidanced(MapTransform, RandomizableTrait):  # Must implement RandomizableTrait otherwise this might be skipped when CacheDataset is used
    """
    Loads in the guidance signal that was saved by ``SaveGuidanced``. All keys in the specified .gh file will be added to
    the dictionary and associated to the LAST entry in the file. If the specified file does not exist this transform does nothing.
    Example:
        LoadGuidanced(keys="guidance")

        input dictionary:
            { "guidance": <output_folder>/image1.gh }

        <output_folder>/image1.gh:
            { "guidance1": [(1,2,3)], "guidance2": [(4,5,6)] };

        output dictionary:
            {
                "guidance": <output_folder>/image1.gh,
                "guidance1": [(1,2,3)],
                "guidance2": [(4,5,6)]
            }

    USAGE:
    - This transform must be used BEFORE ``EncodeBoundaryClickSignald`` and ``AddRandomBoundaryClickSignald``.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        """
        Initializer for ``LoadGuidanced``.
        :param keys: The key (collection) indicating the keys of the corresponding items to be transformed. This should contain
                     only guidance keys.
        :param allow_missing_keys: If set to False, an exception will be raised if one of the keys specified by the ``keys``
                                   parameter is not in the provided dictionary.
        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, any]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if os.path.exists(d[key]):
                with open(d[key], 'r') as f:
                    guidance_history = f.read().split(";")
                    if len(guidance_history) > 1:
                        guidance_dict = json.loads(guidance_history[-2])    # The last element will be "" since there will be and ending ';'
                        for gkey in guidance_dict:
                            d[gkey] = [tuple(x) for x in guidance_dict[gkey]]       # json will convert tuples to list so we need to convert them back
        return d


class DropoutGuidanced(RandomizableTransform, MapTransform):
    """
    Base class for dictionary-based ``Transform`` to simulate a dropout of the guidance channel. Dropping
    out the guidance channel is achieved by setting the guidance channel tensor to zero.
    """
    def __init__(self, keys : KeysCollection, allow_missing_keys : bool = False):
        """
        Initializer for ``DropoutGuidanced``.
        :param keys: The key (collection) indicating the keys of the corresponding items to be transformed. This should contain
                     only guidance keys.
        :param allow_missing_keys: If set to False, an exception will be raised if one of the keys specified by the ``keys``
                                   parameter is not in the provided dictionary.
        """
        MapTransform.__init__(self, keys, allow_missing_keys=allow_missing_keys)
        RandomizableTransform.__init__(self)

    def calculate_prob(self, data: Mapping[Hashable, torch.Tensor]):
        """
        Calculate the probability of dropping out the guidance.
        :param data: The data to work with.
        :return: The probability of dropping out the guidance.
        """
        raise NotImplementedError()

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        """
        Perform the dropout.
        :param data: The data to work with.
        :return: A dictionary that was given, which possibly the guidance channel set to zero.
        """
        d: Dict = dict(data)
        self.randomize(d)  # sets self._do_transform which indicates whether to apply this transform
        for key in self.key_iterator(d):
            chance = self.calculate_prob(data)
            p = self.R.rand()
            if chance >= p:
                d[key].clear()
        return d

class DecayingDropoutGuidanced(DropoutGuidanced):
    """
    Dictionary-based ``Transform`` to simulate a dropout of the guidance channel. The probability of dropping out the
    guidance p(x) decreases as b^(alpha(x-cold_start)) with x the number of samples processed by this ``Transform``. If
    you set alpha = 20.0 and b = 0.99 you get a binary dropout mechanism. Dropping out the guidance channel is achieved
    by setting the guidance channel tensor to zero.
    """
    def __init__(self, keys : KeysCollection, b : float = 0.99, alpha : float = 0.1, cold_start : int = 2760, allow_missing_keys : bool = False):
        """
        Initializer for ``DecayingDropoutGuidanced``.
        :param keys: The key (collection) indicating the keys of the corresponding items to be transformed. This should contain
                     only guidance keys.
        :param b: The base of the probability formula. Defaults to 0.99
        :param alpha: The weight of the exponent in the probability formula. Defaults to 0.1.
        :param cold_start: The initial amount of samples to process with a 100% chance of dropping out the guidance channel.
                           Defaults to 2760.
        :param allow_missing_keys: If set to False, an exception will be raised if one of the keys specified by the ``keys``
                                   parameter is not in the provided dictionary.
        """
        MapTransform.__init__(self, keys, allow_missing_keys=allow_missing_keys)
        RandomizableTransform.__init__(self)
        self.b = b
        self.alpha = alpha
        self.cold_start = cold_start
        self.count = 0

    def calculate_prob(self, data: Mapping[Hashable, torch.Tensor]):
        """
        Calculate the probability of dropping out the guidance.
        :param data: The data to work with.
        :return: The probability of dropping out the guidance.
        """
        return self.b**(self.alpha * (self.count - self.cold_start))

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        self.count += 1
        return super().__call__(data)


class DiceBasedDropoutGuidanced(DropoutGuidanced):
    """
    Dictionary-based ``Transform`` to simulate a dropout of the guidance channel. The probability of dropping out the
    guidance p(x) drops to zero whenever the Dice score reaches a certain threshold. Dropping out the guidance channel is achieved
    by setting the guidance channel tensor to zero.
    """
    def __init__(self, keys: KeysCollection, dice_threshold: float = 0.6, dice_key="dice",
                 allow_missing_keys: bool = False):
        """
        Initializer for ``DecayingDropoutGuidanced``.
        :param keys: The key (collection) indicating the keys of the corresponding items to be transformed. This should contain
                     only guidance keys.
        :param dice_threshold: The threshold for the dice score. If the dice score under the `dice_key` is higher than this
                               value the probability of dropping the guidance drops to zero.
        :param dice_key: The key that stores the dice score. Defaults to "dice".
        :param allow_missing_keys: If set to False, an exception will be raised if one of the keys specified by the ``keys``
                                   parameter is not in the provided dictionary.
        """
        super().__init__(keys, allow_missing_keys)
        self.dice_threshold = dice_threshold
        self.dice_key = dice_key

    def calculate_prob(self, data: Mapping[Hashable, torch.Tensor]):
        """
        Calculate the probability of dropping out the guidance.
        :param data: The data to work with.
        :return: The probability of dropping out the guidance.
        """
        return 1.0 if data[self.dice_key] < self.dice_threshold else 0.0


class CosineAnnealingRestartDropoutGuidanced(DropoutGuidanced):
    """
    Dictionary-based ``Transform`` to simulate a dropout of the guidance channel. The probability of drops according to a
    cosine. Every x epochs the probability is reset. Dropping out the guidance channel is achieved
    by setting the guidance channel tensor to zero.
    """
    def __init__(self, keys: KeysCollection, max_prob : float = 1.0, min_prob : float = 0.0, restart : int = 2730, max_decay : float = 0.0, min_decay : float = 0.0, allow_missing_keys: bool = False):
        """
        Initializer for ``DecayingDropoutGuidanced``.
        :param keys: The key (collection) indicating the keys of the corresponding items to be transformed. This should contain
                     only guidance keys.
        :param max_prob: The maximum probability. Defaults to 1.0.
        :param min_prob: the minimum probability. Defaults to 0.0.
        :param restart: The amount of samples to process before doing a restart.
        :param max_decay: The amount with which the maximum probability decays per sample.
        :param max_decay: The amount with which the minimum probability decays per sample.
        :param allow_missing_keys: If set to False, an exception will be raised if one of the keys specified by the ``keys``
                                   parameter is not in the provided dictionary.
        """
        super().__init__(keys, allow_missing_keys)
        self.max_prob = max_prob
        self.min_prob = min_prob
        self.restart = restart
        self.max_decay = max_decay
        self.min_decay = min_decay
        self.count = 0

    def calculate_prob(self, data: Mapping[Hashable, torch.Tensor]):
        """
        Calculate the probability of dropping out the guidance.
        :param data: The data to work with.
        :return: The probability of dropping out the guidance.
        """
        return min(1.0, max(0.0, self.min_prob + 0.5 * (self.max_prob - self.min_prob) * (1 + cos(self.count / self.restart * pi))))


    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        result = super().__call__(data)
        self.count += 1
        if self.count % self.restart == 0:
            self.count = 0
            self.max_prob = max(0.0, self.max_prob - self.max_decay)
            self.min_prob = max(0.0, self.min_prob - self.min_decay)
        return result


