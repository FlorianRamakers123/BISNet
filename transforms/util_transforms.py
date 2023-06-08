from typing import Mapping, Hashable, Any, Dict, Optional, Sequence, Tuple

import torch
from monai.config import KeysCollection
from monai.metrics import compute_dice
from monai.networks import one_hot
from monai.transforms import MapTransform


class SplitLabelsd(MapTransform):
    """
    Splits a label into multiple labels. A label of k classes with discrete values [0, 1, ..., k-1] will be converted into
    k different binary masks. Each mask is added under a new key.
    """

    def __init__(self, keys: KeysCollection, label_names : Dict[str, int], key_names : Optional[Sequence[str]] = None, allow_missing_keys : bool = False):
        """
        Initializer for ``SplitLabelsd``.
        :param keys: The key (collection) indicating the keys of the corresponding items to be transformed. This should contain
                     only ONE label key.
        :param label_names: A dictionary associating a class name to the class value in the label. For example for a label [[0,1,2],[2,1,0]]
                            this would be { "background": 0, "class1": 1, "class2": 2 }.
        :param key_names: A list of key names to store different label masks under. This must match the order of the keys in ``label_names``.
                          Defaults to None which uses the key ``label_<label_name>`` for each ``label_name`` in ``label_names``.
        :param allow_missing_keys: If set to False, an exception will be raised if one of the keys specified by the ``keys``
                                   parameter is not in the provided dictionary.
        """
        super().__init__(keys, allow_missing_keys)
        if len(self.keys) > 1:
            raise ValueError("This transform only supports one label key at a time.")
        self.label_names = label_names
        self.key_names = key_names if key_names is not None else [f"label_{k}" for k in label_names]

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, any]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            for key_name, label_key in zip(self.key_names, self.label_names):   # key_name = new dictionary key for label, label_key = name of class
                d[key_name] = (d[key] == self.label_names[label_key]).byte()

        return d

class CombineLabelsd(MapTransform):
    """
    Combines multiple binary masks into a one-hot encoded multi-class label.
    """

    def __init__(self, keys: KeysCollection, new_key_name : str = "label", allow_missing_keys: bool = False):
        """
        Initializer for ``CombineLabelsd``.
        :param keys: The key (collection) indicating the keys of the corresponding items to be transformed. This should contain
                     all the label keys that need to be combined.
        :param new_key_name: The name of the key to store the combined label under. Defaults to "label".
        :param allow_missing_keys: If set to False, an exception will be raised if one of the keys specified by the ``keys``
                                   parameter is not in the provided dictionary.
        """
        super().__init__(keys, allow_missing_keys)
        self.new_key_name = new_key_name

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, any]:
        d: Dict = dict(data)
        label = torch.cat([d[key] for key in self.key_iterator(d)])
        background = 1 - label
        d[self.new_key_name] = torch.cat([background, label])
        for key in self.key_iterator(d):
            d.pop(key)
        return d


class UndoLabelAndGuidanceTupled(MapTransform):
    """
    Postprocessing transform for splitting the (label, guidance) tuple that was created in order to support loss functions
    that make use of the guidance points.
    """

    def __init__(self, keys: KeysCollection, new_key_names : Tuple[str, str] = ("label", "guidance"), allow_missing_keys : bool = False):
        """
        Initializer for ``UndoLabelAndGuidanceTupled``.
        :param keys: The key (collection) indicating the keys of the corresponding items to be transformed. This should contain
                     only ONE label key.
        :param new_key_names: A tuple (label_key, guidance_key) that describes the names for the keys for the label and guidance.
                              Defaults to ("label", "guidance")
        :param allow_missing_keys: If set to False, an exception will be raised if one of the keys specified by the ``keys``
                                   parameter is not in the provided dictionary.
        """
        super().__init__(keys, allow_missing_keys)
        if len(self.keys) > 1:
            raise ValueError("This transform only supports one label key at a time.")
        self.new_key_names = new_key_names

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, any]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            d[self.new_key_names[0]], d[self.new_key_names[1]] = d[key]

        return d

class DeleteDeepSupervision(MapTransform):
    """
    Transform to delete the extra tensor channels from the predicate that the deep supervision mechanism adds.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys : bool = False):
        """
        Initializer for ``UndoLabelAndGuidanceTupled``.
        :param keys: The key (collection) indicating the keys of the corresponding items to be transformed. This should contain
                     only predicate keys.
        :param allow_missing_keys: If set to False, an exception will be raised if one of the keys specified by the ``keys``
                                   parameter is not in the provided dictionary.
        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, any]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if len(d[key].shape) > 4:     # during validation there is no deep supervision
                d[key] = torch.unbind(d[key])[0]
        return d

class CalculateDiceScored(MapTransform):
    """
    Transform to calculate the dice scores.
    """

    def __init__(self, keys: KeysCollection, num_classes : int, dice_key_name : str = "dice", allow_missing_keys : bool = False):
        """
        Initializer for ``UndoLabelAndGuidanceTupled``.
        :param keys: The key (collection) indicating the keys of the corresponding items to be transformed. This should contain
                     first the predicate key and then the label key.
        :param num_classes: The number of classes.
        :param dice_key_name: The key to store the Dice score under. Defaults to 'Dice'.
        :param allow_missing_keys: If set to False, an exception will be raised if one of the keys specified by the ``keys``
                                   parameter is not in the provided dictionary.
        """
        super().__init__(keys, allow_missing_keys)
        self.num_classes = num_classes
        self.dice_key_name = dice_key_name

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, any]:
        d: Dict = dict(data)
        pred_key, label_key = self.key_iterator(data)
        dice = compute_dice(d[pred_key].unsqueeze(0), one_hot(d[label_key].unsqueeze(0), self.num_classes).to(d[pred_key].device), include_background=False)
        d[self.dice_key_name] = dice[0]
        return d
