import glob
import os
import torch

from monai.data import CacheDataset, list_data_collate
from monai.transforms import Transform
from torch.utils.data import DataLoader


def get_train_loader(
        root : str,
        image_folder : str ="imagesTr",
        labels_folder : str = "labelsTr",
        batch_size : int = 1,
        shuffle : bool = True,
        train_transform : Transform = None,
        ) -> DataLoader:
    """
    Load a NIftI dataset for segmentation. The root folder of the dataset must contain two folders: one for the image
    data and one for the segmentation labels. Corresponding images and labels must have the same ordering. All files must
    have the .nni.gz file extension.

    :param root: The root folder of the dataset. This folder should contain a folder for the images and for the labels.
    :param image_folder: The folder within the root folder that holds the image data. Defaults to 'image'.
    :param labels_folder: The folder within the root folder that holds the labels. Defaults to 'label'.
    :param batch_size: The batch size to use. Defaults to 1.
    :param shuffle: Boolean indicating whether to shuffle the data. Defaults to True.
    :param train_transform: The transform to use for the train set. Since a dictionary representation is used, only
                            dictionary transforms (those that end with 'd') can be used.
    :return: A DataLoader object that represent the train set.
    """
    images = sorted(glob.glob(os.path.normpath(os.path.join(root, image_folder, "*.nii.gz"))))[:3]
    labels = sorted(glob.glob(os.path.normpath(os.path.join(root, labels_folder, "*.nii.gz"))))[:3]
    dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]
    dataset = CacheDataset(data=dicts, transform=train_transform, num_workers=0)        # Caches all transforms until randomized transform is reached
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())
    return loader


def get_validation_loader(
        root: str,
        image_folder: str = "imagesVal",
        labels_folder: str = "labelsVal",
        batch_size: int = 1,
        eval_transform: Transform = None) -> DataLoader:
    """
    Load a NIftI dataset for segmentation. The root folder of the dataset must contain two folders: one for the image
    data and one for the segmentation labels. Corresponding images and labels must have the same ordering. All files must
    have the .nni.gz file extension.

    :param root: The root folder of the dataset. This folder should contain a folder for the images and for the labels.
    :param image_folder: The folder within the root folder that holds the image data. Defaults to 'image'.
    :param labels_folder: The folder within the root folder that holds the labels. Defaults to 'label'.
    :param batch_size: The batch size to use. Defaults to 1.
    :param eval_transform: The transform to use for the validation set. Since a dictionary representation is used, only
                           dictionary transforms (those that end with 'd') can be used.
    :return: A DataLoader object that represent the validation set.
    """

    images = sorted(glob.glob(os.path.join(root, image_folder, "*.nii.gz")))[:3]
    labels = sorted(glob.glob(os.path.join(root, labels_folder, "*.nii.gz")))[:3]
    dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]
    dataset = CacheDataset(data=dicts, transform=eval_transform, num_workers=0)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())
    return loader


def get_test_loader(
        root: str,
        image_folder: str = "imagesTs",
        labels_folder: str = "labelsTs",
        guidance_folder : str = None,
        batch_size: int = 1,
        eval_transform: Transform = None) -> DataLoader:
    """
    Load a NIftI dataset for segmentation. The root folder of the dataset must contain two folders: one for the image
    data and one for the segmentation labels. Corresponding images and labels must have the same ordering. All files must
    have the .nni.gz file extension. If the ``guidance_folder`` is provided the path to the guidance files (.gh) are also added
    under the 'guidance' key. Pass this key to the ``LoadGuidanced`` transform if you want to extend on previous guidance.

    :param root: The root folder of the dataset. This folder should contain a folder for the images and for the labels.
    :param image_folder: The folder within the root folder that holds the image data. Defaults to 'image'.
    :param labels_folder: The folder within the root folder that holds the labels. Defaults to 'label'.
    :param guidance_folder: The folder within the root folder that holds the guidance files. Defaults to None which does not load
                         previous guidance.
    :param batch_size: The batch size to use. Defaults to 1.
    :param eval_transform: The transform to use for the test set. Since a dictionary representation is used, only
                           dictionary transforms (those that end with 'd') can be used.
    :return: A DataLoader object that represent the test set.
    """

    images = sorted(glob.glob(os.path.join(root, image_folder, "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(root, labels_folder, "*.nii.gz")))
    if guidance_folder:
        guidances = [os.path.join(guidance_folder, os.path.basename(img_path).replace(".nii.gz", ".gh")) for img_path in images]
        dicts = [{"image": i, "label": l, "guidance" : g} for i, l, g in zip(images, labels, guidances)]
    else:
        dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]
    dataset = CacheDataset(data=dicts, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())
    return loader
