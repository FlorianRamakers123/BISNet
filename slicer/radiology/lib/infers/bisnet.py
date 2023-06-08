# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Sequence, Union, Dict
from ..transforms.transforms import AdjustGuidanceSpacing, EncodeBoundaryClickSignald
from monai.apps.deepedit.transforms import (
    AddGuidanceFromPointsDeepEditd,
    AddGuidanceSignalDeepEditd,
    DiscardAddGuidanced,
    ResizeGuidanceMultipleLabelDeepEditd,
)
from monai.config import KeysCollection
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    SqueezeDimd,
    ToNumpyd, Spacingd, NormalizeIntensityd, SpatialPadd, MapTransform,
)

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored



class BISNet(BasicInferTask):
    BISNET_INFER_TYPE = "BISNet"

    """
    This provides Inference Engine for pre-trained model over Multi Atlas Labeling Beyond The Cranial Vault (BTCV)
    dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        type=BISNET_INFER_TYPE,
        labels=None,
        dimension=3,
        spatial_size=(96,96,96),
        target_spacing=(1, 1, 1),
        number_intensity_ch=1,
        description="BISNet model for volumetric (3D) segmentation of 3D US Volumes",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            input_key="image",
            output_label_key="pred",
            output_json_key="result",
            **kwargs,
        )

        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.number_intensity_ch = number_intensity_ch

    def pre_transforms(self, data=None):
        t = [
            LoadImaged(keys="image", allow_missing_keys=True),
            EnsureChannelFirstd(keys="image", allow_missing_keys=True),
            Orientationd(keys="image", axcodes="LPS", allow_missing_keys=True),
            # Spacingd(keys="image", pixdim=self.target_spacing, mode='bilinear', allow_missing_keys=True),
            Resized("image", spatial_size=self.spatial_size, allow_missing_keys=True),
            NormalizeIntensityd(keys="image", subtrahend=22.67, divisor=37.28),
            SpatialPadd(keys="image", spatial_size=self.spatial_size, allow_missing_keys=True),
            # CastToTyped(keys="image", dtype=(np.float32, np.uint8), allow_missing_keys=True),
            # EnsureTyped(keys=["image", "label"], allow_missing_keys=True)
        ]

        self.add_cache_transform(t, data)

        if self.type == self.BISNET_INFER_TYPE:
            t.extend(
                [
                    EncodeBoundaryClickSignald(keys="foreground", image_key="image"),
                ]
            )
        else:
            t.extend(
                [
                    DiscardAddGuidanced(
                        keys="image", label_names=self.labels, number_intensity_ch=self.number_intensity_ch
                    ),
                ]
            )

        t.append(EnsureTyped(keys="image", device=data.get("device") if data else None))
        return t

    def inferer(self, data=None) -> Inferer:
        return SimpleInferer()

    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]
