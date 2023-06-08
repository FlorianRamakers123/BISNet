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

import argparse
import distutils.util
import json
import logging
import os
import sys
import time
import glob
from datetime import datetime

import torch
import torch.distributed as dist
from monai.apps.deepedit.interaction import Interaction

from monai.apps.deepedit.transforms import (
    AddGuidanceSignalDeepEditd,
    AddRandomGuidanceDeepEditd,
    FindDiscrepancyRegionsDeepEditd,
    NormalizeLabelsInDatasetd,
    FindAllValidSlicesMissingLabelsd,
    AddInitialSeedPointMissingLabelsd,
    SplitPredsLabeld,
)
from monai.data import partition_dataset
from monai.data.dataloader import DataLoader
from monai.data.dataset import PersistentDataset, CacheDataset
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    CheckpointLoader,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.inferers import SimpleInferer
from monai.losses import DiceCELoss
from monai.networks.nets import DynUNet, UNETR
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    Resized,
    ScaleIntensityRanged,
    ToNumpyd,
    ToTensord,
    NormalizeIntensityd, EnsureTyped, RandAffined, RandGaussianNoised, RandScaleIntensityd, RandGaussianSmoothd,
    RandZoomd, SpatialPadd
)
from monai.utils import set_determinism, MetricReduction

from data_loader import get_test_loader


def get_network(labels, model_path):
    # network = DynUNet(
    #     spatial_dims=3,
    #     in_channels=len(labels) + 1,
    #     out_channels=len(labels),
    #     kernel_size=((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
    #     strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
    #     upsample_kernel_size=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
    #     norm_name="instance",
    #     deep_supervision=True,
    #     deep_supr_num=2,
    #     res_block=False,
    # )
    network = DynUNet(
        spatial_dims=3,
        in_channels=len(labels) + 1,
        out_channels=len(labels),
        kernel_size=((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        upsample_kernel_size=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=2,
        res_block=False,
    )
    network.load_state_dict(torch.load(model_path))
    return network


def get_pre_transforms(labels):
    t = [
        LoadImaged(keys=("image", "label")),
        EnsureChannelFirstd(keys=("image", "label")),
        NormalizeLabelsInDatasetd(keys="label", label_names=labels),
        Orientationd(keys=["image", "label"], axcodes="LPS"),
        Resized(keys=("image", "label"), spatial_size=(96, 96, 96)),
        NormalizeIntensityd(keys="image", subtrahend=22.6701, divisor=37.27),
        # Transforms for click simulation
        FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),
        AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids"),
        AddGuidanceSignalDeepEditd(keys="image", guidance="guidance"),
        #
        ToTensord(keys=("image", "label")),
    ]

    return Compose(t)


def get_click_transforms():
    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
        ToNumpyd(keys=("image", "label", "pred")),
        # Transforms for click simulation
        FindDiscrepancyRegionsDeepEditd(keys="label", pred="pred", discrepancy="discrepancy"),
        AddRandomGuidanceDeepEditd(
            keys="NA",
            guidance="guidance",
            discrepancy="discrepancy",
            probability="probability",
        ),
        AddGuidanceSignalDeepEditd(keys="image", guidance="guidance"),
        #
        ToTensord(keys=("image", "label")),
    ]

    return Compose(t)



def get_post_transforms():
    t = [
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys=["pred", "label"], threshold=0.5, to_onehot=[None, 2])
    ]
    return Compose(t)


def create_tester(args, num_interactions):
    set_determinism(seed=args.seed)

    multi_gpu = args.multi_gpu
    local_rank = args.local_rank
    if multi_gpu:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda:{}".format(local_rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda:0" if args.use_gpu else "cpu")

    pre_transforms = get_pre_transforms(args.labels)
    click_transforms = get_click_transforms()
    post_transform = get_post_transforms()

    # test_loader = get_loader(args, pre_transforms)
    test_loader = get_test_loader(args.input, eval_transform=pre_transforms)

    # define training components
    network = get_network(args.labels, args.model_path).to(device)
    if multi_gpu:
        network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[local_rank], output_device=local_rank)

    # if args.resume:
    #     logging.info("{}:: Loading Network...".format(local_rank))
    #     map_location = {"cuda:0": "cuda:{}".format(local_rank)}
    #     network.load_state_dict(torch.load(args.model_filepath, map_location=map_location))

    # define event-handlers for engine
    test_handlers = [
        StatsHandler(output_transform=lambda x: None),
        # TensorBoardStatsHandler(log_dir=args.output, output_transform=lambda x: None),
        # CheckpointSaver(
        #     save_dir=args.output,
        #     save_dict={"net": network},
        #     save_key_metric=True,
        #     save_final=True,
        #     save_interval=args.save_interval,
        #     final_filename="pretrained_deepedit_" + args.network + ".pt",
        # ),
    ]
    test_handlers = test_handlers if local_rank == 0 else None

    all_test_metrics = dict()
    all_test_metrics["test_mean_dice"] = MeanDice(
        output_transform=from_engine(["pred", "label"]), include_background=False, reduction=MetricReduction.NONE
    )
    # for key_label in args.labels:
    #     if key_label != "background":
    #         all_val_metrics[key_label + "_dice"] = MeanDice(
    #             output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
    #         )


    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=network,
        iteration_update=Interaction(
            deepgrow_probability=1.0,
            transforms=click_transforms,
            click_probability_key="probability",
            train=False,
            label_names=args.labels,
            max_interactions=1      # should be set to num_interactions but for num_interactions > 1 a NoneType error is raised
        ),
        inferer=SimpleInferer(),
        postprocessing=post_transform,
        key_val_metric=all_test_metrics,
        val_handlers=test_handlers,
    )

    return evaluator


def run(args):
    if args.local_rank == 0:
        for arg in vars(args):
            logging.info("USING:: {} = {}".format(arg, getattr(args, arg)))
        print("")

    if not os.path.exists(args.output):
        logging.info("output path [{}] does not exist. creating it now.".format(args.output))
        os.makedirs(args.output, exist_ok=True)

    dices = []
    for num_interactions in range(1):
        print(f"EVALUATION FOR # INTERACTIONS = {num_interactions} ----------------------------------------")
        tester = create_tester(args, num_interactions)

        start_time = time.time()
        tester.run()
        end_time = time.time()
        dices.append(tester.state.metrics["test_mean_dice"])
        logging.info("Total Testing Time {}".format(end_time - start_time))

    json_data = {
        "model_path": args.model_path,
        "dice": {i: dices[i].tolist() for i in range(len(dices))}
    }
    with open(os.path.join(args.output, "evaluation.json"), "w+") as f:
        json.dump(json_data, f)
    if args.multi_gpu:
        dist.destroy_process_group()


def strtobool(val):
    return bool(distutils.util.strtobool(val))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-mp", "--model_path")
    parser.add_argument("-n", "--network", default="dynunet", choices=["dynunet", "unetr"])
    parser.add_argument("-i", "--input", default="data/EAS/",)
    parser.add_argument("-o", "--output", default=f"deepedit_eval_output/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

    parser.add_argument("-g", "--use_gpu", type=strtobool, default="false")
    parser.add_argument("-a", "--amp", type=strtobool, default="false")

    parser.add_argument("-t", "--limit", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("-r", "--resume", type=strtobool, default="false")

    parser.add_argument("--multi_gpu", type=strtobool, default="false")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    args.spatial_size = [96, 96, 96]
    args.labels = {"anal_sphincter": 1, "background": 0}


    # Restoring previous model if resume flag is True
    # args.model_filepath = args.output + "/net_key_metric=0.8205.pt"
    run(args)


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
