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
    NormalizeIntensityd, RandAffined, RandGaussianNoised, RandScaleIntensityd, RandGaussianSmoothd, RandRicianNoised,
    RandZoomd, SpatialPadd, EnsureTyped, Invertd, Lambdad
)
from monai.utils import set_determinism

from data_loader import get_train_loader, get_validation_loader
from transforms.util_transforms import DeleteDeepSupervision


def get_network(labels):
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
    return network


def get_pre_transforms_train(labels):
    t = [
        LoadImaged(keys=("image", "label")),
        EnsureChannelFirstd(keys=("image", "label")),
        NormalizeLabelsInDatasetd(keys="label", label_names=labels),
        Orientationd(keys=["image", "label"], axcodes="LPS"),
        Resized(keys=("image", "label"), spatial_size=(96,96,96)),
        NormalizeIntensityd(keys="image", subtrahend=22.6701, divisor=37.27),
        RandAffined(keys=("image", "label"), prob=0.6, rotate_range=((-0.5, 0.5), (0, 0), (0, 0)), shear_range=(0.2, 0.2), translate_range=((-0.2, 0.2), (-0.1, 0.1), (-0.3, 0.3)), mode=('bilinear', 'nearest')),
        RandGaussianNoised(keys=["image"], std=0.001, prob=0.6),
        RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.3),
        RandGaussianSmoothd(keys="image", prob=0.2, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)),
        RandFlipd(["image", "label"], spatial_axis=(0,), prob=0.3),
        RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.7, max_zoom=1.3),
        SpatialPadd(keys=["image", "label",], spatial_size=(96,96,96)),
        # Transforms for click simulation
        FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),
        AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids"),
        AddGuidanceSignalDeepEditd(keys="image", guidance="guidance"),
        #
        ToTensord(keys=("image", "label")),
    ]

    return Compose(t)

def get_pre_transforms_validation(labels):
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
        DeleteDeepSupervision(keys="pred"),
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys=["pred", "label"], threshold=0.5, to_onehot=[None, 2])
    ]
    return Compose(t)


def create_trainer(args):

    set_determinism(seed=args.seed)

    multi_gpu = False # args.multi_gpu
    local_rank = args.local_rank
    if multi_gpu:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda:{}".format(local_rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda:0" if args.use_gpu else "cpu")

    pre_transforms_train = get_pre_transforms_train(args.labels)
    pre_transforms_val = get_pre_transforms_validation(args.labels)
    click_transforms = get_click_transforms()
    post_transforms = get_post_transforms()

    train_loader = get_train_loader(args.input, train_transform=pre_transforms_train)
    val_loader = get_validation_loader(args.input, eval_transform=pre_transforms_val)

    # define training components
    network = get_network(args.labels).to(device)
    if multi_gpu:
        network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[local_rank], output_device=local_rank)

    if args.resume:
        logging.info("{}:: Loading Network...".format(local_rank))
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        network.load_state_dict(torch.load(args.model_filepath, map_location=map_location))

    # define event-handlers for engine
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=args.output, output_transform=lambda x: None),
        CheckpointSaver(
            save_dir=args.output,
            save_dict={"net": network},
            save_key_metric=True,
            save_final=True,
            save_interval=args.save_interval,
            final_filename="final_deepedit.pt",
        ),
    ]
    val_handlers = val_handlers if local_rank == 0 else None

    all_val_metrics = dict()
    all_val_metrics["val_mean_dice"] = MeanDice(
        output_transform=from_engine(["pred", "label"]), include_background=False
    )

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=network,
        iteration_update=Interaction(
            deepgrow_probability=args.deepgrow_probability_val,
            transforms=click_transforms,
            click_probability_key="probability",
            train=False,
            label_names=args.labels,
        ),
        inferer=SimpleInferer(),
        postprocessing=post_transforms,
        key_val_metric=all_val_metrics,
        val_handlers=val_handlers,
    )

    dice_loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    def loss_function(outputs, targets):
        loss = sum(0.5**i * dice_loss_function(output_ir, targets) for i, output_ir in enumerate(torch.unbind(outputs, dim=1))).mean()
        return loss

    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.99, weight_decay=3e-5, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / (args.epochs + 1))**0.9)

    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(validator=evaluator, interval=args.val_freq, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        TensorBoardStatsHandler(
            log_dir=args.output,
            tag_name="train_loss",
            output_transform=from_engine(["loss"], first=True),
        ),
        CheckpointSaver(
            save_dir=args.output,
            save_dict={"net": network, "opt": optimizer, "lr": lr_scheduler},
            save_interval=args.save_interval * 2,
            save_final=True,
            final_filename="checkpoint.pt",
        )
    ]
    train_handlers = train_handlers if local_rank == 0 else train_handlers[:2]

    all_train_metrics = dict()
    all_train_metrics["train_dice"] = MeanDice(
        output_transform=from_engine(["pred", "label"]), include_background=False
    )

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        network=network,
        iteration_update=Interaction(
            deepgrow_probability=args.deepgrow_probability_train,
            transforms=click_transforms,
            click_probability_key="probability",
            train=True,
            label_names=args.labels,
            max_interactions=args.deepgrow_max_interactions,
        ),
        optimizer=optimizer,
        loss_function=loss_function,
        inferer=SimpleInferer(),
        postprocessing=post_transforms,
        amp=args.amp,
        key_train_metric=all_train_metrics,
        train_handlers=train_handlers,
    )
    return trainer


def run(args):
    if args.local_rank == 0:
        for arg in vars(args):
            logging.info("USING:: {} = {}".format(arg, getattr(args, arg)))
        print("")

    if not os.path.exists(args.output):
        logging.info("output path [{}] does not exist. creating it now.".format(args.output))
        os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, 'args.txt'), 'w+') as f:
        json.dump(args.__dict__, f, indent=2)

    trainer = create_trainer(args)

    start_time = time.time()
    trainer.run()
    end_time = time.time()

    logging.info("Total Training Time {}".format(end_time - start_time))
    if args.local_rank == 0:
        logging.info("{}:: Saving Final PT Model".format(args.local_rank))
        torch.save(
            trainer.network.state_dict(), os.path.join(args.output, "pretrained_deepedit_final.pt")
        )

    # if not args.multi_gpu:
    #     logging.info("{}:: Saving TorchScript Model".format(args.local_rank))
    #     model_ts = torch.jit.script(trainer.network)
    #     torch.jit.save(model_ts, os.path.join(args.output, "pretrained_deepedit_final.ts"))

    if args.multi_gpu:
        dist.destroy_process_group()


def strtobool(val):
    return bool(distutils.util.strtobool(val))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--seed", type=int, default=0)

    parser.add_argument("-i", "--input", default="data/EAS/",)
    parser.add_argument("-o", "--output", default=f"deepedit_train_output/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

    parser.add_argument("-g", "--use_gpu", type=strtobool, default="true")
    parser.add_argument("-a", "--amp", type=strtobool, default="false")

    parser.add_argument("-e", "--epochs", type=int, default=200)
    parser.add_argument("-t", "--limit", type=int, default=0)
    parser.add_argument("-r", "--resume", type=strtobool, default="false")

    parser.add_argument("-f", "--val_freq", type=int, default=5)

    parser.add_argument("-dpt", "--deepgrow_probability_train", type=float, default=0.9)
    parser.add_argument("-dpv", "--deepgrow_probability_val", type=float, default=1.0)
    parser.add_argument("-dmi", "--deepgrow_max_interactions", type=int, default=1)

    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--multi_gpu", type=strtobool, default="false")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    args.labels = {"anal_sphincter": 1, "background": 0}
    run(args)


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
