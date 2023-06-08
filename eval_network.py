import json
import logging
import os
from datetime import datetime
from typing import List

import numpy as np
import toml
import torch
from monai.apps import get_logger
from monai.engines import SupervisedEvaluator, IterationEvents
from monai.handlers import MeanDice, from_engine, HausdorffDistance
from monai.networks.nets import DynUNet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, Resized, \
    NormalizeIntensityd, SpatialPadd, CastToTyped, EnsureTyped, \
    Activationsd, AsDiscreted, ToDeviced
from monai.utils import CommonKeys, MetricReduction

from data_loader import get_test_loader
from interaction import BoundaryClickInteraction
from transforms.guidance_transforms import AddRandomBoundaryClickSignald, EncodeBoundaryClickSignald, SaveGuidanced, \
    LoadGuidanced, ToImplicitGuidanced
from transforms.util_transforms import SplitLabelsd, CombineLabelsd, UndoLabelAndGuidanceTupled
from util.mean_boundary_dice import MeanBoundaryDice
from util.mean_guidance_distance_metric import MeanGuidanceDistance


def get_loader(args, guidance_output_dir):
    pta = args["pre-transforms"]
    pre_transforms = [
        LoadImaged(keys=["image", "label"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
        Orientationd(keys=["image", "label"], axcodes=pta["orientationd_axcodes"], allow_missing_keys=True),
        Spacingd(keys=["image", "label"], pixdim=pta["spacingd_pixdim"], mode=pta["spacingd_mode"], allow_missing_keys=True),
        Resized(["image", "label"], spatial_size=pta["resized_spatial_size"], allow_missing_keys=True),
        NormalizeIntensityd(keys="image", subtrahend=pta["normalizeintensityd_subtrahend"], divisor=pta["normalizeintensityd_divisor"]),
        SpatialPadd(keys=["image", "label"], spatial_size=pta["spatialpadd_spatial_size"], allow_missing_keys=True),
        CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8), allow_missing_keys=True),
        EnsureTyped(keys=["image", "label"], allow_missing_keys=True)
    ]

    dta = args["data"]
    test_loader = get_test_loader(dta["root"], dta["image_folder"][2], dta["label_folder"][2], batch_size=dta["batch_size"], guidance_folder=guidance_output_dir, eval_transform=Compose(pre_transforms))

    return test_loader

def get_click_transforms(args, guidance_output_dir, num_points):
    cta = args["click-transforms"]

    label_key_names = [f"label_{label_name}" for label_name in args["data"]["label_names"] if label_name != "background"]
    guidance_key_names = [f"guidance_{label_name}" for label_name in args["data"]["label_names"] if label_name != "background"]

    click_transforms = [
        Activationsd(keys="zero_pred", softmax=True),
        AsDiscreted(keys="zero_pred", threshold=0.5),
        SplitLabelsd(keys="label", label_names=args["data"]["label_names"], key_names=label_key_names),
        LoadGuidanced("guidance"),
        AddRandomBoundaryClickSignald(keys=label_key_names, pred_key="zero_pred", guidance_key=guidance_key_names, num_points=num_points, prob=1.0, sigma=cta["user_variability"], error_influence=cta["error_influence"]),
        EncodeBoundaryClickSignald(keys=guidance_key_names, image_key="image", pred_key="zero_pred", encoding=cta["encoding"], differ_inward_points=cta["differ_inward_points"], mix_image_features=cta["mix_image_features"], tau=cta["tau"], sigma=cta["sigma"], scaling=cta["scaling"], allow_missing_keys=True),
        SaveGuidanced(keys=guidance_key_names, image_meta_dict_key="image_meta_dict", output_folder=guidance_output_dir),
        CombineLabelsd(keys=label_key_names),
        ToImplicitGuidanced(keys=guidance_key_names, shape=args["pre-transforms"]["spatialpadd_spatial_size"]),
        ToDeviced(keys="guidance", device="cuda:0")
    ]

    return Compose(click_transforms).set_random_state(seed=args["training"]["seed"]+num_points) # Update seed otherwise every guidance point is the same

def get_post_transforms():
    post_transforms = [
        # UndoLabelAndGuidanceTupled(keys="label"),
        EnsureTyped(keys=["pred", "prev_pred"]),
        Activationsd(keys=["pred", "prev_pred"], softmax=True),
        AsDiscreted(keys=["pred", "prev_pred"], threshold=0.5)
    ]

    return Compose(post_transforms)

def get_network(args, model_path):
    nwa = args["network"]
    network = DynUNet(spatial_dims=3, in_channels=nwa["input_channels"], out_channels=2, kernel_size=nwa["kernel_sizes"],
                      strides=nwa["strides"], upsample_kernel_size=nwa["strides"][1:], deep_supervision=True,
                      deep_supr_num=2, norm_name="instance")
    network.load_state_dict(torch.load(model_path))
    return network

PREV_PRED : List[torch.Tensor] = []
BATCH_I = 0

def run_evaluation(args, model_path, guidance_output_dir):
    ta = args["training"]
    device = torch.device(ta["device"])

    # Load network and data loaders
    network = get_network(args, model_path).to(device)
    test_loader = get_loader(args, guidance_output_dir)

    test_metric_dict = {
        "test_dice": MeanDice(include_background=False, output_transform=from_engine(["pred", "label"])),
        "test_hd": HausdorffDistance(output_transform=from_engine(["label", "pred"]), percentile=95.0, directed=True),
        "test_boundary_dice": MeanBoundaryDice(output_transform=from_engine(["pred", "label", "guidance"])),
        "test_gd1": MeanGuidanceDistance(output_transform=from_engine(["zero_pred", "guidance"])),
        "test_gd2": MeanGuidanceDistance(output_transform=from_engine(["pred", "guidance"]))
    }

    global PREV_PRED
    PREV_PRED = [None] * len(test_loader) * args["data"]["batch_size"]
    post_transforms = get_post_transforms()

    def prepare_batch(batch, device, *_):
        if PREV_PRED[BATCH_I] is None:
            return batch["image"].to(device), (batch["label"].to(device), batch["guidance"], batch["zero_pred"].to(device))
        else:
            return batch["image"].to(device), (batch["label"].to(device), batch["guidance"], batch["zero_pred"].to(device), PREV_PRED[BATCH_I].to(device))



    for num_points in range(6):
        print(f"NUM POINTS: {num_points} ----------------------------------------------------------------------------------")
        click_transforms = get_click_transforms(args, guidance_output_dir, num_points)
        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=test_loader,
            network=network,
            inferer=None,
            iteration_update=BoundaryClickInteraction(click_transforms, differ_inward_points=args["click-transforms"]["differ_inward_points"], pred_key="zero_pred"),
            prepare_batch=prepare_batch,
            postprocessing=post_transforms,
            key_val_metric=test_metric_dict,
        )

        @evaluator.on(IterationEvents.FORWARD_COMPLETED)
        def add_guidance_to_output(engine):
            global PREV_PRED, BATCH_I
            engine.state.output["guidance"] = engine.state.output[CommonKeys.LABEL][1]
            engine.state.output["zero_pred"] = engine.state.output[CommonKeys.LABEL][2]
            if len(engine.state.output[CommonKeys.LABEL]) == 4:
                engine.state.output["prev_pred"] = engine.state.output[CommonKeys.LABEL][3]
            else:
                engine.state.output["prev_pred"] = engine.state.output["zero_pred"].clone()
            PREV_PRED[BATCH_I] = engine.state.output["pred"].clone()
            engine.state.output[CommonKeys.LABEL] = engine.state.output[CommonKeys.LABEL][0]
            BATCH_I = (BATCH_I + 1) % len(PREV_PRED)


        evaluator.run()
        yield [evaluator.state.metrics[key] for key in evaluator.state.metrics]

def main(toml_file, model_file):
    # Parse the TOML file
    if toml_file is not None:
        new_args = toml.load(toml_file)
    else:
        new_args = {}

    # Read the defaults and update them with the given parameters
    args = toml.load("default.toml")
    args["training"].update(new_args.get("training", {}))
    args["data"].update(new_args.get("data", {}))
    args["network"].update(new_args.get("network", {}))
    args["pre-transforms"].update(new_args.get("pre-transforms", {}))
    args["train-transforms"].update(new_args.get("train-transforms", {}))
    args["click-transforms"].update(new_args.get("click-transforms", {}))

    # Create the output folder
    os.makedirs("test_output", exist_ok=True)
    output_dir = f"test_output/{args['network']['model_name']}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}/"
    os.makedirs(output_dir)

    guidance_output_dir = os.path.join(output_dir, "guidance")
    os.makedirs(guidance_output_dir, exist_ok=True)

    # Create the logger
    fmt = "[%(levelname)-5.5s][%(asctime)s] %(message)s"
    formatter = logging.Formatter(fmt)
    file_handler = logging.FileHandler(os.path.join(output_dir, f"{args['network']['model_name']}_train_stdout.log"))
    file_handler.setFormatter(formatter)
    l = get_logger("test", fmt=fmt, logger_handler=file_handler)       # logger can be retrieved by calling logging.getLogger("train")

    metrics = list(run_evaluation(args, model_file, guidance_output_dir))
    # json_data = {
    #     "model_path": model_file,
    #     "dice" :  [[dice] if isinstance(dice, (float, int)) else dice.tolist() for dice,_,_,_,_ in metrics],
    #     "hausdorff" : [[hd] if isinstance(hd, (float, int)) else hd.tolist() for _,hd,_,_,_ in metrics],
    #     "gd1" : {
    #         i: [gd1] if isinstance(gd1, (float, int)) else gd1.tolist() for i,(_,_,gd1,_,_) in enumerate(metrics)
    #     },
    #     "gd2": {
    #         i: [gd2] if isinstance(gd2, (float, int)) else gd2.tolist() for i, (_,_,_,gd2,_) in enumerate(metrics)
    #     },
    #     "boundary_dice" : [[dice] if isinstance(dice, (float, int)) else dice.tolist() for _,_,_,_,dice in metrics]
    # }


    json_data = {
        "model_path": model_file,
        # "dice": [[t[0]] if isinstance(t[0], (float, int)) else t[0].tolist() for t in metrics],
        "dice": [t[0] for t in metrics],
        "hausdorff": [t[1] for t in metrics],
        "boundary_dice": [t[2] for t in metrics],
        "gd_impr" : ((metrics[-1][3] - metrics[-1][4]) / metrics[-1][3]).mean().item()
    }

    with open(os.path.join(output_dir, "evaluation.json"), "w+") as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        toml_file = sys.argv[1]
        model_file = sys.argv[2]
        main(toml_file, model_file)
    else:
        print("invalid arguments")
        exit(1)
