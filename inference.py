import json
import os
from datetime import datetime

import numpy as np
import toml
import torch
from monai.engines import SupervisedEvaluator, IterationEvents
from monai.handlers import MeanDice, from_engine
from monai.networks.nets import DynUNet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, Resized, \
    NormalizeIntensityd, SpatialPadd, CastToTyped, EnsureTyped, \
    Activationsd, AsDiscreted, ToDeviced, Invertd, SaveImaged
from monai.utils import CommonKeys

from data_loader import get_test_loader
from interaction import BoundaryClickInteraction
from transforms.guidance_transforms import AddRandomBoundaryClickSignald, EncodeBoundaryClickSignald, SaveGuidanced, \
    LoadGuidanced, ToImplicitGuidanced
from transforms.util_transforms import SplitLabelsd, CombineLabelsd, UndoLabelAndGuidanceTupled


def get_loader(args):
    pta = args["pre-transforms"]
    pre_transforms = Compose([
        LoadImaged(keys=["image", "label"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
        Orientationd(keys=["image", "label"], axcodes=pta["orientationd_axcodes"], allow_missing_keys=True),
        Spacingd(keys=["image", "label"], pixdim=pta["spacingd_pixdim"], mode=pta["spacingd_mode"], allow_missing_keys=True),
        Resized(["image", "label"], spatial_size=pta["resized_spatial_size"], allow_missing_keys=True),
        NormalizeIntensityd(keys="image", subtrahend=pta["normalizeintensityd_subtrahend"], divisor=pta["normalizeintensityd_divisor"]),
        SpatialPadd(keys=["image", "label"], spatial_size=pta["spatialpadd_spatial_size"], allow_missing_keys=True),
        CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8), allow_missing_keys=True),
        EnsureTyped(keys=["image", "label"], allow_missing_keys=True)
    ])

    dta = args["data"]
    test_loader = get_test_loader(dta["root"], dta["image_folder"][2], dta["label_folder"][2], batch_size=dta["batch_size"], eval_transform=pre_transforms)

    return pre_transforms, test_loader

def get_click_transforms(args):
    cta = args["click-transforms"]

    label_key_names = [f"label_{label_name}" for label_name in args["data"]["label_names"] if label_name != "background"]
    guidance_key_names = [f"guidance_{label_name}" for label_name in args["data"]["label_names"] if label_name != "background"]

    click_transforms = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", threshold=0.5),
        SplitLabelsd(keys="label", label_names=args["data"]["label_names"], key_names=label_key_names),
        AddRandomBoundaryClickSignald(keys=label_key_names, guidance_key=guidance_key_names, num_points=cta["num_points"], prob=0.0, sigma=cta["user_variability"], error_influence=cta["error_influence"]),
        EncodeBoundaryClickSignald(keys=guidance_key_names, image_key="image", pred_key="zero_pred", encoding=cta["encoding"], differ_inward_points=cta["differ_inward_points"], mix_image_features=cta["mix_image_features"], sigma=cta["sigma"], scaling=cta["scaling"], allow_missing_keys=True),
        CombineLabelsd(keys=label_key_names),
        ToImplicitGuidanced(keys=guidance_key_names, shape=args["pre-transforms"]["spatialpadd_spatial_size"]),
        ToDeviced(keys="guidance", device=args["training"]["device"])
    ]

    return Compose(click_transforms).set_random_state(seed=args["training"]["seed"])

def get_post_transforms(pretransforms, output_dir):
    post_transforms = [
        # UndoLabelAndGuidanceTupled(keys="label"),
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        Invertd(
            keys=["pred"],
            transform=pretransforms,
            orig_keys=["label"],
            nearest_interp=False,
            allow_missing_keys=True
        ),
        AsDiscreted(keys=["pred"], threshold=0.5, argmax=True),
        SaveImaged(keys=["pred"], output_dir=output_dir)
    ]

    return Compose(post_transforms)

def get_network(args, model_path):
    nwa = args["network"]
    network = DynUNet(spatial_dims=3, in_channels=nwa["input_channels"], out_channels=2, kernel_size=nwa["kernel_sizes"],
                      strides=nwa["strides"], upsample_kernel_size=nwa["strides"][1:], deep_supervision=True,
                      deep_supr_num=2, norm_name="instance")
    network.load_state_dict(torch.load(model_path))
    return network

def run_inference(args, model_path, output_dir):
    ta = args["training"]
    device = torch.device(ta["device"])

    # Load network and data loaders
    network = get_network(args, model_path).to(device)
    pretransforms, test_loader = get_loader(args)

    test_metric_dict = {
        # "test_dice": MeanDice(include_background=False, output_transform=from_engine(["pred", "label"]))
    }

    post_transforms = get_post_transforms(pretransforms, output_dir)

    def prepare_batch(batch, device, *_):
        # In order to use the guidance points in a loss function we have to pair it with the label
        # The SplitGuidanceAndLabeld in the post-transforms undoes this
        return batch["image"].to(device), (batch["label"].to(device), batch["guidance"]),

    click_transforms = get_click_transforms(args)
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=network,
        inferer=None,
        iteration_update=BoundaryClickInteraction(click_transforms),
        prepare_batch=prepare_batch,
        postprocessing=post_transforms,
        # key_val_metric=test_metric_dict,
    )

    @evaluator.on(IterationEvents.FORWARD_COMPLETED)
    def add_guidance_to_output(engine):
        engine.state.output["guidance"] = engine.state.output[CommonKeys.LABEL][1]
        engine.state.output[CommonKeys.LABEL] = engine.state.output[CommonKeys.LABEL][0]

    evaluator.run()
    return evaluator.state.metrics[f"test_dice"]

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
    os.makedirs("inference_output", exist_ok=True)
    output_dir = f"inference_output/{args['network']['model_name']}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}/"
    os.makedirs(output_dir)

    dice = run_inference(args, model_file, output_dir)
    json_data = {
        "model_path": model_file,
        "dice" : dice
    }
    with open(os.path.join(output_dir, "inference.json"), "w+") as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        toml_file = sys.argv[1]
        model_file = sys.argv[2]
    else:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        toml_file = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select TOML file", filetypes=(("TOML files","*.toml"),))
        model_file = filedialog.askopenfilename(initialdir=toml_file, title="Select model file", filetypes=(("Model files", "*.pt"),))

    main(toml_file, model_file)
