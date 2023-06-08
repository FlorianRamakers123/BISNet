import logging
import os
from datetime import datetime
import numpy as np
import toml
import torch
from monai.apps import get_logger
from monai.engines import SupervisedEvaluator, SupervisedTrainer, IterationEvents
from monai.handlers import StatsHandler, TensorBoardStatsHandler, CheckpointSaver, MeanDice, from_engine, \
    LrScheduleHandler, ValidationHandler
from monai.losses import DiceCELoss
from monai.networks.nets import DynUNet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, Resized, \
    NormalizeIntensityd, SpatialPadd, CastToTyped, EnsureTyped, RandAffined, RandGaussianNoised, RandScaleIntensityd, \
    RandGaussianSmoothd, RandFlipd, RandZoomd, Activationsd, AsDiscreted, ToDeviced, Identityd
from monai.utils import CommonKeys, set_determinism
from tensorboardX import SummaryWriter

from data_loader import get_train_loader, get_validation_loader
from interaction import BoundaryClickInteraction
from loss import WeightedDiceCELoss, GuidanceDistanceLoss
from transforms.guidance_transforms import AddRandomBoundaryClickSignald, EncodeBoundaryClickSignald, \
    ToImplicitGuidanced, SaveGuidanced, DropoutGuidanced, DiceBasedDropoutGuidanced, DecayingDropoutGuidanced, \
    CosineAnnealingRestartDropoutGuidanced
from transforms.util_transforms import SplitLabelsd, CombineLabelsd, UndoLabelAndGuidanceTupled, \
    DeleteDeepSupervision, CalculateDiceScored
from util.mean_guidance_dice_metric import MeanGuidanceDice
from util.mean_interaction_metric import MeanInteractionMetric
from util.tensorboard_util import SegmentationVisualizer

def get_loaders(args):
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

    tta = args["train-transforms"]
    train_transforms = [
        RandAffined(keys=["image", "label"], prob=tta["randaffined_prob"], rotate_range=tta["randaffined_rotate_range"], shear_range=
        tta["randaffined_shear_range"], translate_range=tta["randaffined_translate_range"], mode=tta["randaffined_mode"], allow_missing_keys=True),
        RandGaussianNoised(keys=["image"], std=tta["randgausssiannoised_std"], prob=tta["randgaussiannoised_prob"]),
        RandScaleIntensityd(keys=["image"], factors=tta["randscaleintensityd_factors"], prob=tta["randscaleintensityd_prob"]),
        RandGaussianSmoothd(keys="image", prob=tta["randgaussiansmooth_prob"], sigma_x=tta["randgaussiansmoothd_sigma_x"], sigma_y=tta["randgaussiansmoothd_sigma_y"], sigma_z=tta["randgaussiansmoothd_sigma_z"]),
        RandFlipd(["image", "label"], spatial_axis=tta["randflipd_spatial_axis"], prob=tta["randflipd_prob"], allow_missing_keys=True),
        RandZoomd(keys=["image", "label"], prob=tta["randzoomd_prob"], min_zoom=tta["randzoomd_min_zoom"], max_zoom=tta["randzoomd_max_zoom"], allow_missing_keys=True),
        SpatialPadd(keys=["image", "label"], spatial_size=tta["spatialpadd_spatial_size"], allow_missing_keys=True),
        CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8), allow_missing_keys=True),
        EnsureTyped(keys=["image", "label"], allow_missing_keys=True),
    ]

    dta = args["data"]
    train_loader = get_train_loader(dta["root"], dta["image_folder"][0], dta["label_folder"][0], dta["batch_size"], train_transform=Compose(pre_transforms + train_transforms).set_random_state(seed=args["training"]["seed"]))
    val_loader = get_validation_loader(dta["root"], dta["image_folder"][1], dta["label_folder"][1], dta["batch_size"], eval_transform=Compose(pre_transforms))

    return train_loader, val_loader

def get_click_transforms(args, output_dir, train):
    cta = args["click-transforms"]

    label_key_names = [f"label_{label_name}" for label_name in args["data"]["label_names"] if label_name != "background"]
    guidance_key_names = [f"guidance_{label_name}" for label_name in args["data"]["label_names"] if label_name != "background"]

    guidance_output_dir = os.path.join(output_dir, "guidance")
    os.makedirs(guidance_output_dir, exist_ok=True)

    dropout_transform = None
    if cta["dropout_type"] == "none":
        dropout_transform = Identityd(keys=guidance_key_names)
    if cta["dropout_type"] == "dice":
        dropout_transform = DiceBasedDropoutGuidanced(keys=guidance_key_names, dice_threshold=cta["dice_threshold"]).set_random_state(seed=args["training"]["seed"])
    if cta["dropout_type"] == "decaying":
        dropout_transform = DecayingDropoutGuidanced(keys=guidance_key_names, b=cta["dropout_b"], alpha=cta["dropout_alpha"], cold_start=cta["dropout_cold_start"]).set_random_state(seed=args["training"]["seed"])
    if cta["dropout_type"] == "cosine annealing":
        dropout_transform = CosineAnnealingRestartDropoutGuidanced(keys=guidance_key_names, max_prob=cta["dropout_max_prob"], min_prob=cta["dropout_min_prob"], restart=cta["dropout_restart"], max_decay=cta["dropout_max_decay"], min_decay=cta["dropout_min_decay"])
    if dropout_transform is None:
        raise RuntimeError(f"Unsupported value '{cta['dropout_type']}' for dropout_type")

    click_transforms = [
        Activationsd(keys="pred" if train else "zero_pred", softmax=True),
        AsDiscreted(keys="pred" if train else "zero_pred", threshold=0.5),
        CalculateDiceScored(keys=("pred" if train else "zero_pred", "label"), num_classes=len(args["data"]["label_names"]) + 1),
        SplitLabelsd(keys="label", label_names=args["data"]["label_names"], key_names=label_key_names),
        AddRandomBoundaryClickSignald(keys=label_key_names, pred_key="pred" if train else "zero_pred", guidance_key=guidance_key_names, num_points=cta["num_points"] if train else cta["val_num_points"], prob=cta["probability"] if train else 1.0, sigma=cta["user_variability"], error_influence=cta["error_influence"]),
        dropout_transform,
        EncodeBoundaryClickSignald(keys=guidance_key_names, image_key="image", pred_key="pred" if train else "zero_pred", encoding=cta["encoding"], differ_inward_points=cta["differ_inward_points"], mix_image_features=cta["mix_image_features"], tau=cta["tau"], sigma=cta["sigma"], scaling=cta["scaling"], allow_missing_keys=True),
        SaveGuidanced(keys=guidance_key_names, image_meta_dict_key="image_meta_dict", output_folder=guidance_output_dir),
        CombineLabelsd(keys=label_key_names),
        ToImplicitGuidanced(keys=guidance_key_names, shape=args["pre-transforms"]["spatialpadd_spatial_size"]),
        ToDeviced(keys="guidance", device="cuda:0")
    ]

    if not train:
        click_transforms.remove(dropout_transform)

    return Compose(click_transforms).set_random_state(seed=args["training"]["seed"])

def get_post_transforms():
    post_transforms = [
        DeleteDeepSupervision(keys="pred"),
        # UndoLabelAndGuidanceTupled(keys="label"),
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys=["pred"], threshold=0.5)
    ]

    return Compose(post_transforms)

def get_network(args):
    nwa = args["network"]
    network = DynUNet(spatial_dims=3, in_channels=nwa["input_channels"], out_channels=2, kernel_size=nwa["kernel_sizes"],
                      strides=nwa["strides"], upsample_kernel_size=nwa["strides"][1:], deep_supervision=True,
                      deep_supr_num=2, norm_name="instance")

    return network

def create_trainer(args, output_dir):

    ta = args["training"]
    nwa = args["network"]
    device = torch.device(ta["device"])
    summary_writer = SummaryWriter(output_dir)
    set_determinism(ta["seed"])

    # Load network and data loaders
    network = get_network(args).to(device)
    train_loader, validation_loader = get_loaders(args)

    # Setup an evaluator for validation
    val_handlers = [
        StatsHandler(name="train", output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=output_dir, output_transform=lambda x: None, summary_writer=summary_writer),
        SegmentationVisualizer(summary_writer, frame_dim=-1),
        CheckpointSaver(name="train", save_dir=output_dir, key_metric_negative_sign=ta["smallest_key_metric"],
                        save_dict={"net": network}, key_metric_name=f"{ta['key_metric']}", save_key_metric=True, save_final=True,
                        save_interval=ta["model_save_interval"], file_prefix=nwa["model_name"], final_filename=f"{nwa['model_name']}_final_val.pt")
    ]

    val_metric_dict = {
        "val_dice": MeanDice(include_background=False, output_transform=from_engine(["pred", "label"])),
        "val_guidance_dice": MeanGuidanceDice(include_background=False, output_transform=from_engine(["pred", "label", "guidance"])),
        "val_interaction": MeanInteractionMetric(sigma=ta["interaction_sigma"], output_transform=from_engine(["zero_pred", "pred", "label", "guidance"]))
    }

    post_transforms = get_post_transforms()
    click_transforms_val = get_click_transforms(args, output_dir, train=False)
    click_transforms_train = get_click_transforms(args, output_dir, train=True)

    def prepare_batch(batch, device, *_):
        # In order to use the guidance points in a loss function we have to pair it with the label
        # The add_guidance_to_output undoes this
        if "zero_pred" in batch:
            return batch["image"].to(device), (batch["label"].to(device), batch["guidance"], batch["zero_pred"])
        else:
            return batch["image"].to(device), (batch["label"].to(device), batch["guidance"]),

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=validation_loader,
        network=network,
        inferer=None,
        iteration_update=BoundaryClickInteraction(click_transforms_val, differ_inward_points=args["click-transforms"]["differ_inward_points"], pred_key="zero_pred"),
        prepare_batch=prepare_batch,
        postprocessing=post_transforms,
        key_val_metric=val_metric_dict,
        val_handlers=val_handlers,
    )

    @evaluator.on(IterationEvents.FORWARD_COMPLETED)
    def add_guidance_to_output(engine):
        engine.state.output["guidance"] = engine.state.output[CommonKeys.LABEL][1]
        engine.state.output["zero_pred"] = engine.state.output[CommonKeys.LABEL][2]
        engine.state.output[CommonKeys.LABEL] = engine.state.output[CommonKeys.LABEL][0]

    optimizer = torch.optim.SGD(network.parameters(), lr=ta["init_learning_rate"], momentum=ta["momentum"], weight_decay=ta["weight_decay"], nesterov=ta["nesterov"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / (ta["max_epochs"] + 1))**ta["lr_decay"])

    train_handlers = [
        LrScheduleHandler(lr_scheduler=scheduler, print_lr=True),
        ValidationHandler(validator=evaluator, interval=ta["validation_interval"], epoch_level=True),
        StatsHandler(name="train", tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        TensorBoardStatsHandler(log_dir=output_dir, tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        SegmentationVisualizer(summary_writer, frame_dim=-1, interval=ta["validation_interval"], prefix="train_"),
        CheckpointSaver(save_dir=output_dir, save_dict={"model": network, "opt": optimizer, "lr": scheduler},
                        save_interval=ta["checkpoint_save_interval"], save_final=True, file_prefix="train",
                        final_filename="train_final_checkpoint.pt"),
    ]

    train_metric_dict = {
        "train_dice": MeanDice(include_background=False, output_transform=from_engine(["pred", "label"])),
        "train_guidance_dice": MeanGuidanceDice(include_background=False, output_transform=from_engine(["pred", "label", "guidance"]))
    }

    losses = {}
    for loss_name in ta["loss_funcs"]:
        if loss_name == "DiceCELoss":
            dice_ce_loss = DiceCELoss(softmax=True)
            losses["DiceCELoss"] = lambda i, l, g: dice_ce_loss(i, l)
        if loss_name == "WeightedDiceCELoss":
            wdce_loss = WeightedDiceCELoss(sigma=ta["WDCE_sigma"])
            losses["WeightedDiceCELoss"] = lambda i, l, g: wdce_loss(i, l, g)
        if loss_name == "GuidanceDistanceLoss":
            gd_loss = GuidanceDistanceLoss(sigma=ta["GDL_sigma"], sigma_decay=ta["GDL_sigma_decay"])
            losses["GuidanceDistanceLoss"] = lambda i, _, g: gd_loss(i, g)

    class LossResult:
        def __init__(self, d):
            self.d = d
        def mean(self):
            return self

    def loss_function(outputs, targets_and_guidance):
        l = {}
        total = 0
        for i, loss_name in enumerate(losses):
            loss = sum(0.5**i * losses[loss_name](output_ir, *targets_and_guidance) for i, output_ir in enumerate(torch.unbind(outputs, dim=1))).mean()
            total += ta["loss_weights"][i] * loss
            l[loss_name] = loss
        l["total_loss"] = total
        return LossResult(l)


    trainer = SupervisedTrainer(
        device=device,
        max_epochs=ta["max_epochs"],
        prepare_batch=prepare_batch,
        train_data_loader=train_loader,
        network=network,
        optimizer=optimizer,
        iteration_update=BoundaryClickInteraction(click_transforms_train, differ_inward_points=args["click-transforms"]["differ_inward_points"]),
        loss_function=loss_function,
        train_handlers=train_handlers,
        postprocessing=post_transforms,
        key_train_metric=train_metric_dict
    )

    @trainer.on(IterationEvents.LOSS_COMPLETED)
    def log_loss(engine):
        d = engine.state.output[CommonKeys.LOSS].d
        engine.state.output[CommonKeys.LOSS] = d["total_loss"]
        d.pop("total_loss")
        for key in d:
            summary_writer.add_scalar(tag=key, scalar_value=d[key].mean().item(), global_step=engine.state.iteration)

    @trainer.on(IterationEvents.LOSS_COMPLETED)
    def add_guidance_to_output(engine):
        engine.state.output["guidance"] = engine.state.output[CommonKeys.LABEL][1]
        engine.state.output[CommonKeys.LABEL] = engine.state.output[CommonKeys.LABEL][0]

    return trainer

def main():
    # Parse the TOML file
    import sys
    if len(sys.argv) > 1:
        toml_file = sys.argv[1]
        new_args = toml.load(toml_file)
    else:
        toml_file = None
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
    os.makedirs("train_output", exist_ok=True)
    output_dir = f"train_output/{args['network']['model_name']}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}/"
    os.makedirs(output_dir)

    # Write the resulting TOML file to the output directory
    output_toml_file = output_dir + "config.toml"
    with open(output_toml_file, "w+") as f:
        toml.dump(args, f)

    # Create the logger
    fmt = "[%(levelname)-5.5s][%(asctime)s] %(message)s"
    formatter = logging.Formatter(fmt)
    file_handler = logging.FileHandler(os.path.join(output_dir, f"{args['network']['model_name']}_train_stdout.log"))
    file_handler.setFormatter(formatter)
    l = get_logger("train", fmt=fmt, logger_handler=file_handler)       # logger can be retrieved by calling logging.getLogger("train")
    if toml_file is not None:
        l.info(f"Using TOML file: '{toml_file}'")
    else:
        l.info("No TOML file specified. Using Defaults.")

    trainer = create_trainer(args, output_dir)
    trainer.run()


if __name__ == "__main__":
    main()