from typing import Union, Dict, Any, Callable, Mapping

import torch
from monai.data import decollate_batch, list_data_collate
from monai.data.utils import collate_meta_tensor
from monai.engines import SupervisedTrainer, SupervisedEvaluator, IterationEvents
from monai.metrics import compute_dice
from monai.transforms import Transform
from monai.utils import CommonKeys


class BoundaryClickInteraction:
    """
    Ignite process_function used to introduce boundary click interactions for training/evaluation. This function will first
    pass the sample without any guidance to the network to obtain a 'clean' prediction. Based on this prediction the boundary
    clicks are generated and added to the input. The updated input is than passed again to the network for training/evaluation.
    """

    def __init__(self, click_transforms: Transform, num_classes : int = 2, differ_inward_points : bool = False, pred_key : str = "pred"):
        """
        Initializer for ``BoundaryClickInteraction``.
        :param click_transforms: The transforms to apply to generate the boundary clicks. These transforms can assume that
                                 the prediction of them model is stored under the predicate key as specified by ``CommonKeys.PRED``.
        :param differ_inward_points: Whether to use an additional channel for inward points. The additional channel will use the same
                                     encoding.
        :param num_classes: The number of classes (including the background). Defaults to 2.
        """
        self.transforms = click_transforms
        self.differ_inward_points = differ_inward_points
        self.num_classes = num_classes
        self.pred_key = pred_key

    def __call__(self, engine: Union[SupervisedTrainer, SupervisedEvaluator], batchdata: Dict[str, Any]):
        # Extract the input volume
        inputs = batchdata["image"]
        inputs = inputs.to(engine.state.device)

        engine.fire_event(IterationEvents.INNER_ITERATION_STARTED)
        engine.network.eval()       # put the network in eval mode since we just want the prediction for boundary click estimation

        # Calculate the prediction
        with torch.no_grad():
            # Add in an empty guidance channel (inputs.shape[0] = batch size, inputs.shape[-3:] = image volume size)
            guidance = torch.zeros(inputs.shape[0], (self.num_classes - 1) * (2 if self.differ_inward_points else 1), *inputs.shape[-3:], dtype=torch.float).to(inputs.device)      # Create guidance channel for every class except background
            inputs = torch.cat([inputs, guidance], dim=1)       # concatenate them in the channel dimension
            predictions = engine.inferer(inputs, engine.network)
        batchdata.update({self.pred_key: predictions})

        # Apply the click transforms to the batch data before passing it back to the engine
        batchdata_list = decollate_batch(batchdata, detach=True)
        for i in range(len(batchdata_list)):
            batchdata_list[i] = self.transforms(batchdata_list[i])
        batchdata = list_data_collate(batchdata_list)

        engine.fire_event(IterationEvents.INNER_ITERATION_COMPLETED)

        # Run the default evaluation/training loop with the updated batch data
        engine.state.batch = batchdata
        return engine._iteration(engine, batchdata)