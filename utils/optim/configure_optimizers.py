import copy
from collections import defaultdict
from typing import List, Mapping, Sequence, Tuple, Union

from lightning.pytorch import LightningModule
from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.utilities.types import LRSchedulerType
from torch.optim import Optimizer


def parser_optim_config(optim_config):
    """
    Parse the optimizer config.

    Args:
        optim_config (dict): The optimizer and lr_scheduler config.
    """
    optim_config = copy.deepcopy(optim_config)
    if not isinstance(optim_config, Sequence):
        optim_config = [optim_config]

    all_required_parameters = set()
    for i, optim_cfg in enumerate(optim_config):
        # parse the optimizer config
        if "optimizer" not in optim_cfg:
            optim_config[i] = {"optimizer": optim_cfg}
            optim_cfg = optim_config[i]

        # parse the params of optimizers
        if "init_args" not in optim_cfg["optimizer"]:
            optim_cfg["optimizer"]["init_args"] = {}
        optimizer_init_args = optim_cfg["optimizer"]["init_args"]
        if "params" not in optimizer_init_args:
            optimizer_init_args["params"] = [{"params": None}]
        if not isinstance(optimizer_init_args["params"], Sequence):
            optimizer_init_args["params"] = [optimizer_init_args["params"]]

        optimizer_init_args["params"] = [
            p if isinstance(p, Mapping) else {"params": p}
            for p in optimizer_init_args["params"]
        ]

        assert all(
            [
                isinstance(p["params"], str) or p["params"] is None
                for p in optimizer_init_args["params"]
            ]
        ), "params must be None or str"

        # get all required parameters
        all_required_parameters.update(
            [p["params"] for p in optimizer_init_args["params"]]
        )

        # parse the lr_scheduler config
        if "lr_scheduler" in optim_cfg:
            if "scheduler" not in optim_cfg["lr_scheduler"]:
                optim_cfg["lr_scheduler"] = {"scheduler": optim_cfg["lr_scheduler"]}
    return optim_config, all_required_parameters


def get_parameters(model, all_required_parameters):
    """
    Get all optimizer parameters.

    Args:
        model: a LightningModule.
        all_required_parameters: a set of required parameter names.
    """
    parameters = defaultdict(list)
    set_rest = None in all_required_parameters
    all_required_parameters.discard(None)
    all_required_parameters = sorted(
        sorted(all_required_parameters), key=len, reverse=True
    )
    for name, p in model.named_parameters():
        for required_parameter in all_required_parameters:
            if required_parameter in name:
                parameters[required_parameter].append(p)
                break
        else:
            if set_rest:
                parameters[None].append(p)
            else:
                raise ValueError(f"parameter {name} is not in required_parameters")
    return parameters


def construct_optimizer(model, optimizer, set_lr=False):
    """
    Constructs the optimizer.

    Args:
        model: a LightningModule.
        optimizer: dictionary containing optimizer configuration.
        set_lr: whether to set the learning rate by the model.lr
    """
    # set lr for lr finder
    if model.lr is not None and set_lr:
        if "init_args" not in optimizer:
            optimizer["init_args"] = {}
        optimizer["init_args"]["lr"] = model.lr

    # construct optimizer
    optimizer = instantiate_class(tuple(), optimizer)
    if set_lr and model.lr is None:
        model.lr = optimizer.defaults["lr"]
    return optimizer


def construct_lr_scheduler(lr_scheduler, optimizer):
    """
    Constructs the lr_scheduler.

    Args:
        lr_scheduler: dictionary containing lr_scheduler configuration.
        optimizer: optimizer used to construct lr_scheduler.
    """

    # construct lr_scheduler
    lr_scheduler["scheduler"] = instantiate_class(optimizer, lr_scheduler["scheduler"])

    # construct warmup_lr_scheduler
    manual_lr_scheduler = None
    if "warmup_config" in lr_scheduler:
        manual_lr_scheduler = lr_scheduler.pop("warmup_config")
        if "scheduler" not in manual_lr_scheduler:
            manual_lr_scheduler = {
                "scheduler": {
                    "class_path": "utils.optim.WarmupScheduler",
                    "init_args": manual_lr_scheduler,
                }
            }
        manual_lr_scheduler.setdefault("frequency", 1)
        manual_lr_scheduler["scheduler"] = instantiate_class(
            optimizer, manual_lr_scheduler["scheduler"]
        )
    return lr_scheduler, manual_lr_scheduler


def get_configure_optimizers_method(optim_config):
    def configure_optimizers(
        self: LightningModule,
    ) -> Union[Optimizer, Tuple[List[Optimizer], List[LRSchedulerType]]]:
        optim_cfg, all_required_parameters = parser_optim_config(optim_config)
        parameters = get_parameters(self, all_required_parameters)
        manual_step_scedulers = []

        for i, cfg in enumerate(optim_cfg):
            # set parameters
            for p in cfg["optimizer"]["init_args"]["params"]:
                p["params"] = parameters[p["params"]]
            # construct optimizer
            cfg["optimizer"] = construct_optimizer(
                self, cfg["optimizer"], set_lr=i == 0
            )
            # construct lr_scheduler
            if "lr_scheduler" in cfg:
                cfg["lr_scheduler"], manual_lr_scheduler = construct_lr_scheduler(
                    cfg["lr_scheduler"], cfg["optimizer"]
                )
                if manual_lr_scheduler is not None:
                    manual_step_scedulers.append(manual_lr_scheduler)
        self.manual_step_scedulers = manual_step_scedulers
        return optim_cfg

    return configure_optimizers
