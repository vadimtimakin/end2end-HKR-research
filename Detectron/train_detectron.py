import argparse
import logging
import os

import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate, LazyConfig, LazyCall
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.data.transforms import RandomRotation
from detectron2.engine import (
    launch,
    AMPTrainer,
    SimpleTrainer,
    default_writers,
    hooks,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.model_zoo import model_zoo, get_config
from detectron2.utils import comm
from detectron2.data.transforms import Augmentation, Transform
from fvcore.common.param_scheduler import CosineParamScheduler
import albumentations as A
import torch
from detectron2.solver.build import get_default_optimizer_params
from typing import TYPE_CHECKING, Any, Callable, Optional
import math

from convnext import convnext_small, LayerNorm, convnext_tiny

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class MADGRAD(torch.optim.Optimizer):
    """
    MADGRAD_: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic
    Optimization.
    .. _MADGRAD: https://arxiv.org/abs/2101.11075
    MADGRAD is a general purpose optimizer that can be used in place of SGD or
    Adam may converge faster and generalize better. Currently GPU-only.
    Typically, the same learning rate schedule that is used for SGD or Adam may
    be used. The overall learning rate is not comparable to either method and
    should be determined by a hyper-parameter sweep.
    MADGRAD requires less weight decay than other methods, often as little as
    zero. Momentum values used for SGD or Adam's beta1 should work here also.
    On sparse problems both weight_decay and momentum should be set to 0.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate (default: 1e-2).
        momentum (float):
            Momentum value in  the range [0,1) (default: 0.9).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-6).
    """

    def __init__(
            self, params: _params_t, lr: float = 1e-2, momentum: float = 0.9, weight_decay: float = 0, eps: float = 1e-6,
    ):
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Momentum {momentum} must be in the range [0,1]")
        if lr <= 0:
            raise ValueError(f"Learning rate {lr} must be positive")
        if weight_decay < 0:
            raise ValueError(f"Weight decay {weight_decay} must be non-negative")
        if eps < 0:
            raise ValueError(f"Eps must be non-negative")

        defaults = dict(lr=lr, eps=eps, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return False

    @property
    def supports_flat_params(self) -> bool:
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # step counter must be stored in state to ensure correct behavior under
        # optimizer sharding
        if 'k' not in self.state:
            self.state['k'] = torch.tensor([0], dtype=torch.long)
        k = self.state['k'].item()

        for group in self.param_groups:
            eps = group["eps"]
            lr = group["lr"] + eps
            decay = group["weight_decay"]
            momentum = group["momentum"]

            ck = 1 - momentum
            lamb = lr * math.pow(k + 1, 0.5)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if "grad_sum_sq" not in state:
                    state["grad_sum_sq"] = torch.zeros_like(p.data).detach()
                    state["s"] = torch.zeros_like(p.data).detach()
                    if momentum != 0:
                        state["x0"] = torch.clone(p.data).detach()

                if momentum != 0.0 and grad.is_sparse:
                    raise RuntimeError("momentum != 0 is not compatible with sparse gradients")

                grad_sum_sq = state["grad_sum_sq"]
                s = state["s"]

                # Apply weight decay
                if decay != 0:
                    if grad.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")

                    grad.add_(p.data, alpha=decay)

                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_val = grad._values()

                    p_masked = p.sparse_mask(grad)
                    grad_sum_sq_masked = grad_sum_sq.sparse_mask(grad)
                    s_masked = s.sparse_mask(grad)

                    # Compute x_0 from other known quantities
                    rms_masked_vals = grad_sum_sq_masked._values().pow(1 / 3).add_(eps)
                    x0_masked_vals = p_masked._values().addcdiv(s_masked._values(), rms_masked_vals, value=1)

                    # Dense + sparse op
                    grad_sq = grad * grad
                    grad_sum_sq.add_(grad_sq, alpha=lamb)
                    grad_sum_sq_masked.add_(grad_sq, alpha=lamb)

                    rms_masked_vals = grad_sum_sq_masked._values().pow_(1 / 3).add_(eps)

                    s.add_(grad, alpha=lamb)
                    s_masked._values().add_(grad_val, alpha=lamb)

                    # update masked copy of p
                    p_kp1_masked_vals = x0_masked_vals.addcdiv(s_masked._values(), rms_masked_vals, value=-1)
                    # Copy updated masked p to dense p using an add operation
                    p_masked._values().add_(p_kp1_masked_vals, alpha=-1)
                    p.data.add_(p_masked, alpha=-1)
                else:
                    if momentum == 0:
                        # Compute x_0 from other known quantities
                        rms = grad_sum_sq.pow(1 / 3).add_(eps)
                        x0 = p.data.addcdiv(s, rms, value=1)
                    else:
                        x0 = state["x0"]

                    # Accumulate second moments
                    grad_sum_sq.addcmul_(grad, grad, value=lamb)
                    rms = grad_sum_sq.pow(1 / 3).add_(eps)

                    # Update s
                    s.data.add_(grad, alpha=lamb)

                    # Step
                    if momentum == 0:
                        p.data.copy_(x0.addcdiv(s, rms, value=-1))
                    else:
                        z = x0.addcdiv(s, rms, value=-1)

                        # p is a moving average of z
                        p.data.mul_(1 - ck).add_(z, alpha=ck)

        self.state['k'] += 1
        return loss


def _register_dataset(name: str):
    # def _load():
    #     with open(, 'r') as f:
    #         return json.load(f)
    # DatasetCatalog.register(name, _load)
    register_coco_instances(name, {}, f'{fold_dir}/{name}.json', image_root=f'{data_dir}/data/train_segmentation/images')
    # MetadataCatalog.get(name).set(thing_classes=["Dog", "Cat", "Mouse"])


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(cfg, resume_from):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.setLevel(logging.INFO)
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            # hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint if resume_from is None else resume_from, resume=resume_from is not None)
    if resume_from is not None and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


class AlbumentationsTransform(Transform):
    def __init__(self, aug, param):
        self.aug = aug
        self.param = param

    def apply_image(self, img):
        return self.aug.apply(img, **self.param)

    def apply_coords(self, coords: np.ndarray):
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        return box

    def apply_polygons(self, polygons: list) -> list:
        return polygons


class Albumentations(Augmentation):
    def __init__(self, augmentor):
        self._aug = augmentor

    def get_transform(self, img):
        return AlbumentationsTransform(self._aug, self._aug.get_params())


if __name__ == '__main__':
    setup_logger()
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--fold_n", type=int)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--output", type=str, default="detectron")
    parser.add_argument('--resume_from')
    args = parser.parse_args()
    data_dir = args.data_dir
    fold_n = args.fold_n
    fold_dir = f'{data_dir}/data/train_segmentation/folds/{fold_n}'
    _register_dataset('train')
    _register_dataset('val')

    cfg = get_config("new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_400ep_LSJ.py")

    cfg.dataloader.train.mapper.augmentations[0].min_scale = 0.2
    cfg.dataloader.train.mapper.augmentations.insert(0, LazyCall(RandomRotation)(angle=[-8, 8]))
    cfg.dataloader.train.mapper.augmentations.append(LazyCall(Albumentations)(augmentor=A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5)))
    cfg.dataloader.train.mapper.augmentations.append(LazyCall(Albumentations)(augmentor=A.CLAHE(clip_limit=1.5, p=0.15)))
    cfg.dataloader.train.mapper.augmentations.append(LazyCall(Albumentations)(augmentor=A.RandomShadow(p=0.5)))
    # cfg.dataloader.train.mapper.augmentations.insert(2, LazyCall(Albumentations)(augmentor=A.OpticalDistortion(distort_limit=1.0, p=0.5)))
    cfg.dataloader.train.mapper.augmentations.append(LazyCall(Albumentations)(augmentor=A.Blur(blur_limit=1.5, p=0.3)))
    # cfg.dataloader.train.mapper.augmentations.append(LazyCall(Albumentations)(augmentor=A.MotionBlur(blur_limit=(3, 6), p=0.3)))

    cfg.train.output_dir = f'{args.output}_fold_{fold_n}'
    cfg.dataloader.train.dataset.names = ("train",)
    cfg.dataloader.train.num_workers = args.workers
    cfg.dataloader.train.total_batch_size = args.bs
    cfg.dataloader.test.dataset.names = ("val",)
    cfg.dataloader.test.num_workers = args.workers
    cfg.dataloader.evaluator.dataset_name = 'val'
    cfg.dataloader.evaluator.output_dir = cfg.train.output_dir

    cfg.model.roi_heads.num_classes = 1

    def _get_ln(x):
        return LayerNorm(x, eps=1e-6, data_format="channels_first")

    # cfg.model.backbone.norm = _get_ln
    # cfg.model.roi_heads.mask_head.conv_norm = _get_ln
    # cfg.model.roi_heads.box_head.conv_norm = _get_ln
    cfg.model.backbone.norm = 'BN'
    cfg.model.roi_heads.mask_head.conv_norm = 'BN'
    cfg.model.roi_heads.box_head.conv_norm = 'BN'
    cfg.model.backbone.bottom_up = LazyCall(convnext_small)(
        pretrained=True,
        drop_path_rate=0.2,
        layer_scale_init_value=1.0,
    )
    cfg.model.backbone.in_features = ['out_0', 'out_1', 'out_2', 'out_3']

    cfg.model.pixel_mean = [0.678 * 255, 0.690 * 255, 0.701 * 255]
    cfg.model.pixel_std = [0.154 * 255, 0.163 * 255, 0.162 * 255]
    cfg.model.roi_heads.box_predictor.test_topk_per_image = 900

    # cfg.model.proposal_generator.head.num_anchors = 4
    # cfg.model.proposal_generator.anchor_generator.aspect_ratios = [0.25, 0.5, 1.0, 2.0]  # 1:4, 1:2, 1:1, 2:1
    #
    # cfg.model.proposal_generator.anchor_generator.sizes = [[16], [32], [64], [128], [256], [512]]
    # cfg.model.proposal_generator.in_features = ["p2", "p2", "p3", "p4", "p5", "p6"]
    # cfg.model.proposal_generator.anchor_generator.strides = [4, 4, 8, 16, 32, 64]

    cfg.model.proposal_generator.pre_nms_topk = {True: 4000, False: 4000}
    cfg.model.proposal_generator.post_nms_topk = {True: 2000, False: 2000}

    cfg.train.max_iter = args.iters
    cfg.train.checkpointer.period = 500
    cfg.train.init_checkpoint = model_zoo.get_checkpoint_url("new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_400ep_LSJ.py")
    cfg.train.eval_period = 5000
    # cfg.optimizer.lr = 0.001
    cfg.optimizer = LazyCall(MADGRAD)(params=LazyCall(get_default_optimizer_params)(), lr=0.001)
    # cfg.optimizer = LazyCall(MADGRAD)(params=LazyCall(get_default_optimizer_params)(), lr=0.0005)  # tuning
    cfg.lr_multiplier.warmup_factor = 0.001
    cfg.lr_multiplier.warmup_length = 0.05
    cfg.lr_multiplier.scheduler = LazyCall(CosineParamScheduler)(start_value=1.0, end_value=1e-8)

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    do_train(cfg, resume_from=args.resume_from)
