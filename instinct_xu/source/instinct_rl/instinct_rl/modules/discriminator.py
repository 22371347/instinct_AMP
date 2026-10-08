from __future__ import annotations

import importlib
from typing import Dict

import torch.nn as nn

import instinct_rl.modules as instinct_modules
from instinct_rl.modules.mlp import MlpModel
from instinct_rl.modules.normalizer import (
    EmpiricalDiscountedVariationNormalization,
    EmpiricalNormalization,
)
from instinct_rl.utils.utils import get_subobs_size


class Discriminator(nn.Module):
    """A general implementation of returning +/- value to discriminate the input sequence."""

    is_recurrent = False

    def __init__(
        self,
        input_segment: dict[str, tuple],
        output_size=1,
        output_nonlinearity: str = None,
        normalizer_class_name: str = None,  # if provided, the normalizer will be used to normalize the input
        normalizer_kwargs: dict = None,
        encoder_class_name: str = None,
        encoder_configs: dict[str, dict] = None,
        **kwargs,
    ):
        """
        Args:
            output_size: (int) the output size of the model.
            output_nonlinearity: (str) the nonlinearity of the output layer.
            normalizer_class_name: (str) the class name of the normalizer. If None, no normalizer will be used.
            encoder_configs: Dict[str, dict]: encoder configs for the input segment, requiring the following keys:
                - class_name: (str) the class name of the ParallelLayer or the customed encoder class.
                - {name}: {dict} encoder configs to build each specific encoder. The required keys of this dict are referred to
                    `instinct_rl.modules.parallel_layer.ParallelLayer.__init__`.
        """
        super().__init__()

        self.input_segment = input_segment
        self.output_size = output_size
        self.output_nonlinearity = output_nonlinearity
        self.normalizer_class_name = normalizer_class_name
        self.normalizer_kwargs = normalizer_kwargs
        self.encoder_class_name = encoder_class_name
        self.encoder_configs = encoder_configs

        assert (
            encoder_configs is None or normalizer_class_name is None
        ), "The encoder_configs and normalizer_class_name cannot be used at the same time (currently)."
        self.normalizer = instinct_modules.build_normalizer(
            input_shape=(
                get_subobs_size(self.input_segment)
                if len(self.input_segment) > 1
                else list(self.input_segment.values())[0]
            ),  # in case of image-like non-flattened tensor
            normalizer_class_name=normalizer_class_name,
            normalizer_kwargs=normalizer_kwargs,
        )
        self.build_model(input_segment, kwargs)

        print(f"Discriminator Network Structure: {self}")

    def build_model(self, input_segment, kwargs):
        """Build the model. The model is a MLP model by default."""
        if self.encoder_configs is not None:
            EncoderClass = (
                getattr(
                    importlib.import_module(self.encoder_class_name.split(":")[0]),
                    self.encoder_class_name.split(":")[1],
                )
                if ":" in self.encoder_class_name
                else getattr(instinct_modules, self.encoder_class_name)
            )
            encoders = EncoderClass(
                input_segment=input_segment,
                block_configs=self.encoder_configs,
            )
            input_segment = encoders.output_segment
            self.encoders = encoders
        else:
            self.encoders = None
        # Build the output model (head).
        self.model = MlpModel(
            input_size=get_subobs_size(input_segment),
            output_size=self.output_size,
            **kwargs,
        )
        if self.output_nonlinearity is not None:
            self.model = nn.Sequential(
                self.model,
                getattr(nn, self.output_nonlinearity)(),
            )

    def backbone_run(self, x):
        if self.normalizer is not None:
            # The normalizer will determine and update the mean and std of the input data by itself.
            x = self.normalizer(x)
        return self.model(x)

    def forward(self, x, hidden_states=None, masks=None, update_normalizer: bool = True):
        """NOTE: saving hidden_states and masks for future recurrent models. (Maybe not necessary)

        Args:
            update_normalizer: 是否允许 normalizer 在本次前向时更新统计量。
                默认 `True`（= 上游行为）；AMP 判别器的"状态白化"改造（2026-10-03）里，
                **actor 侧前向应传 `False`** —— 我们希望统计量只由 expert/参考侧累积
                （对齐 AMP_mjlab 的 `amp_normalizer`），否则策略越跑偏、统计量越被带偏。
        """
        if self.encoders is not None:
            # The encoders will determine and update the input segment by itself.
            x = self.encoders(x)
        if self.normalizer is not None:
            # The normalizer will determine and update the mean and std of the input data by itself.
            x = self.normalizer(x) if update_normalizer else self.normalize_input(x)
        return self.model(x)

    def normalize_input(self, x):
        """把**原始**输入白化成网络真正吃到的张量；**不更新统计量**。

        没接 normalizer 时原样返回。供两类调用方使用：
        * `forward(..., update_normalizer=False)`：避免 actor 侧污染统计量；
        * **GP 在归一化空间求梯度**（`‖∂d/∂x̃‖`）：若在原始空间求，则
          `∂d/∂x = (∂d/∂x̃)/σ`，而 `σ` 在近常数维上→0 ⇒ `‖∇d‖` 会从 ~5e-3 爆到 10²~10³，
          GP 项直接支配损失（实测见 `RESULTS_amp_vs_mjlab_params.md` §F）。
        """
        if self.normalizer is None:
            return x
        fn = getattr(self.normalizer, "normalize", None)
        if fn is not None:
            return fn(x, update=False)
        # 兼容其它 normalizer 实现（没有 normalize 方法）：临时切 eval 再前向（forward 只在 training 时更新）
        was_training = self.normalizer.training
        self.normalizer.eval()
        try:
            with torch.no_grad():
                return self.normalizer(x)
        finally:
            self.normalizer.train(was_training)

    def forward_normalized(self, x, hidden_states=None, masks=None):
        """输入**已白化**时跳过 normalizer 的前向（供 GP 在归一化空间求梯度用）。

        与 `forward(x, update_normalizer=False)` 的区别：后者会把 `x` **再**归一化一次。
        """
        if self.encoders is not None:
            x = self.encoders(x)
        return self.model(x)

    def logit_layer_weights(self):
        """Get the weights of the last layer of the discriminator."""
        # NOTE: This logic is not very general, only works for MlpModel as discriminator's backbone.
        model = self.model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        if isinstance(model, nn.Sequential):
            model = model[0]
        return model.model[-1].weight
