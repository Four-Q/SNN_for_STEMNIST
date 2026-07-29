"""第一阶段使用的多步卷积脉冲神经网络。"""

from __future__ import annotations

from typing import Any

import torch
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from torch import nn


class ConvSNN(nn.Module):
    """用于 35 类 STEMNIST 识别的单通道 Conv-SNN。"""

    EXPECTED_PARAMETER_COUNT = 6_379

    def __init__(
        self,
        num_classes: int = 35,
        dropout: float = 0.3,
        tau: float = 10.0,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_probability = dropout
        self.tau = tau
        self.backend = backend

        self.conv1 = layer.Conv2d(1, 8, kernel_size=4, step_mode="m")
        self.lif1 = self._make_lif()
        self.pool1 = layer.MaxPool2d(kernel_size=2, step_mode="m")

        self.conv2 = layer.Conv2d(
            8,
            16,
            kernel_size=3,
            padding=1,
            step_mode="m",
        )
        self.lif2 = self._make_lif()
        self.pool2 = layer.MaxPool2d(kernel_size=2, step_mode="m")

        self.flatten = layer.Flatten(step_mode="m")
        self.dropout = layer.Dropout(dropout, step_mode="m")
        self.classifier = layer.Linear(16 * 3 * 3, num_classes, step_mode="m")
        self.output_lif = self._make_lif()

        functional.set_step_mode(self, "m")
        functional.set_backend(self, backend, instance=neuron.LIFNode)

    def _make_lif(self) -> neuron.LIFNode:
        """创建配置一致的 LIF 神经元。"""

        return neuron.LIFNode(
            tau=self.tau,
            decay_input=False,
            surrogate_function=surrogate.ATan(),
            step_mode="m",
            backend="torch",
        )

    def _forward_layers(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """依次执行网络各层，并保留三层脉冲用于统计。"""

        hidden1 = self.lif1(self.conv1(inputs))
        hidden2 = self.lif2(self.conv2(self.pool1(hidden1)))
        features = self.flatten(self.pool2(hidden2))
        output_spikes = self.output_lif(self.classifier(self.dropout(features)))
        firing_rates = {
            "lif1": hidden1.detach().float().mean(),
            "lif2": hidden2.detach().float().mean(),
            "output_lif": output_spikes.detach().float().mean(),
        }
        return output_spikes, firing_rates

    def forward_sequence(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """处理 `[T,N,1,16,16]` 输入并返回逐时间步输出脉冲。"""

        if inputs.ndim != 5:
            raise ValueError("多步输入必须是 [T,N,C,H,W] 五维张量。")
        if tuple(inputs.shape[2:]) != (1, 16, 16):
            raise ValueError("输入的通道和空间形状必须是 [1,16,16]。")
        return self._forward_layers(inputs)

    def forward_single_step(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """在单步模式比较测试中处理一个 `[N,1,16,16]` 时间步。"""

        if inputs.ndim != 4:
            raise ValueError("单步输入必须是 [N,C,H,W] 四维张量。")
        return self._forward_layers(inputs)

    def forward(
        self,
        inputs: torch.Tensor,
        return_firing_rates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """返回沿时间维累加的分类脉冲计数。"""

        output_spikes, firing_rates = self.forward_sequence(inputs)
        spike_counts = output_spikes.sum(dim=0)
        if return_firing_rates:
            return spike_counts, firing_rates
        return spike_counts

    def parameter_count(self) -> int:
        """返回所有可训练参数的总数。"""

        return sum(parameter.numel() for parameter in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"dropout={self.dropout_probability}, "
            f"tau={self.tau}, backend={self.backend!r}"
        )


def build_model(**kwargs: Any) -> ConvSNN:
    """创建模型并立即检查参数量。"""

    model = ConvSNN(**kwargs)
    parameter_count = model.parameter_count()
    if model.num_classes == 35 and parameter_count != model.EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"模型参数量应为 {model.EXPECTED_PARAMETER_COUNT}，"
            f"实际为 {parameter_count}。"
        )
    return model
