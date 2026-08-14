"""第一阶段的数据加载、训练、验证、记录与可视化工具。"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from spikingjelly.activation_based import functional, neuron
from torch import nn
from torch.utils.data import DataLoader

from data import INDEX_TO_LABEL, LABELS, STEMNISTDataset


def set_random_seed(seed: int = 42, deterministic: bool = True) -> None:
    """固定 Python、NumPy、PyTorch 和 CUDA 的随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device() -> torch.device:
    """优先选择 CUDA，否则使用 CPU。"""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_environment_info() -> dict[str, Any]:
    """收集复现实验所需的软件和硬件环境信息。"""

    package_names = [
        "torch",
        "spikingjelly",
        "numpy",
        "pandas",
        "h5py",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ]
    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "未安装"

    cuda_info: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        cuda_info["devices"] = [
            {
                "name": torch.cuda.get_device_name(index),
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(index).total_memory / 1024**3,
                    3,
                ),
            }
            for index in range(torch.cuda.device_count())
        ]

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "cuda": cuda_info,
    }


def write_json(content: Any, file_path: str | Path) -> None:
    """以 UTF-8 写入可读的 JSON。"""

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        raise TypeError(f"无法序列化类型：{type(value)!r}")

    file_path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, default=convert),
        encoding="utf-8",
    )


def make_balanced_records(
    records: pd.DataFrame,
    samples_per_class: int,
    seed: int = 42,
) -> pd.DataFrame:
    """从每个类别抽取相同数量的样本。"""

    class_counts = records["label_index"].value_counts()
    if (class_counts < samples_per_class).any():
        raise ValueError("至少一个类别的样本数不足，无法构造平衡子集。")
    sampled = (
        records.groupby("label_index", group_keys=False)
        .sample(n=samples_per_class, random_state=seed)
        .sort_values(["label_index", "sample_id"])
        .reset_index(drop=True)
    )
    return sampled


def create_dataloaders(
    splits: dict[str, pd.DataFrame],
    time_steps: int,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
    smoke_test: bool = False,
    smoke_samples_per_class: dict[str, int] | None = None,
) -> tuple[dict[str, DataLoader], dict[str, pd.DataFrame]]:
    """根据固定划分创建训练、验证和测试 DataLoader。"""

    selected_records: dict[str, pd.DataFrame] = {}
    smoke_samples_per_class = smoke_samples_per_class or {
        "train": 2,
        "val": 1,
        "test": 1,
    }
    for split_name, records in splits.items():
        if smoke_test:
            selected_records[split_name] = make_balanced_records(
                records,
                samples_per_class=smoke_samples_per_class[split_name],
                seed=seed,
            )
        else:
            selected_records[split_name] = records.reset_index(drop=True).copy()

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    loaders: dict[str, DataLoader] = {}
    for split_name, records in selected_records.items():
        dataset = STEMNISTDataset(records, time_steps=time_steps)
        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": split_name == "train",
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "generator": generator,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
        loaders[split_name] = DataLoader(**loader_kwargs)
    return loaders, selected_records


def _prepare_batch(
    batch: tuple[torch.Tensor, torch.Tensor, tuple[str, ...]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """将 `[N,T,C,H,W]` 批次转为模型需要的 `[T,N,C,H,W]`。"""

    samples, labels, sample_ids = batch
    samples = samples.to(device, non_blocking=True)
    samples = samples.permute(1, 0, 2, 3, 4).contiguous()
    labels = labels.to(device, non_blocking=True)
    return samples, labels, list(sample_ids)


def _check_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    """发现 NaN 或 Inf 时立即终止，避免产生无效结果。"""

    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} 中出现 NaN 或 Inf。")


def _check_finite_gradients(model: nn.Module) -> None:
    """检查所有已生成的参数梯度。"""

    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"参数 {name} 的梯度中出现 NaN 或 Inf。")


def _classification_metrics(
    targets: list[int],
    predictions: list[int],
) -> dict[str, float]:
    """计算整体 Accuracy 和 macro-F1。"""

    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=list(range(len(LABELS))),
                average="macro",
                zero_division=0,
            )
        ),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """执行一轮训练或无梯度评估，并统计分类和发放率指标。"""

    is_training = optimizer is not None
    model.train(is_training)
    if is_training:
        optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_loss = 0.0
    total_samples = 0
    targets: list[int] = []
    predictions: list[int] = []
    sample_ids: list[str] = []
    firing_rate_sums = {"lif1": 0.0, "lif2": 0.0, "output_lif": 0.0}
    start_time = time.perf_counter()

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs, labels, batch_sample_ids = _prepare_batch(batch, device)
        batch_size = labels.shape[0]

        try:
            with torch.set_grad_enabled(is_training):
                spike_counts, firing_rates = model(
                    inputs,
                    return_firing_rates=True,
                )
                _check_finite_tensor(spike_counts, "模型输出")
                loss = criterion(spike_counts.float(), labels)
                _check_finite_tensor(loss, "损失")

                if is_training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    _check_finite_gradients(model)
                    optimizer.step()

            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size
            batch_predictions = spike_counts.detach().argmax(dim=1)
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            sample_ids.extend(batch_sample_ids)
            for name, rate in firing_rates.items():
                firing_rate_sums[name] += float(rate) * batch_size
        finally:
            functional.reset_net(model)

    if total_samples == 0:
        raise RuntimeError("本轮没有处理任何样本。")

    elapsed_seconds = time.perf_counter() - start_time
    metrics: dict[str, Any] = {
        "loss": total_loss / total_samples,
        **_classification_metrics(targets, predictions),
        "samples": total_samples,
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": total_samples / elapsed_seconds,
        "frames_per_second": (
            total_samples * loader.dataset.time_steps / elapsed_seconds
        ),
        "firing_rates": {
            name: value / total_samples for name, value in firing_rate_sums.items()
        },
        "targets": targets,
        "predictions": predictions,
        "sample_ids": sample_ids,
    }
    metrics["peak_memory_mb"] = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """训练一个 epoch。"""

    return run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        optimizer=optimizer,
        max_batches=max_batches,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    detailed: bool = False,
) -> dict[str, Any]:
    """执行验证或测试，并可选生成逐类指标。"""

    metrics = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        max_batches=max_batches,
    )
    if detailed:
        labels = list(range(len(LABELS)))
        metrics["confusion_matrix"] = confusion_matrix(
            metrics["targets"],
            metrics["predictions"],
            labels=labels,
        )
        metrics["classification_report"] = classification_report(
            metrics["targets"],
            metrics["predictions"],
            labels=labels,
            target_names=list(LABELS),
            output_dict=True,
            zero_division=0,
        )
    return metrics


def _checkpoint_content(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
) -> dict[str, Any]:
    """组织可恢复的 checkpoint 内容。"""

    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "train_metrics": {
            key: value
            for key, value in train_metrics.items()
            if key not in {"targets", "predictions", "sample_ids"}
        },
        "val_metrics": {
            key: value
            for key, value in val_metrics.items()
            if key not in {"targets", "predictions", "sample_ids"}
        },
    }


def fit(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    output_dir: str | Path,
    config: dict[str, Any],
    epochs: int,
    learning_rate: float = 0.005,
    weight_decay: float = 1e-4,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> pd.DataFrame:
    """训练模型，以验证集 macro-F1 保存最佳 checkpoint。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    write_json(config, output_dir / "config.json")
    write_json(collect_environment_info(), output_dir / "environment.json")

    history_rows: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            max_batches=max_train_batches,
        )
        val_metrics = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            max_batches=max_eval_batches,
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "train_seconds": train_metrics["elapsed_seconds"],
            "val_seconds": val_metrics["elapsed_seconds"],
            "train_samples_per_second": train_metrics["samples_per_second"],
            "val_samples_per_second": val_metrics["samples_per_second"],
            "peak_memory_mb": max(
                train_metrics["peak_memory_mb"],
                val_metrics["peak_memory_mb"],
            ),
            **{
                f"train_rate_{name}": value
                for name, value in train_metrics["firing_rates"].items()
            },
            **{
                f"val_rate_{name}": value
                for name, value in val_metrics["firing_rates"].items()
            },
        }
        history_rows.append(history_row)
        checkpoint = _checkpoint_content(
            model,
            optimizer,
            epoch,
            config,
            train_metrics,
            val_metrics,
        )
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(checkpoint, output_dir / "best.pt")

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train loss {train_metrics['loss']:.4f} | "
            f"Val loss {val_metrics['loss']:.4f} | "
            f"Val accuracy {val_metrics['accuracy']:.4f} | "
            f"Val macro-F1 {val_metrics['macro_f1']:.4f}"
        )

    history = pd.DataFrame(history_rows)
    history.to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")
    return history


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """恢复模型权重并返回 checkpoint 内容。"""

    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    functional.reset_net(model)
    return checkpoint


def save_evaluation(
    metrics: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "test",
) -> None:
    """保存指标 JSON 和逐样本预测 CSV。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_without_samples = {
        key: value
        for key, value in metrics.items()
        if key not in {"targets", "predictions", "sample_ids"}
    }
    write_json(metrics_without_samples, output_dir / f"{prefix}_metrics.json")

    predictions = pd.DataFrame(
        {
            "sample_id": metrics["sample_ids"],
            "target_index": metrics["targets"],
            "target_label": [
                INDEX_TO_LABEL[index] for index in metrics["targets"]
            ],
            "prediction_index": metrics["predictions"],
            "prediction_label": [
                INDEX_TO_LABEL[index] for index in metrics["predictions"]
            ],
        }
    )
    predictions.to_csv(
        output_dir / f"{prefix}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )


def network_state_is_reset(model: nn.Module) -> bool:
    """检查所有 LIF 神经元的膜电位是否处于复位值。"""

    for module in model.modules():
        if not isinstance(module, neuron.LIFNode):
            continue
        voltage = module.v
        if isinstance(voltage, torch.Tensor):
            if not torch.allclose(voltage, torch.zeros_like(voltage)):
                return False
        elif float(voltage) != 0.0:
            return False
    return True


def compare_single_and_multi_step(
    model: nn.Module,
    inputs: torch.Tensor,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """在关闭 Dropout 的评估模式下比较单步循环与多步输出。"""

    was_training = model.training
    model.eval()
    functional.reset_net(model)
    single_step_model = copy.deepcopy(model)
    functional.set_step_mode(single_step_model, "s")
    single_step_model.eval()
    functional.reset_net(single_step_model)

    try:
        with torch.no_grad():
            multi_output, _ = model.forward_sequence(inputs)
            single_outputs = []
            for time_index in range(inputs.shape[0]):
                step_output, _ = single_step_model.forward_single_step(
                    inputs[time_index]
                )
                single_outputs.append(step_output)
            single_output = torch.stack(single_outputs, dim=0)
        maximum_error = float((multi_output - single_output).abs().max())
        matches = bool(
            torch.allclose(multi_output, single_output, atol=atol, rtol=rtol)
        )
    finally:
        functional.reset_net(model)
        functional.reset_net(single_step_model)
        model.train(was_training)

    return {
        "matches": matches,
        "maximum_absolute_error": maximum_error,
        "shape": list(multi_output.shape),
    }


def _finish_figure(
    figure: plt.Figure,
    save_path: str | Path | None,
    show: bool,
) -> None:
    """统一保存、显示并关闭图像。"""

    figure.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def plot_pressure_sequence(
    sample: torch.Tensor | np.ndarray,
    label: str,
    save_path: str | Path | None = None,
    show: bool = True,
    number_of_frames: int = 6,
) -> None:
    """显示压力序列中均匀抽取的若干时间步。"""

    values = (
        sample.detach().cpu().numpy()
        if isinstance(sample, torch.Tensor)
        else np.asarray(sample)
    )
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    frame_indices = np.linspace(
        0,
        values.shape[0] - 1,
        number_of_frames,
        dtype=int,
    )
    figure, all_axes = plt.subplots(
        1,
        number_of_frames + 1,
        figsize=(16, 3.0),
        gridspec_kw={
            "width_ratios": [1] * number_of_frames + [0.06],
        },
    )
    axes = all_axes[:number_of_frames]
    color_axis = all_axes[-1]
    image = None
    for axis, frame_index in zip(axes, frame_indices):
        image = axis.imshow(values[frame_index], cmap="viridis", vmin=0, vmax=255)
        axis.set_title(f"Step {frame_index + 1}")
        axis.set_axis_off()
    figure.suptitle(f"Pressure Evolution — Label {label}", fontsize=13)
    if image is not None:
        figure.colorbar(
            image,
            cax=color_axis,
            label="Pressure (ADC)",
        )
    _finish_figure(figure, save_path, show)


def plot_class_distribution(
    split_records: dict[str, pd.DataFrame],
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """绘制训练、验证和测试集的类别数量。"""

    figure, axis = plt.subplots(figsize=(14, 5))
    x_positions = np.arange(len(LABELS))
    width = 0.25
    colors = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}
    for offset, split_name in enumerate(("train", "val", "test")):
        counts = (
            split_records[split_name]["label"]
            .value_counts()
            .reindex(LABELS, fill_value=0)
        )
        axis.bar(
            x_positions + (offset - 1) * width,
            counts.values,
            width=width,
            label=split_name.title(),
            color=colors[split_name],
        )
    axis.set_title("Class Distribution by Split")
    axis.set_xlabel("Class")
    axis.set_ylabel("Number of Samples")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(LABELS)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _finish_figure(figure, save_path, show)


def plot_training_history(
    history: pd.DataFrame,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """绘制训练与验证的 Loss、Accuracy 和 macro-F1。"""

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
    ]
    for axis, (metric_key, metric_label) in zip(axes, panels):
        axis.plot(
            history["epoch"],
            history[f"train_{metric_key}"],
            marker="o",
            label="Train",
        )
        axis.plot(
            history["epoch"],
            history[f"val_{metric_key}"],
            marker="o",
            label="Validation",
        )
        axis.set_title(metric_label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric_label)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Training History", fontsize=13)
    _finish_figure(figure, save_path, show)


def plot_confusion_matrix(
    matrix: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """绘制固定标签顺序的 35 类混淆矩阵。"""

    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        np.asarray(matrix),
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        square=True,
        cbar_kws={"label": "Number of Samples"},
        ax=axis,
    )
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("True Class")
    _finish_figure(figure, save_path, show)
