# 数据处理Pipeline，用于提供DataLoader
from collections import Counter
from hashlib import md5 # 用于检查原始ZIP文件是否完好
from pathlib import Path
from zipfile import ZipFile

import h5py # 用于读取.h5文件
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

# 数据路径
MYCODE_DIR = Path(__file__).resolve().parent
# SNN_for_STEMNIST/first_stage
FIRST_STAGE_DIR = MYCODE_DIR.parent

# SNN_for_STEMNIST/first_stage/STEMNIST
DATA_ROOT = FIRST_STAGE_DIR / "STEMNIST"
ZIP_PATH = DATA_ROOT / "STEMNIST Dataset.zip"
EXTRACT_ROOT = DATA_ROOT / "extracted"
RAW_DIR = EXTRACT_ROOT / "STEMNIST Dataset" / "RawCharacters"

# 预期的ZIP文件MD5值，用于验证文件完整性
EXPECTED_MD5 = "6ca4638b2f95bf34f59873ab62399bd8"   
RAW_TIME_STEPS = 240
EXPECTED_SHAPE = (RAW_TIME_STEPS, 16, 16)

# 标签顺序一旦确定，之后不能改变
LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789")
LABEL_TO_INDEX = {
    label: index
    for index, label in enumerate(LABELS)
}


class NormalizePressure:
    """使用训练集估计的逐传感器基线归一化压力序列。

    ``baseline`` 的形状为 ``[1,1,16,16]``，可以广播到单个样本的
    ``[T,1,16,16]``。归一化后的数据为 float32，范围为 ``[0,1]``。
    """

    def __init__(self, baseline, adc_max=255.0):
        baseline = torch.as_tensor(
            baseline,
            dtype=torch.float32,
        ).detach().clone()

        if tuple(baseline.shape) != (1, 1, 16, 16):
            raise ValueError(
                "baseline 形状必须是 [1,1,16,16]，"
                f"实际为 {tuple(baseline.shape)}"
            )

        if not torch.isfinite(baseline).all():
            raise ValueError("baseline 中存在非有限值")

        self.baseline = baseline
        self.adc_max = float(adc_max)
        self.span = self.adc_max - self.baseline

        if (self.span <= 0.0).any():
            raise ValueError(
                "adc_max 必须大于每个传感器的 baseline"
            )

    def __call__(self, pressure):
        if tuple(pressure.shape[1:]) != (1, 16, 16):
            raise ValueError(
                "pressure 形状必须是 [T,1,16,16]，"
                f"实际为 {tuple(pressure.shape)}"
            )

        pressure = pressure.to(dtype=torch.float32)

        return (
            (pressure - self.baseline) / self.span
        ).clamp_(0.0, 1.0)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"baseline_min={self.baseline.min().item():.1f}, "
            f"baseline_max={self.baseline.max().item():.1f}, "
            f"adc_max={self.adc_max:.1f})"
        )


def estimate_training_pressure_baseline(
    pressures,
    train_indices,
    batch_size=64,
):
    """仅使用训练集，以众数估计每个传感器的静止 ADC 基线。

    参数：
        pressures:
            完整数据张量，形状 ``[N,T,1,16,16]``，类型 uint8。
        train_indices:
            训练集在完整数据张量中的索引。
        batch_size:
            计算直方图时每次处理的训练样本数。

    返回：
        float32 张量，形状为 ``[1,1,16,16]``。

    每个传感器分别统计 0～255 的出现次数。书写压力只在较少时间和
    空间位置出现，因此众数比均值更适合作为无压力状态的基线。
    """

    if pressures.ndim != 5:
        raise ValueError(
            "pressures 必须是 [N,T,C,H,W] 五维张量"
        )
    if tuple(pressures.shape[2:]) != (1, 16, 16):
        raise ValueError(
            "pressures 的通道和空间形状必须是 [1,16,16]"
        )
    if pressures.dtype != torch.uint8:
        raise TypeError(
            "基线估计要求 pressures 为 torch.uint8，"
            f"实际为 {pressures.dtype}"
        )
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")

    train_indices = torch.as_tensor(
        train_indices,
        dtype=torch.long,
    ).flatten()

    if train_indices.numel() == 0:
        raise ValueError("train_indices 不能为空")
    if train_indices.min() < 0 or train_indices.max() >= len(pressures):
        raise IndexError("train_indices 中存在越界索引")

    sensor_count = 16 * 16
    adc_levels = 256
    histogram = torch.zeros(
        (sensor_count, adc_levels),
        dtype=torch.int64,
    )
    sensor_offsets = (
        torch.arange(sensor_count, dtype=torch.int64)
        .mul(adc_levels)
        .unsqueeze(1)
    )

    for index_batch in train_indices.split(batch_size):
        # [B,T,1,16,16] -> [256,B*T]
        values = (
            pressures[index_batch, :, 0]
            .permute(2, 3, 0, 1)
            .reshape(sensor_count, -1)
            .to(dtype=torch.int64)
        )

        encoded_values = values + sensor_offsets
        batch_histogram = torch.bincount(
            encoded_values.flatten(),
            minlength=sensor_count * adc_levels,
        ).reshape(sensor_count, adc_levels)

        histogram.add_(batch_histogram)

    baseline = (
        histogram.argmax(dim=1)
        .to(dtype=torch.float32)
        .reshape(1, 1, 16, 16)
    )

    return baseline


# 检查时间步长是否合规
def validate_time_steps(time_steps):
    """
    检查目标时间步数。

    当前实现使用等宽时间窗口平均，因此要求
    time_steps 能够整除原始的 240 帧。
    """

    if isinstance(time_steps, bool) or not isinstance(time_steps, int):
        raise TypeError("time_steps 必须是整数")

    if not 1 <= time_steps <= RAW_TIME_STEPS:
        raise ValueError(
            f"time_steps 必须位于 [1, {RAW_TIME_STEPS}]，"
            f"实际为 {time_steps}"
        )

    if RAW_TIME_STEPS % time_steps != 0:
        raise ValueError(
            f"当前实现要求 time_steps 能整除 {RAW_TIME_STEPS}，"
            f"实际为 {time_steps}"
        )

    return time_steps

# 将压力数据在时间维度上进行聚合
def aggregate_pressure_time(pressure, time_steps):
    """
    将原始的 [240,16,16] 压力序列聚合为
    [time_steps,16,16]。

    例如 time_steps=120 时，每相邻两帧取平均：

        [240,16,16]
            ->
        [120,2,16,16]
            ->
        [120,16,16]

    聚合后四舍五入并转回 uint8，以维持当前数据加载器
    的低内存设计。
    """

    time_steps = validate_time_steps(time_steps)

    if tuple(pressure.shape) != EXPECTED_SHAPE:
        raise ValueError(
            f"压力数组形状必须是 {EXPECTED_SHAPE}，"
            f"实际为 {tuple(pressure.shape)}"
        )

    if pressure.dtype != np.uint8:
        raise TypeError(
            "压力数组类型必须是 np.uint8，"
            f"实际为 {pressure.dtype}"
        )

    if time_steps == RAW_TIME_STEPS:
        return pressure

    group_size = RAW_TIME_STEPS // time_steps

    # 先转换为 uint32，避免多帧相加时 uint8 溢出。
    pressure_sums = (
        pressure.astype(np.uint32, copy=False)
        .reshape(
            time_steps,
            group_size,
            16,
            16,
        )
        .sum(axis=1)
    )

    # 整数四舍五入：
    # time_steps=120 时，等价于 (frame_0 + frame_1 + 1) // 2。
    aggregated_pressure = (
        (pressure_sums + group_size // 2)
        // group_size
    ).astype(np.uint8)

    return aggregated_pressure

###########################################################################

# 根据Zip数据文件计算其MD5值，用于验证文件完整性
def calculate_md5(path):
    """分块计算 ZIP 文件的 MD5。"""

    digest = md5()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()

def prepare_and_scan_samples():
    """校验、解压并扫描所有原始压力样本。"""

    if not ZIP_PATH.is_file():
        raise FileNotFoundError(
            f"找不到数据压缩包：{ZIP_PATH.resolve()}"
        )

    actual_md5 = calculate_md5(ZIP_PATH)

    if actual_md5 != EXPECTED_MD5:
        raise RuntimeError(
            f"ZIP 文件 MD5 校验失败：\n"
            f"期望：{EXPECTED_MD5}\n"
            f"实际：{actual_md5}"
        )

    # 只有尚未解压时才执行解压
    if not RAW_DIR.is_dir():
        EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

        with ZipFile(ZIP_PATH, "r") as archive:
            archive.extractall(EXTRACT_ROOT)

    paths = sorted(RAW_DIR.glob("*.h5"))

    if len(paths) != 7700:
        raise RuntimeError(
            f"应有 7700 个原始样本，实际找到 {len(paths)} 个"
        )

    records = []

    for path in paths:
        # 例如：AT_A_1.h5
        parts = path.stem.split("_")

        if len(parts) != 3:
            raise RuntimeError(f"无法解析文件名：{path.name}")

        participant_id, label, repetition = parts

        if label not in LABEL_TO_INDEX:
            raise RuntimeError(
                f"{path.name} 中出现未知标签：{label}"
            )

        # 先检查 HDF5 的内部结构
        with h5py.File(path, "r") as file:
            if "pressure_data" not in file:
                raise RuntimeError(
                    f"{path.name} 缺少 pressure_data"
                )

            pressure_dataset = file["pressure_data"]

            if pressure_dataset.shape != EXPECTED_SHAPE:
                raise RuntimeError(
                    f"{path.name} 形状错误："
                    f"{pressure_dataset.shape}"
                )

            if pressure_dataset.dtype != np.uint8:
                raise RuntimeError(
                    f"{path.name} 类型错误："
                    f"{pressure_dataset.dtype}"
                )

        records.append(
            {
                "sample_id": path.stem, # AT_A_1
                "participant_id": participant_id,
                "label": label,
                "label_index": LABEL_TO_INDEX[label],
                "repetition": int(repetition),
                "path": path,
            }
        )

    class_counts = Counter(
        record["label"] for record in records
    )
    participant_ids = {
        record["participant_id"] for record in records
    }

    if set(class_counts) != set(LABELS):
        raise RuntimeError("数据集没有完整包含固定的 35 类")

    if any(count != 220 for count in class_counts.values()):
        raise RuntimeError(
            f"每类应有 220 个样本，实际为：{dict(class_counts)}"
        )

    if len(participant_ids) != 34:
        raise RuntimeError(
            f"应有 34 名参与者，实际找到 {len(participant_ids)} 名"
        )

    return records

class STEMNISTDataset(Dataset):
    """STEMNIST 原始连续压力 Dataset。"""

    def __init__(
        self,
        records,
        time_steps=240,
        transform=None,
    ):
        self.records = records
        self.time_steps = validate_time_steps(time_steps)
        self.transform = transform

        sample_count = len(records)
        # 保持 uint8：
        # 240 步约占 451 MiB，120 步约占 225 MiB
        self.pressures = torch.empty(
            (sample_count, self.time_steps, 1, 16, 16),
            dtype=torch.uint8,
        )

        self.labels = torch.empty(
            sample_count,
            dtype=torch.long,
        )

        print(f"开始预读取 {sample_count} 个样本……")

        for index, record in enumerate(records):
            with h5py.File(record["path"], "r") as file:
                raw_pressure = file["pressure_data"][...]
    
            pressure = aggregate_pressure_time(raw_pressure, self.time_steps)

            # pressure: [self.time_steps,16,16]
            # self.pressures[index]: [self.time_steps,1,16,16]
            self.pressures[index, :, 0].copy_(
                torch.from_numpy(pressure)
            )

            self.labels[index] = record["label_index"]

            if (index + 1) % 500 == 0:
                print(
                    f"已读取 {index + 1}/{sample_count}"
                )

        print(
            "预读取完成：",
            self.pressures.shape,
            self.pressures.dtype,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 返回指定索引的样本，包括压力数据和标签
        x = self.pressures[index]
        y = self.labels[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

def make_dataloaders(
    time_steps=240,
    batch_size=64,
    train_workers=0, 
    eval_workers=0,
    pin_memory=True,
):
    records = prepare_and_scan_samples()

    labels = np.array(
        [record["label_index"] for record in records]
    )
    all_indices = np.arange(len(records))

    # 第一次划分：80% train_val，20% test
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=42,
        stratify=labels,    # stratify 参数会根据你提供的类别标签进行分层，使划分后的子集尽量保持原始类别比例。
    )

    # 第二次划分：从 train_val 中取 20% 作为 val
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.20,
        random_state=42,
        stratify=labels[train_val_indices],
    )

    dataset = STEMNISTDataset(
        records,
        time_steps=time_steps,
    )

    # 数据划分完成后，仅使用训练集估计逐传感器压力基线。
    # 验证集和测试集不会参与任何统计量计算，因此不存在数据泄漏。
    pressure_baseline = estimate_training_pressure_baseline(
        pressures=dataset.pressures,
        train_indices=train_indices,
    )
    dataset.transform = NormalizePressure(
        baseline=pressure_baseline,
        adc_max=255.0,
    )

    print(
        "训练集压力基线："
        f"min={pressure_baseline.min().item():.1f}, "
        f"median={pressure_baseline.median().item():.1f}, "
        f"max={pressure_baseline.max().item():.1f}"
    )

    # 固定训练集打乱的随机数生成器
    generator = torch.Generator().manual_seed(42)

    train_common_arguments = {
        "batch_size": batch_size,
        "num_workers": train_workers,
        "pin_memory": pin_memory,
    }
    eval_common_arguments = {
        "batch_size": batch_size,
        "num_workers": eval_workers,
        "pin_memory": pin_memory,
    }

    if train_workers > 0:
        train_common_arguments["persistent_workers"] = True
        train_common_arguments["prefetch_factor"] = 4
    if eval_workers > 0:
        eval_common_arguments["persistent_workers"] = True
        eval_common_arguments["prefetch_factor"] = 4

    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        shuffle=True,
        generator=generator,
        **train_common_arguments,
    )

    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        shuffle=False,
        **eval_common_arguments,
    )

    test_loader = DataLoader(
        Subset(dataset, test_indices.tolist()),
        shuffle=False,
        **eval_common_arguments,
    )

    return train_loader, val_loader, test_loader


# 测试数据加载器是否正常工作，并打印一些信息
if __name__ == "__main__":
    train_loader, val_loader, test_loader = make_dataloaders(
        time_steps=120,
        batch_size=64,
    )

    x, y = next(iter(train_loader))

    print(
        "划分大小：",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )
    print("DataLoader x：", x.shape, x.dtype)
    print("DataLoader y：", y.shape, y.dtype)

    # DataLoader 输出是 batch 优先：
    # [N,T,C,H,W]
    #
    # 进入 SpikingJelly 多步网络前，只交换一次 N 和 T：
    x_snn = x.permute(1, 0, 2, 3, 4).contiguous()

    print("SNN 输入：", x_snn.shape)
