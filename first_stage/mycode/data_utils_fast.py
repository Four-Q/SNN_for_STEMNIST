# 数据处理Pipeline，用于提供DataLoader
from collections import Counter
from hashlib import md5 # 用于检查原始ZIP文件是否完好
from pathlib import Path
from zipfile import ZipFile
import json

import h5py # 用于读取.h5文件
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

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
EXPECTED_SHAPE = (240, 16, 16)

# 标签顺序一旦确定，之后不能改变
LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789")
LABEL_TO_INDEX = {
    label: index
    for index, label in enumerate(LABELS)
}
EXPECTED_SAMPLE_COUNT = 7700
EXPECTED_SAMPLES_PER_CLASS = 220
# 新增缓存文件。
#
# tensor_cache:
#   pressure: [7700,240,1,16,16]，uint8
#   labels:   [7700]，int64
#
# manifest:
#   保存缓存中每个位置对应的 sample_id
#
# split:
#   保存训练、验证、测试集的固定 sample_id
CACHE_DIR = DATA_ROOT / "cache"
TENSOR_CACHE_PATH = CACHE_DIR / "stemnist_240_u8.pt"
MANIFEST_PATH = CACHE_DIR / "manifest_cache.json"
SPLIT_PATH = CACHE_DIR / "split_seed42.json"

# ============================================================
# 原始数据检查
# ============================================================
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

    if len(paths) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"应有 {EXPECTED_SAMPLE_COUNT} 个原始样本，实际找到 {len(paths)} 个"
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

    if any(count != EXPECTED_SAMPLES_PER_CLASS for count in class_counts.values()):
        raise RuntimeError(
            f"每类应有 {EXPECTED_SAMPLES_PER_CLASS} 个样本，实际为：{dict(class_counts)}"
        )

    if len(participant_ids) != 34:
        raise RuntimeError(
            f"应有 34 名参与者，实际找到 {len(participant_ids)} 名"
        )

    return records

# ============================================================
# 一次性连续缓存
# ============================================================
def build_cache_if_needed(force_rebuild=False):
    """
    首次运行时，把 7700 个 HDF5 打包成单个连续 Tensor。

    后续运行如果缓存已存在，会直接跳过。
    """

    cache_is_complete = (
        TENSOR_CACHE_PATH.is_file()
        and MANIFEST_PATH.is_file()
    )

    if cache_is_complete and not force_rebuild:
        print(f"使用已有数据缓存：{TENSOR_CACHE_PATH}")
        return

    print("没有找到完整缓存，开始读取原始 HDF5。")

    records = prepare_and_scan_samples()

    # [修改 3]
    # 原始版本每个 epoch 都从 HDF5 读取数据。
    # 新版本只读取一次，保存为一个连续 uint8 Tensor。
    #
    # uint8 缓存约 451 MiB。
    pressure = torch.empty(
        (
            EXPECTED_SAMPLE_COUNT,
            240,
            1,
            16,
            16,
        ),
        dtype=torch.uint8,
    )

    labels = torch.empty(
        EXPECTED_SAMPLE_COUNT,
        dtype=torch.long,
    )

    manifest = []

    for index, record in enumerate(records):
        with h5py.File(record["path"], "r") as file:
            raw_pressure = file["pressure_data"][...]

        # raw_pressure: [T,H,W]
        # pressure[index]: [T,C,H,W]
        pressure[index, :, 0].copy_(
            torch.from_numpy(raw_pressure)
        )

        labels[index] = record["label_index"]

        # manifest 不保存绝对路径。
        # 远程平台路径变化时，缓存仍然可以使用。
        manifest.append(
            {
                "sample_id": record["sample_id"],
                "participant_id": record["participant_id"],
                "label": record["label"],
                "label_index": record["label_index"],
                "repetition": record["repetition"],
            }
        )

        if (index + 1) % 500 == 0:
            print(
                f"已打包 {index + 1}/"
                f"{EXPECTED_SAMPLE_COUNT} 个样本"
            )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 先写临时文件，成功后再替换正式缓存
    temporary_tensor_path = (
        TENSOR_CACHE_PATH.with_name(
            TENSOR_CACHE_PATH.name + ".tmp"
        )
    )
    temporary_manifest_path = (
        MANIFEST_PATH.with_name(
            MANIFEST_PATH.name + ".tmp"
        )
    )

    torch.save(
        {
            "pressure": pressure,
            "labels": labels,
        },
        temporary_tensor_path,
    )

    temporary_manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_tensor_path.replace(
        TENSOR_CACHE_PATH
    )
    temporary_manifest_path.replace(
        MANIFEST_PATH
    )

    print("连续数据缓存生成完成：")
    print(TENSOR_CACHE_PATH)

def load_cached_data(force_rebuild=False):
    """读取连续 Tensor 缓存和样本清单。"""

    build_cache_if_needed(
        force_rebuild=force_rebuild
    )

    # [修改 4]
    # 不再在训练阶段读取 HDF5。
    # 训练开始时只读取一次约 451 MiB 的缓存文件。
    cache = torch.load(
        TENSOR_CACHE_PATH,
        map_location="cpu",
        weights_only=True,
    )

    pressure = cache["pressure"]
    labels = cache["labels"]

    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    expected_cache_shape = (
        EXPECTED_SAMPLE_COUNT,
        240,
        1,
        16,
        16,
    )

    if tuple(pressure.shape) != expected_cache_shape:
        raise RuntimeError(
            "缓存压力张量形状错误："
            f"{tuple(pressure.shape)}"
        )

    if pressure.dtype != torch.uint8:
        raise RuntimeError(
            "缓存压力类型应为 torch.uint8，"
            f"实际为 {pressure.dtype}"
        )

    if tuple(labels.shape) != (
        EXPECTED_SAMPLE_COUNT,
    ):
        raise RuntimeError(
            f"缓存标签形状错误：{labels.shape}"
        )

    if labels.dtype != torch.long:
        raise RuntimeError(
            "缓存标签类型应为 torch.int64，"
            f"实际为 {labels.dtype}"
        )

    if len(manifest) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            "manifest 样本数量错误："
            f"{len(manifest)}"
        )

    return pressure, labels, manifest

# ============================================================
# 固定数据划分
# ============================================================

def create_or_load_splits(manifest, labels):
    """
    创建或读取固定的数据划分。

    JSON 中保存 sample_id，而不是只保存数字索引，
    防止以后缓存排列发生变化。
    """

    sample_ids = [
        record["sample_id"]
        for record in manifest
    ]

    if SPLIT_PATH.is_file():
        print(f"读取已有数据划分：{SPLIT_PATH}")

        split_sample_ids = json.loads(
            SPLIT_PATH.read_text(
                encoding="utf-8"
            )
        )

    else:
        print("没有找到固定划分，使用 seed=42 创建。")

        all_indices = np.arange(
            EXPECTED_SAMPLE_COUNT
        )

        label_array = labels.cpu().numpy()

        train_val_indices, test_indices = (
            train_test_split(
                all_indices,
                test_size=0.20,
                random_state=42,
                stratify=label_array,
            )
        )

        train_indices, val_indices = (
            train_test_split(
                train_val_indices,
                test_size=0.20,
                random_state=42,
                stratify=label_array[
                    train_val_indices
                ],
            )
        )

        split_sample_ids = {
            "train": [
                sample_ids[index]
                for index in train_indices
            ],
            "val": [
                sample_ids[index]
                for index in val_indices
            ],
            "test": [
                sample_ids[index]
                for index in test_indices
            ],
        }

        CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        SPLIT_PATH.write_text(
            json.dumps(
                split_sample_ids,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"固定划分已保存：{SPLIT_PATH}")

    # 把保存的 sample_id 转换回当前缓存中的索引
    sample_id_to_index = {
        sample_id: index
        for index, sample_id in enumerate(sample_ids)
    }

    expected_sizes = {
        "train": 4928,
        "val": 1232,
        "test": 1540,
    }

    split_indices = {}

    if set(split_sample_ids) != set(expected_sizes):
        raise RuntimeError(
            "划分文件必须包含 train、val、test"
        )

    for split_name, expected_size in expected_sizes.items():
        ids = split_sample_ids[split_name]

        if len(ids) != expected_size:
            raise RuntimeError(
                f"{split_name} 应有 {expected_size} 个样本，"
                f"实际为 {len(ids)}"
            )

        try:
            indices = [
                sample_id_to_index[sample_id]
                for sample_id in ids
            ]
        except KeyError as error:
            raise RuntimeError(
                f"划分中出现未知 sample_id：{error}"
            ) from error

        split_indices[split_name] = torch.tensor(
            indices,
            dtype=torch.long,
        )

    # 验证三个集合互斥，并且覆盖全部数据
    all_split_indices = torch.cat(
        [
            split_indices["train"],
            split_indices["val"],
            split_indices["test"],
        ]
    )

    unique_indices = torch.unique(
        all_split_indices
    )

    if len(all_split_indices) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            "数据划分没有包含全部 7700 个样本"
        )

    if len(unique_indices) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            "训练、验证或测试集之间存在重复样本"
        )

    return split_indices

# ============================================================
# 内存缓存 Dataset
# ============================================================

class STEMNISTDataset(Dataset):
    """
    从连续内存 Tensor 中读取 STEMNIST。

    与原版本的关键区别：
    __getitem__ 不再打开任何 HDF5 文件。
    """

    def __init__(
        self,
        pressure,
        labels,
        indices,
        sample_ids,
    ):
        self.pressure = pressure
        self.labels = labels
        self.indices = indices
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        global_index = int(
            self.indices[index]
        )

        # [修改 5]
        # 原始版本：
        #
        # with h5py.File(record["path"], "r") as file:
        #     pressure = file["pressure_data"][...]
        #     pressure = pressure.astype(np.float32)
        #
        # 修改后直接索引内存中的连续 uint8 Tensor。
        x = self.pressure[global_index]

        y = self.labels[global_index]
        return x, y, global_index


# ============================================================
# DataLoader
# ============================================================

def _make_loader_arguments(
    batch_size,
    num_workers,
    pin_memory,
    persistent_workers,
):
    """根据 worker 数量安全创建 DataLoader 参数。"""

    arguments = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    # prefetch_factor 和 persistent_workers
    # 只有 num_workers > 0 时才能使用
    if num_workers > 0:
        arguments.update(
            {
                "prefetch_factor": 4,
                "persistent_workers": persistent_workers,
            }
        )

    return arguments

def make_dataloaders(
    time_steps=240,
    batch_size=64,
    train_workers=4,
    eval_workers=2,
    pin_memory=True,
    force_rebuild=False,
):
    if time_steps != 240:
        raise ValueError(
            "当前缓存版本只支持 time_steps=240"
        )

    pressure, labels, manifest = (
        load_cached_data(
            force_rebuild=force_rebuild
        )
    )

    split_indices = create_or_load_splits(
        manifest,
        labels,
    )

    sample_ids = [
        record["sample_id"]
        for record in manifest
    ]

    train_dataset = STEMNISTDataset(
        pressure=pressure,
        labels=labels,
        indices=split_indices["train"],
        sample_ids=sample_ids,
    )

    val_dataset = STEMNISTDataset(
        pressure=pressure,
        labels=labels,
        indices=split_indices["val"],
        sample_ids=sample_ids,
    )

    test_dataset = STEMNISTDataset(
        pressure=pressure,
        labels=labels,
        indices=split_indices["test"],
        sample_ids=sample_ids,
    )

    generator = torch.Generator().manual_seed(42)


    # [修改 7]
    # 原来：
    #   num_workers = 0
    #
    # 修改后：
    #   训练集默认 4 个 worker
    #   验证和测试默认 2 个 worker
    #
    # 数据已经在内存里，不需要直接开到 25 个 worker。
    train_arguments = _make_loader_arguments(
        batch_size=batch_size,
        num_workers=train_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    val_arguments = _make_loader_arguments(
        batch_size=batch_size,
        num_workers=eval_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    test_arguments = _make_loader_arguments(
        batch_size=batch_size,
        num_workers=eval_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **train_arguments,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **val_arguments,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **test_arguments,
    )

    return train_loader, val_loader, test_loader


# 测试数据加载器是否正常工作，并打印一些信息
if __name__ == "__main__":
    train_loader, val_loader, test_loader = make_dataloaders(
        time_steps=240,
        batch_size=64,
    )

    x, y, sample_ids = next(iter(train_loader))

    print(
        "划分大小：",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )
    print("DataLoader x：", x.shape, x.dtype)
    print("DataLoader y：", y.shape, y.dtype)
    print("样本 ID 示例：", sample_ids[:3])

    # DataLoader 输出是 batch 优先：
    # [N,T,C,H,W]
    #
    # 进入 SpikingJelly 多步网络前，只交换一次 N 和 T：
    x_snn = x.permute(1, 0, 2, 3, 4).contiguous()

    print("SNN 输入：", x_snn.shape)