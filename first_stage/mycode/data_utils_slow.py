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
EXPECTED_SHAPE = (240, 16, 16)

# 标签顺序一旦确定，之后不能改变
LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789")
LABEL_TO_INDEX = {
    label: index
    for index, label in enumerate(LABELS)
}


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
    ):
        if time_steps != 240:
            raise ValueError(
                "time_steps 目前只支持 240"
            )

        self.records = records
        self.time_steps = time_steps

        sample_count = len(records)
        # 保持 uint8：
        # 总内存占用约 451 MiB
        self.pressures = torch.empty(
            (sample_count, 240, 1, 16, 16),
            dtype=torch.uint8,
        )

        self.labels = torch.empty(
            sample_count,
            dtype=torch.long,
        )

        print(f"开始预读取 {sample_count} 个样本……")

        for index, record in enumerate(records):
            with h5py.File(record["path"], "r") as file:
                pressure = file["pressure_data"][...]

            # pressure: [240,16,16]
            # self.pressures[index]: [240,1,16,16]
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
        time_steps=240,
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