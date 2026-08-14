"""STEMNIST 原始压力数据的下载、校验、划分与数据集接口。"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


LABELS = tuple(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("123456789"))
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}

DATASET_URL = (
    "https://zenodo.org/records/19469535/files/"
    "STEMNIST%20Dataset.zip?download=1"
)
ARCHIVE_NAME = "STEMNIST Dataset.zip"
EXPECTED_MD5 = "6ca4638b2f95bf34f59873ab62399bd8"
EXPECTED_SAMPLE_COUNT = 7_700
EXPECTED_SAMPLES_PER_CLASS = 220
PRESSURE_SHAPE = (240, 16, 16)


class DatasetValidationError(RuntimeError):
    """表示数据内容或数据划分不符合预期。"""


def calculate_md5(file_path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """分块计算文件的 MD5。"""

    digest = hashlib.md5()
    with Path(file_path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path) -> None:
    """先下载到临时文件，完成后再原子替换目标文件。"""

    temporary_path = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, temporary_path)
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def find_raw_h5_files(extracted_root: str | Path) -> list[Path]:
    """递归查找原始压力文件，并排除已经编码的脉冲文件。"""

    root = Path(extracted_root)
    return sorted(
        path
        for path in root.rglob("*.h5")
        if not path.name.lower().endswith("_spikes.h5")
    )


def download_and_extract_stemnist(
    data_root: str | Path,
    download_if_missing: bool = True,
) -> tuple[Path, Path]:
    """准备压缩包和解压目录，并严格校验压缩包 MD5。"""

    data_root = Path(data_root).resolve()
    archive_path = data_root / ARCHIVE_NAME
    extracted_root = data_root / "extracted"

    if not archive_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"找不到数据压缩包：{archive_path}")
        print(f"正在下载 STEMNIST：{archive_path}")
        _download_file(DATASET_URL, archive_path)

    actual_md5 = calculate_md5(archive_path)
    if actual_md5 != EXPECTED_MD5:
        raise DatasetValidationError(
            "数据压缩包 MD5 校验失败："
            f"期望 {EXPECTED_MD5}，实际 {actual_md5}。"
        )

    raw_files = find_raw_h5_files(extracted_root) if extracted_root.exists() else []
    if len(raw_files) != EXPECTED_SAMPLE_COUNT:
        extracted_root.mkdir(parents=True, exist_ok=True)
        print(f"正在解压 STEMNIST：{extracted_root}")
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extracted_root)

    raw_files = find_raw_h5_files(extracted_root)
    if len(raw_files) != EXPECTED_SAMPLE_COUNT:
        raise DatasetValidationError(
            f"解压后应有 {EXPECTED_SAMPLE_COUNT} 个原始样本，"
            f"实际找到 {len(raw_files)} 个。"
        )
    return archive_path, extracted_root


def _decode_value(value: Any) -> Any:
    """将 HDF5 中的字节串或 NumPy 标量转换为普通 Python 值。"""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _find_pressure_dataset(handle: h5py.File) -> str:
    """寻找唯一的三维压力数据集。"""

    candidates: list[str] = []

    def visitor(name: str, item: h5py.Dataset | h5py.Group) -> None:
        if (
            isinstance(item, h5py.Dataset)
            and tuple(item.shape) == PRESSURE_SHAPE
            and np.issubdtype(item.dtype, np.integer)
        ):
            candidates.append(name)

    handle.visititems(visitor)
    if len(candidates) != 1:
        raise DatasetValidationError(
            f"HDF5 文件中应有一个形状为 {PRESSURE_SHAPE} 的整数数据集，"
            f"实际找到：{candidates}"
        )
    return candidates[0]


def _parse_filename(file_path: Path) -> tuple[str, str, str]:
    """从“参与者_字符_重复次数.h5”文件名中解析元数据。"""

    parts = file_path.stem.split("_")
    if len(parts) != 3:
        raise DatasetValidationError(f"无法解析样本文件名：{file_path.name}")
    participant_id, label, repetition = parts
    return participant_id, label.upper(), repetition


def inspect_h5_sample(
    file_path: str | Path,
    load_pressure: bool = False,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """读取一个 HDF5 样本的元数据，并可选返回压力数组。"""

    file_path = Path(file_path).resolve()
    filename_participant, filename_label, filename_repetition = _parse_filename(
        file_path
    )

    with h5py.File(file_path, "r") as handle:
        dataset_key = _find_pressure_dataset(handle)
        dataset = handle[dataset_key]
        values = dataset[...]
        pressure = values if load_pressure else None
        minimum = int(values.min())
        maximum = int(values.max())
        attributes = {key: _decode_value(value) for key, value in handle.attrs.items()}

        participant_id = str(
            attributes.get("experimenter", filename_participant)
        ).strip()
        label = str(attributes.get("character", filename_label)).strip().upper()
        repetition = str(
            attributes.get("experiment_number", filename_repetition)
        ).strip()
        sampling_rate = (
            int(handle["sampling_rate"][()])
            if "sampling_rate" in handle
            else int(attributes.get("sampling_rate", 120))
        )

        metadata = {
            "sample_id": file_path.stem,
            "participant_id": participant_id,
            "label": label,
            "label_index": LABEL_TO_INDEX.get(label, -1),
            "repetition": repetition,
            "filename_participant_id": filename_participant,
            "filename_label": filename_label,
            "filename_repetition": filename_repetition,
            "participant_metadata_matches_filename": (
                participant_id == filename_participant
            ),
            "label_metadata_matches_filename": label == filename_label,
            "repetition_metadata_matches_filename": (
                repetition == filename_repetition
            ),
            "raw_file_path": str(file_path),
            "dataset_key": dataset_key,
            "shape": str(list(dataset.shape)),
            "dtype": str(dataset.dtype),
            "minimum": minimum,
            "maximum": maximum,
            "sampling_rate": sampling_rate,
            "capture_time": str(attributes.get("capture_time", "")),
        }
    return metadata, pressure


def read_pressure_array(file_path: str | Path) -> np.ndarray:
    """读取一个样本的原始压力数组。"""

    _, pressure = inspect_h5_sample(file_path, load_pressure=True)
    if pressure is None:
        raise RuntimeError("压力数据读取失败。")
    return pressure


def validate_manifest(manifest: pd.DataFrame, strict: bool = True) -> None:
    """校验样本清单的唯一性、标签、形状和类型。"""

    required_columns = {
        "sample_id",
        "participant_id",
        "label",
        "label_index",
        "repetition",
        "filename_participant_id",
        "filename_label",
        "filename_repetition",
        "participant_metadata_matches_filename",
        "label_metadata_matches_filename",
        "repetition_metadata_matches_filename",
        "raw_file_path",
        "shape",
        "dtype",
        "minimum",
        "maximum",
    }
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise DatasetValidationError(f"样本清单缺少字段：{sorted(missing_columns)}")
    if manifest["sample_id"].duplicated().any():
        raise DatasetValidationError("样本清单中存在重复的 sample_id。")
    if set(manifest["label"]) != set(LABELS):
        raise DatasetValidationError("样本清单中的标签集合不等于固定的 35 类。")
    if not (manifest["label_index"] == manifest["label"].map(LABEL_TO_INDEX)).all():
        raise DatasetValidationError("样本清单中的标签索引不正确。")
    if not (manifest["shape"] == str(list(PRESSURE_SHAPE))).all():
        raise DatasetValidationError("存在形状不是 [240, 16, 16] 的样本。")
    if not (manifest["dtype"] == "uint8").all():
        raise DatasetValidationError("存在数据类型不是 uint8 的样本。")
    if manifest[["participant_id", "label", "repetition"]].isna().any().any():
        raise DatasetValidationError("样本清单中存在缺失元数据。")

    mismatch_columns = {
        "参与者": "participant_metadata_matches_filename",
        "标签": "label_metadata_matches_filename",
        "重复次数": "repetition_metadata_matches_filename",
    }
    for display_name, column_name in mismatch_columns.items():
        mismatch_count = int((~manifest[column_name].astype(bool)).sum())
        if mismatch_count:
            print(
                f"警告：发现 {mismatch_count} 个样本的 HDF5 {display_name}属性"
                "与文件名不一致；清单按 HDF5 属性记录。"
            )

    if strict:
        if len(manifest) != EXPECTED_SAMPLE_COUNT:
            raise DatasetValidationError(
                f"应有 {EXPECTED_SAMPLE_COUNT} 个样本，实际为 {len(manifest)}。"
            )
        class_counts = manifest["label"].value_counts()
        if not (class_counts == EXPECTED_SAMPLES_PER_CLASS).all():
            raise DatasetValidationError(
                "每个类别都应包含 "
                f"{EXPECTED_SAMPLES_PER_CLASS} 个样本，实际为："
                f"{class_counts.sort_index().to_dict()}"
            )


def build_manifest(
    extracted_root: str | Path,
    manifest_path: str | Path,
    strict: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """扫描原始 HDF5 文件，生成或读取固定样本清单。"""

    manifest_path = Path(manifest_path).resolve()
    if manifest_path.exists() and not overwrite:
        manifest = pd.read_csv(
            manifest_path,
            dtype={"participant_id": str, "label": str, "repetition": str},
        )
        for column_name in (
            "participant_metadata_matches_filename",
            "label_metadata_matches_filename",
            "repetition_metadata_matches_filename",
        ):
            if column_name in manifest:
                manifest[column_name] = (
                    manifest[column_name].astype(str).str.lower() == "true"
                )
        validate_manifest(manifest, strict=strict)
        return manifest

    raw_files = find_raw_h5_files(extracted_root)
    rows = []
    for index, file_path in enumerate(raw_files, start=1):
        metadata, _ = inspect_h5_sample(file_path)
        rows.append(metadata)
        if index % 1000 == 0:
            print(f"已检查 {index}/{len(raw_files)} 个原始样本。")

    manifest = (
        pd.DataFrame(rows)
        .sort_values(["label_index", "participant_id", "repetition", "sample_id"])
        .reset_index(drop=True)
    )
    validate_manifest(manifest, strict=strict)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest


def _validate_splits(
    manifest: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
) -> None:
    """验证三个划分互斥、完整且数量正确。"""

    expected_sizes = {"train": 4_928, "val": 1_232, "test": 1_540}
    split_ids = {
        name: set(frame["sample_id"].astype(str)) for name, frame in splits.items()
    }
    if set(split_ids) != set(expected_sizes):
        raise DatasetValidationError("数据划分必须包含 train、val 和 test。")
    for name, expected_size in expected_sizes.items():
        if len(split_ids[name]) != expected_size:
            raise DatasetValidationError(
                f"{name} 应有 {expected_size} 个样本，实际为 {len(split_ids[name])}。"
            )
    if split_ids["train"] & split_ids["val"]:
        raise DatasetValidationError("训练集与验证集存在重复样本。")
    if split_ids["train"] & split_ids["test"]:
        raise DatasetValidationError("训练集与测试集存在重复样本。")
    if split_ids["val"] & split_ids["test"]:
        raise DatasetValidationError("验证集与测试集存在重复样本。")
    if set.union(*split_ids.values()) != set(manifest["sample_id"].astype(str)):
        raise DatasetValidationError("数据划分没有完整覆盖样本清单。")


def create_stratified_splits(
    manifest: pd.DataFrame,
    split_path: str | Path,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """按类别分层生成 64%/16%/20% 的固定划分。"""

    split_path = Path(split_path).resolve()
    manifest_by_id = manifest.set_index("sample_id", drop=False)

    if split_path.exists() and not overwrite:
        split_ids = json.loads(split_path.read_text(encoding="utf-8"))
        splits = {
            name: manifest_by_id.loc[ids].reset_index(drop=True)
            for name, ids in split_ids.items()
        }
        _validate_splits(manifest, splits)
        return splits

    train_val, test = train_test_split(
        manifest,
        test_size=0.20,
        random_state=seed,
        stratify=manifest["label_index"],
    )
    train, val = train_test_split(
        train_val,
        test_size=0.20,
        random_state=seed,
        stratify=train_val["label_index"],
    )
    splits = {
        "train": train.sort_values("sample_id").reset_index(drop=True),
        "val": val.sort_values("sample_id").reset_index(drop=True),
        "test": test.sort_values("sample_id").reset_index(drop=True),
    }
    _validate_splits(manifest, splits)

    split_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        name: frame["sample_id"].astype(str).tolist()
        for name, frame in splits.items()
    }
    split_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return splits


def aggregate_pressure_time(pressure: np.ndarray, time_steps: int) -> np.ndarray:
    """将 240 帧压力序列保留为 240 步，或均值聚合为 20 步。"""

    if tuple(pressure.shape) != PRESSURE_SHAPE:
        raise ValueError(f"压力数组形状必须是 {PRESSURE_SHAPE}。")
    pressure = pressure.astype(np.float32, copy=False)
    if time_steps == 240:
        return pressure
    if time_steps == 20:
        return pressure.reshape(20, 12, 16, 16).mean(axis=1)
    raise ValueError("time_steps 只支持 20 或 240。")


class STEMNISTDataset(Dataset):
    """将样本清单转换为 PyTorch 可训练的原始压力数据集。"""

    def __init__(self, records: pd.DataFrame, time_steps: int) -> None:
        if time_steps not in (20, 240):
            raise ValueError("time_steps 只支持 20 或 240。")
        self.records = records.reset_index(drop=True).copy()
        self.time_steps = time_steps

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        record = self.records.iloc[index]
        pressure = read_pressure_array(record["raw_file_path"])
        pressure = aggregate_pressure_time(pressure, self.time_steps)
        sample = torch.from_numpy(pressure).unsqueeze(1)
        label_index = int(record["label_index"])
        return sample, label_index, str(record["sample_id"])
