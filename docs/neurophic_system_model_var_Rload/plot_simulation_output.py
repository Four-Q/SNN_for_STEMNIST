from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "System_with_TIA.raw"
OUTPUT_PATH = ROOT / "figures" / "simulation_output_AT_A_1_r07_c09.png"


def read_ltspice_transient_raw(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read an uncompressed real LTspice transient RAW file."""

    raw = path.read_bytes()
    marker = "Binary:\n".encode("utf-16le")
    marker_index = raw.find(marker)

    if marker_index < 0:
        raise RuntimeError(f"Cannot find Binary marker in {path}")

    data_offset = marker_index + len(marker)
    header = raw[:data_offset].decode("utf-16le")

    point_match = re.search(r"No\. Points:\s+(\d+)", header)
    variable_match = re.search(r"No\. Variables:\s+(\d+)", header)

    if point_match is None or variable_match is None:
        raise RuntimeError("Incomplete LTspice RAW header")

    point_count = int(point_match.group(1))
    variable_count = int(variable_match.group(1))
    variable_names = [
        match.group(1)
        for match in re.finditer(r"^\s*\d+\s+([^\t]+)\t", header, re.MULTILINE)
    ]

    if len(variable_names) != variable_count:
        raise RuntimeError(
            f"Expected {variable_count} variables, found {len(variable_names)}"
        )

    record_dtype = np.dtype(
        [
            ("time", "<f8"),
            ("values", "<f4", (variable_count - 1,)),
        ]
    )
    records = np.frombuffer(
        raw,
        dtype=record_dtype,
        count=point_count,
        offset=data_offset,
    )

    time = records["time"].copy()
    signals = {
        name: records["values"][:, index].astype(np.float64, copy=True)
        for index, name in enumerate(variable_names[1:])
    }
    return time, signals


def count_rising_crossings(signal: np.ndarray, threshold: float = 1.4) -> int:
    return int(
        np.count_nonzero(
            (signal[:-1] < threshold) & (signal[1:] >= threshold)
        )
    )


def resistance_from_pressure(pressure: np.ndarray) -> np.ndarray:
    pressure_over_baseline = pressure - 75.0
    resistance = np.full_like(pressure, 1e12, dtype=np.float64)
    active = pressure_over_baseline > 0
    resistance[active] = np.clip(
        900e3 / pressure_over_baseline[active],
        5e3,
        1e12,
    )
    return resistance


def main() -> None:
    time, signals = read_ltspice_transient_raw(RAW_PATH)

    # In this schematic n001 is the fourth U1 pin driven by V_PCTRL.
    pressure = signals["V(n001)"]
    rload = resistance_from_pressure(pressure)
    vout = signals["V(vout)"]
    vtia = signals["V(vtia)"]
    vdrive = signals["V(vdrive)"]
    final_out = signals["V(final_out)"]

    # Ignore the all-zero initialization point when reporting the pressure range.
    pressure_valid = pressure[time > 1e-9]
    valid_time = time > 1e-9
    rload_valid = rload[valid_time]
    u1_spikes = count_rising_crossings(vout)
    u2_spikes = count_rising_crossings(final_out)

    peak_index = int(np.argmax(pressure))
    peak_time = float(time[peak_index])
    zoom_start = max(0.0, peak_time - 0.04)
    zoom_end = min(2.0, peak_time + 0.04)
    zoom = (time >= zoom_start) & (time <= zoom_end)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "SimHei"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "#f7f8fa",
            "axes.facecolor": "#ffffff",
        }
    )

    colors = {
        "pressure": "#cb5b35",
        "resistance": "#6d4aa2",
        "u1": "#147d92",
        "tia": "#4372b8",
        "drive": "#d18a20",
        "u2": "#237a57",
        "threshold": "#777777",
    }

    figure, axes = plt.subplots(
        6,
        1,
        figsize=(12, 13.5),
        gridspec_kw={"height_ratios": [1.0, 1.05, 1.1, 1.0, 1.1, 1.1]},
    )
    figure.subplots_adjust(hspace=0.47, top=0.925, bottom=0.065, left=0.095, right=0.97)

    figure.suptitle(
        "STEMNIST 压力驱动神经形态系统：LTspice 2 秒瞬态仿真",
        fontsize=18,
        y=0.975,
    )
    figure.text(
        0.5,
        0.946,
        (
            "样本 AT_A_1 · taxel (7, 9) · Vin=2.5 V · "
            f"压力 {pressure_valid.min():.0f}–{pressure_valid.max():.0f} · "
            f"U1 上升沿 {u1_spikes} 次 · U2 上升沿 {u2_spikes} 次"
        ),
        ha="center",
        fontsize=10.5,
        color="#4f5965",
    )

    ax = axes[0]
    ax.plot(time, pressure, color=colors["pressure"], linewidth=1.25)
    ax.axhline(75, color=colors["threshold"], linestyle="--", linewidth=1, label="无压力阈值 75")
    ax.axhline(95.52, color="#b33f62", linestyle=":", linewidth=1.2, label="理论起振压力 ≈95.5")
    ax.set_xlim(0, 2)
    ax.set_ylim(65, max(190, pressure_valid.max() + 10))
    ax.set_ylabel("压力 ADC")
    ax.set_title("① 原始压力帧（120 Hz，零阶保持）", loc="left", fontsize=11)
    ax.legend(loc="upper right", ncol=2, fontsize=8.5, frameon=False)

    ax = axes[1]
    ax.semilogy(time[valid_time], rload_valid / 1e3, color=colors["resistance"], linewidth=1.25)
    ax.axhline(43.8615, color="#b33f62", linestyle=":", linewidth=1.2, label="2.5 V 理论起振边界 43.86 kΩ")
    ax.set_xlim(0, 2)
    ax.set_ylim(4, max(250, 1.25 * rload_valid.max() / 1e3))
    ax.set_ylabel("Rload (kΩ，对数轴)")
    ax.set_title("② 压力映射得到的可变负载电阻", loc="left", fontsize=11)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

    ax = axes[2]
    ax.plot(time, vout, color=colors["u1"], linewidth=0.85)
    ax.axhline(1.4, color=colors["threshold"], linestyle="--", linewidth=0.9, alpha=0.8)
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.05, 1.85)
    ax.set_ylabel("VOUT (V)")
    ax.set_title("③ 前振荡神经元 U1 输出", loc="left", fontsize=11)

    ax = axes[3]
    ax.plot(time[zoom], vout[zoom], color=colors["u1"], linewidth=1.15, label="U1 VOUT")
    ax.step(time[zoom], pressure[zoom] / 120.0, where="post", color=colors["pressure"], alpha=0.55, linewidth=1.0, label="压力/120（仅供对照）")
    ax.axhline(1.4, color=colors["threshold"], linestyle="--", linewidth=0.9)
    ax.set_xlim(zoom_start, zoom_end)
    ax.set_ylim(-0.05, 1.85)
    ax.set_ylabel("电压 (V)")
    ax.set_title(f"④ U1 振荡细节（峰值压力附近 {zoom_start:.3f}–{zoom_end:.3f} s）", loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)

    ax = axes[4]
    ax.plot(time, vtia, color=colors["tia"], linewidth=0.95, label="VTIA")
    ax.plot(time, vdrive, color=colors["drive"], linewidth=0.95, label="VDRIVE")
    ax.set_xlim(0, 2)
    ax.set_ylabel("电压 (V)")
    ax.set_title("⑤ 突触电流经 TIA 与电平移位后的驱动", loc="left", fontsize=11)
    ax.legend(loc="upper right", ncol=2, fontsize=8.5, frameon=False)

    ax = axes[5]
    ax.plot(time, final_out, color=colors["u2"], linewidth=0.85)
    ax.axhline(1.4, color=colors["threshold"], linestyle="--", linewidth=0.9, alpha=0.8)
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.05, 1.85)
    ax.set_ylabel("final_out (V)")
    ax.set_xlabel("时间 (s)")
    ax.set_title("⑥ 后振荡神经元 U2 最终输出", loc="left", fontsize=11)

    for ax in axes:
        ax.grid(True, which="major", color="#dfe3e8", linewidth=0.6, alpha=0.85)
        ax.tick_params(labelsize=9)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=220, facecolor=figure.get_facecolor())
    plt.close(figure)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Pressure range: {pressure_valid.min():.0f}..{pressure_valid.max():.0f}")
    print(
        f"Rload range: {rload_valid.min() / 1e3:.3f}.."
        f"{rload_valid.max() / 1e3:.3f} kOhm"
    )
    print(f"U1 rising crossings: {u1_spikes}")
    print(f"U2 rising crossings: {u2_spikes}")
    print(f"VTIA range: {vtia.min():.6f}..{vtia.max():.6f} V")
    print(f"VDRIVE range: {vdrive.min():.6f}..{vdrive.max():.6f} V")


if __name__ == "__main__":
    main()
