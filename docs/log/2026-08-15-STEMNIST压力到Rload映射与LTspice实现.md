# 2026-08-15 STEMNIST 压力到 Rload 映射与 LTspice 实现工作日志

## 1. 本次工作目标

本次工作围绕以下问题展开：

1. 如何把 STEMNIST 原始压力帧中的 ADC 数值映射为神经形态系统前振荡神经元的 `Rload`；
2. 在输入电压固定为 2.5 V 时，确定 NbOx 振荡器允许工作的电阻区间；
3. 如何在 LTspice 的 2 秒瞬态仿真中，使 `Rload` 按 240 帧压力序列随时间变化；
4. 如何处理无压力时压敏电阻接近开路、前神经元静默以及突触输入残留的问题；
5. 明确单通道 LTspice 原理图与 STEMNIST 16×16、共 256 个 taxel 之间的关系。

本次工作先完成方案分析和实现设计，随后已在独立目录 `docs/neurophic_system_model_var_Rload` 中完成可变 Rload 版本，保留了原来的 `docs/neurophic_system_model` 基线目录。已创建可变振荡器模型、四引脚符号、压力 PWL 生成 Notebook、一个真实 taxel 的两秒 PWL 文件和修改后的系统原理图，并已成功运行一次 LTspice 瞬态仿真。

### 1.1 实施状态更新

当前实现目录为：

```text
docs/neurophic_system_model_var_Rload/
```

已落地的主要文件包括：

- `NbOx_OSC_VAR.lib`：带 `PCTRL` 引脚和压力控制行为电阻的 U1 模型；
- `NbOx_OSC_VAR.asy`：`VIN/OUTPUT/GND/PCTRL` 四引脚符号；
- `System_with_TIA.asc`：U1 可变 Rload、U2 固定 10 kΩ 的系统原理图；
- `pressure_pwl_generation.ipynb`：从 HDF5 原始压力生成零阶保持 PWL；
- `PCTRL_AT_A_1_r07_c09.pwl`：样本 `AT_A_1`、taxel `(7,9)` 的两秒压力序列；
- `System_with_TIA.log/.raw/.plt`：一次成功瞬态运行的输出。

当前实现没有加入 BGATE，U1 输出仍直接驱动突触，因此保留压力撤除后的真实 RC 放电尾迹。

## 2. 检查的主要文件

- `first_stage/mycode/data_utils_slow.py`
- `first_stage/STEMNIST/cache/stemnist_240_u8.pt`
- `docs/STEMNIST_raw_paper.md`
- `docs/neurophic_system_model/System_with_TIA.asc`
- `docs/neurophic_system_model/NbOx_OSC_stable.lib`
- `docs/neurophic_system_model/NbOx_OSC.asy`
- `docs/neurophic_system_model/synapse_advanced_v2.sub`
- `docs/neurophic_system_model/System_with_TIA.log`
- `docs/neurophic_system_model/6-0114-2.3V-1uF-10kohm-2-imtegrate-Vd1V.pwl`

## 3. STEMNIST 数据与时间轴结论

### 3.1 原始数据形式

`data_utils_slow.py` 中定义：

```python
RAW_TIME_STEPS = 240
EXPECTED_SHAPE = (RAW_TIME_STEPS, 16, 16)
```

每个字符样本包含：

- 240 帧；
- 每帧 16×16，共 256 个 taxel；
- 采样率 120 Hz；
- 总时长 2 秒；
- 原始数据类型为 `uint8`。

因此每帧对应的物理时间是：

\[
\Delta t=\frac{2}{240}=\frac{1}{120}\ \mathrm{s}
=8.333333\ \mathrm{ms}
\]

如果要保留原始压力时序，LTspice 输入必须使用 240 个时间片。`time_steps=120` 会将相邻两帧平均，每步时间变为 16.667 ms，不再是原始帧级输入。

### 3.2 压力值实际分布

论文材料中的严谨表述是：ADC 数值**主要位于 75–255**，约 75 表示无压力或接近基线，接近 255 表示较强接触压力。

对缓存 `stemnist_240_u8.pt` 中全部 473,088,000 个压力值进行统计后得到：

- 缓存中的绝对最小值为 64；
- 绝对最大值为 255；
- 众数和中位数均为 81；
- 小于 75 的值一共只有 17 个，占比约为 `3.59e-8`；
- 其中 64 有 1 个，72 有 2 个，73 有 3 个，74 有 11 个；
- 因而“64–255”是绝对观测范围，“75–255”是主要有效范围。

这些小于 75 的极少数值可以统一归入无压力/开路区，不影响映射设计。

## 4. 当前电路结构结论

`System_with_TIA.asc` 当前包含：

1. 前振荡神经元 U1；
2. 突触模型 `SYNAPSE_ADV_V2`；
3. TIA；
4. 电平移位运放；
5. 后振荡神经元 U2。

当前两个振荡器均使用 `NbOx_OSC_stable.lib`，实例参数中都包含：

```spice
VH=1.676 VL=1.127 Rin=89213.44 Rme=806 Rload=10k Cparal=1u
```

模型内的固定负载电阻为：

```spice
RLOAD_IN VIN OUTPUT {Rload}
```

为了只让前振荡神经元 U1 随压力改变，不能直接把原模型全局改成可变 Rload，否则 U2 也会受到影响。现已复制并实现：

```text
NbOx_OSC_VAR.lib
NbOx_OSC_VAR.asy
```

修改后的原理图中，U1 已使用 `NbOx_OSC_VAR`，U2 保持原来的 `NbOx_OSC` 和固定 `Rload=10k`。

当前原理图只有一条前神经元—突触—后神经元通道。STEMNIST 一帧有 256 个 taxel，因此：

- 单通道 LTspice 验证时，每次只能选择一个 `(row, column)` 的压力序列；
- 保留完整 16×16 空间信息时，需要 256 个独立前级编码通道，或先用单通道仿真建立代理/LUT 后在 Python 中批量编码；
- 把整帧平均成一个 Rload 会丢失空间结构，不能直接替代 `[T,1,16,16]` 的 SNN 输入。

## 5. 2.5 V 输入下的振荡电阻范围

当前模型参数为：

\[
V_H=1.676\ \mathrm{V},\quad
V_L=1.127\ \mathrm{V}
\]

\[
R_{in}=89213.44\ \Omega,\quad
R_{me}=806\ \Omega,\quad
C=1\ \mu\mathrm{F}
\]

输入固定为：

\[
V_{in}=2.5\ \mathrm{V}
\]

高阻状态的输出终值为：

\[
V_{\infty,H}=V_{in}\frac{R_{in}}{R_{load}+R_{in}}
\]

为了能够越过上阈值，需要：

\[
R_{load}<R_{in}\left(\frac{V_{in}}{V_H}-1\right)
=43.86\ \mathrm{k}\Omega
\]

低阻状态的输出终值为：

\[
V_{\infty,L}=V_{in}\frac{R_{me}}{R_{load}+R_{me}}
\]

为了能够跌破下阈值，需要：

\[
R_{load}>R_{me}\left(\frac{V_{in}}{V_L}-1\right)
=0.982\ \mathrm{k}\Omega
\]

所以 2.5 V 下的理论振荡区间约为：

\[
0.982\ \mathrm{k}\Omega<R_{load}<43.86\ \mathrm{k}\Omega
\]

在 5–40 kΩ 区间内，Rload 越小，理论振荡频率总体越高。部分稳态估算值为：

| Rload | 理论频率 |
|---:|---:|
| 40 kΩ | 约 14.5 Hz |
| 30 kΩ | 约 32.9 Hz |
| 20 kΩ | 约 65.4 Hz |
| 15 kΩ | 约 96.5 Hz |
| 10 kΩ | 约 156 Hz |
| 5 kΩ | 约 319 Hz |

上述频率是理想稳态估算；240 帧动态切换时，Rload 可能在一个振荡周期中途变化，最终结果应以 LTspice 瞬态仿真为准。

## 6. 映射方案的讨论与最终选择

### 6.1 曾讨论的有限区间线性电阻映射

曾讨论将压力直接线性映射为：

\[
p=75\rightarrow40\ \mathrm{k}\Omega
\]

\[
p=255\rightarrow5\ \mathrm{k}\Omega
\]

对应公式为：

\[
R_{load}(p)=
40\ \mathrm{k}\Omega-
\frac{p-75}{180}\times35\ \mathrm{k}\Omega
\]

并对区间外数值进行截断。

该方案能表达“压力越大、电阻越小”，且 5–40 kΩ 均处于振荡工作区。但它会使无压力时的 40 kΩ 仍产生约 14.5 Hz 的背景发放，不满足“无压力时压敏电阻接近开路、神经元静默”的最新物理假设。

### 6.2 当前选择：压力与电导近似线性

有限斜率的线性电阻函数无法同时满足：

\[
R(75)=\infty
\]

和：

\[
R(255)=5\ \mathrm{k}\Omega
\]

因此当前选择让**电导**随有效压力增量近似线性：

\[
x(p)=\operatorname{clip}\left(\frac{p-75}{255-75},0,1\right)
\]

\[
G(p)=\frac{x(p)}{R_{min}},\qquad R_{min}=5\ \mathrm{k}\Omega
\]

由 `R=1/G` 得到**最终映射：**

\[
R_{load}(p)=
\begin{cases}
\infty, & p\le75\\[4pt]
\dfrac{900\ \mathrm{k}\Omega}{p-75}, & 75<p<255\\[8pt]
5\ \mathrm{k}\Omega, & p\ge255
\end{cases}
\]

LTspice 中使用 1 TΩ 近似无穷大。

典型映射值为：

| 压力 ADC | Rload | 2.5 V 下状态 |
|---:|---:|---|
| ≤75 | 1 TΩ | 静默/近似开路 |
| 76 | 900 kΩ | 静默 |
| 81 | 150 kΩ | 静默 |
| 90 | 60 kΩ | 静默 |
| 95 | 45 kΩ | 接近振荡边界，仍不满足稳态振荡条件 |
| 96 | 42.86 kΩ | 开始进入振荡区 |
| 100 | 36 kΩ | 振荡 |
| 120 | 20 kΩ | 振荡 |
| 150 | 12 kΩ | 振荡 |
| 180 | 8.57 kΩ | 振荡 |
| 208 | 6.77 kΩ | 振荡 |
| 255 | 5 kΩ | 高发放率 |

由：

\[
\frac{900\ \mathrm{k}\Omega}{p-75}<43.86\ \mathrm{k}\Omega
\]

可得当前模型大约在：

\[
p>95.52
\]

时进入稳态振荡区。因此整数 ADC 上大致表现为：

- `p<=95`：无持续振荡；
- `p>=96`：开始振荡；
- 压力继续增大时，Rload 继续减小，发放频率升高。

这一静默区由压敏电阻开路特性和振荡器工作条件自然共同产生，而不是额外人为设置固定频率阈值。

## 7. LTspice 可变 Rload 的已实现方案

LTspice 普通 `.param` 在瞬态仿真开始时求值，不适合直接把子电路参数写成 `Rload(time)`。当前实现使用 B 行为电阻，并通过第 4 个控制引脚 `PCTRL` 读取压力 PWL。

### 7.1 新子电路接口

```spice
.SUBCKT NbOx_OSC_VAR VIN OUTPUT GND PCTRL PARAMS:
+ VH=1.676 VL=1.127
+ Rin=89213.44 Rme=806
+ Cparal=1u
+ A=1000000 Rstate=1 Cstate=100n
```

### 7.2 行为负载电阻

原固定电阻：

```spice
RLOAD_IN VIN OUTPUT {Rload}
```

实际已替换为：

```spice
BLOAD VIN OUTPUT R=if(V(PCTRL,GND)<=75,1T,limit(900k/max(V(PCTRL,GND)-75,1u),5k,1T))
```

其含义是：

- `PCTRL<=75`：`Rload=1TΩ`，近似开路；
- `PCTRL>75`：`Rload=900k/(PCTRL-75)`；
- `max(...,1u)` 防止除零；
- `limit(...,5k,1T)` 保证电阻始终为正且位于允许范围；
- `PCTRL` 只作为行为表达式的数值载体，不向主信号链供电。

### 7.3 新符号要求

已复制 `NbOx_OSC.asy` 为 `NbOx_OSC_VAR.asy`，实际内容满足：

- `Prefix=X` 保持不变；
- `Value` 改为 `NbOx_OSC_VAR`；
- 增加第 4 个引脚 `PCTRL`；
- `PCTRL` 的 `Netlist Order` 必须为 4；
- 原有引脚顺序继续为 `VIN=1`、`OUTPUT=2`、`GND=3`。

### 7.4 已完成的原理图连接

`docs/neurophic_system_model_var_Rload/System_with_TIA.asc` 中已完成：

1. 保留 `.include "NbOx_OSC_stable.lib"`；
2. 新增 `.include "NbOx_OSC_VAR.lib"`；
3. 只用 `NbOx_OSC_VAR` 替换 U1；
4. 删除 U1 实例参数中的 `Rload=10k`；
5. U2 继续使用原模型和固定 `Rload=10k`；
6. 添加压力控制源 `V_PCTRL`；
7. `V_PCTRL` 正端连接 U1 的 `PCTRL`，负端接地；
8. 将 U1 输入 V1 改成固定 2.5 V。

压力源实际设置为：

```spice
V_PCTRL PCTRL 0 PWL FILE=PCTRL_AT_A_1_r07_c09.pwl
```

## 8. PWL 时间编码方案

PWL 文件直接保存原始 ADC 压力值，而不是保存电阻值。行为电阻在 LTspice 内部完成压力—电阻换算。

为避免 LTspice 在整帧时间内对相邻压力值做线性插值，采用近似零阶保持：

1. 在当前帧结束前 1 ns 重复当前压力；
2. 在精确帧边界写入下一帧压力；
3. 每个原始帧持续 8.333333 ms；
4. 文件从 `t=0` 开始，结束于 `t=2.0 s`；
5. 不再使用当前从负时间开始的旧 PWL 文件。

示例：连续三帧压力为 75、100、150 时：

```text
0                 75
0.008333332333    75
0.008333333333    100
0.016666665667    100
0.016666666667    150
0.024999999       150
0.025             150
```

Notebook 中已实现的 Python 核心逻辑为：

```python
def write_pressure_pwl(pressure_sequence, output_path, duration=2.0):
    pressure_sequence = np.asarray(
        pressure_sequence,
        dtype=np.float64,
    )

    if pressure_sequence.shape != (240,):
        raise ValueError("原始压力序列必须包含240帧")

    pressure_sequence = np.clip(pressure_sequence, 0, 255)

    dt = duration / len(pressure_sequence)
    edge = min(1e-9, dt / 1000)

    with Path(output_path).open(
        "w",
        encoding="ascii",
        newline="\n",
    ) as file:
        file.write(f"0 {pressure_sequence[0]:.9g}\n")

        for frame_index, pressure in enumerate(pressure_sequence):
            frame_end = (frame_index + 1) * dt

            file.write(
                f"{frame_end-edge:.12g} {pressure:.9g}\n"
            )

            if frame_index + 1 < len(pressure_sequence):
                next_pressure = pressure_sequence[frame_index + 1]
                file.write(
                    f"{frame_end:.12g} {next_pressure:.9g}\n"
                )

        file.write(
            f"{duration:.12g} {pressure_sequence[-1]:.9g}\n"
        )
```

单通道测试时，从一个 HDF5 文件中选择：

```python
pressure = file["pressure_data"][...]
taxel_pressure = pressure[:, row, column]
```

然后生成一个对应 `(row,column)` 的 PWL 文件。

## 9. BGATE 是否使用

### 9.1 不使用 BGATE

连接保持为：

```text
U1 OUTPUT → Synapse G
```

当压力回到 `p<=75` 时，Rload 变为 1 TΩ。前振荡器失去来自 2.5 V 输入的驱动，但 `Cparal=1uF` 上已有的电荷仍会通过内部 NbOx 电阻释放，因而 U1 输出和突触输入可能存在一段 RC 尾迹。

这一方案保留了电容放电的物理过程，更接近没有额外数字门控的硬件电路。

### 9.2 使用 BGATE

如果要求：

\[
p\le75\Rightarrow V_{synapse\ gate}=0
\]

则计划把 U1 原始输出命名为 `VOUT_RAW`，断开它与突触 G 端的直接连线，并增加：

```spice
BGATE PRE_SPIKE 0 V=if(V(PCTRL)<=75,0,V(VOUT_RAW))
```

然后连接：

```text
PRE_SPIKE → Synapse G
```

逻辑是：

- `p<=75`：突触输入立即变为 0；
- `p>75`：透传前振荡器波形；
- `VOUT_RAW` 与 `PRE_SPIKE` 之间不能再有直接导线，否则 BGATE 会被旁路。

当前原理图没有使用 BGATE，U1 输出节点 `Vout` 仍直接驱动突触。下一步应先量化真实 RC 尾迹；只有当无压力后的残留激励明显破坏事件编码或突触状态时，再把 BGATE 作为受控消融方案加入。最终实验报告需要明确是否存在这一额外门控。

## 10. 两秒 LTspice 仿真设置

当前 U1 输入源已经改为：

```spice
V1 VIN 0 2.5
```

瞬态仿真设置为：

```spice
.tran 0 2 0 20u uic
.options plotwinsize=0
```

其中：

- 仿真区间为 0–2 秒；
- 最大步长 20 µs；
- 相对于最快约 3 ms 的振荡周期，20 µs 可提供足够时间分辨率；
- `uic` 使用模型内的初始条件；
- `plotwinsize=0` 禁止波形压缩，便于精确计数，但在未来扩展到 256 通道时会显著增加 RAW 文件体积。

当前旧 PWL 文件首行时间为负数，`System_with_TIA.log` 已报告：

```text
Negative value detected.
```

因此新的压力 PWL 必须从非负时间开始。

## 11. 验证状态与后续分阶段验证

### 11.0 已完成的一次动态运行

`System_with_TIA.log` 记录了一次成功运行：

```text
LTspice 26.0.2 for Windows
solver = Normal
Maximum thread count: 8
Per .tran options, skipping operating point for transient analysis.
Total elapsed time: 4.393 seconds.
```

运行成功加载了：

```text
NbOx_OSC_stable.lib
NbOx_OSC_VAR.lib
synapse_advanced_v2.sub
UniversalOpAmp2.lib
```

日志中没有负时间 PWL 警告、未知子电路、端口数量不匹配或收敛错误。运行生成的 `System_with_TIA.raw` 约为 18.1 MB。波形配置中已选择三个节点：压力控制节点（自动节点名 `n002`）、`V(vout)` 和 `V(final_out)`。

需要注意：`System_with_TIA.log/.raw` 的时间戳为 16:45，而当前保存的 `System_with_TIA.asc/.plt` 时间戳约为 16:52。说明成功运行后原理图或绘图配置又被保存过。虽然改动看起来与当前方案一致，仍应对最终保存版本再运行一次，建立严格对应的最终回归记录。

### 11.0.1 已生成的真实压力 PWL

`PCTRL_AT_A_1_r07_c09.pwl` 已通过格式检查：

- 来源：`AT_A_1.h5`；
- taxel：`row=7, column=9`；
- 时间范围：0–2 s；
- 行数：481；
- 时间单调不减；
- 采用帧边界前 1 ns 重复当前值的零阶保持格式；
- 压力最小值：81；
- 压力最大值：180；
- 出现的压力值：81、82、83、85、92、99、131、135、180；
- 序列中共有 4 个原始帧高于理论启振压力 95，对应 PWL 中因零阶保持而重复出现的 8 个点。

`pressure_pwl_generation.ipynb` 已包含生成该文件的完整代码。Notebook 源代码当前把输出目录设为 `neurophic_system_model_var_Rload`，但保存的旧单元格输出文字仍显示 `neurophic_system_model`，说明代码在执行后修改过路径或输出未重新刷新。应重新执行并保存 Notebook，使输出记录和当前源代码一致。

### 11.1 固定压力工作点

真实 240 帧 PWL 已经接入并完成一次运行，但固定压力工作点仍应作为独立回归补做。把 `V_PCTRL` 依次设为直流值：

| PCTRL | Rload | 预期 |
|---:|---:|---|
| 75 | 1 TΩ | 无持续振荡，输出最终回到低电平 |
| 90 | 60 kΩ | 无持续振荡 |
| 96 | 42.86 kΩ | 接近振荡起始点 |
| 120 | 20 kΩ | 约 65 Hz |
| 255 | 5 kΩ | 约 319 Hz |

### 11.2 动态单 taxel 定量测试

样本 `AT_A_1` 的 `(7,9)` taxel 已成功运行 2 秒 PWL。下一步需要对结果进行定量分析，观察：

```text
V(PCTRL)
V(VOUT_RAW) 或 V(Vout)
V(PRE_SPIKE)（如果使用BGATE）
V(VTIA)
V(VDRIVE)
V(final_out)
```

### 11.3 系统级检查

需要重点检查：

1. 压力升高时，U1 发放是否更密；
2. 压力回到基线后，U1 和突触输入的尾迹持续多久；
3. TIA 输出是否长时间卡在 ±5 V 电源轨；
4. `VDRIVE` 是否足以触发 U2；
5. U2 固定 `Rload=10k` 时，驱动电压大约需要超过 1.86 V 才能跨越上阈值并开始振荡；
6. 动态切换 Rload 时是否出现数值收敛错误或非物理尖峰。

固定 Rload 测频时可使用：

```spice
.meas tran TPER TRIG V(VOUT_RAW) VAL=1.4 RISE=10
+ TARG V(VOUT_RAW) VAL=1.4 RISE=11
.meas tran FREQ PARAM 1/TPER
```

动态压力下不宜用单个周期代表整个两秒序列，应按时间窗统计阈值上穿事件，或导出 RAW 数据后在 Python 中计数。

## 12. 计算规模与后续编码路线

完整数据集包含：

```text
7700 samples × 240 frames × 256 taxels
```

逐 taxel、逐样本运行完整两秒 LTspice 系统不现实。建议路线为：

1. 用 LTspice 对固定压力、固定 Rload 和代表性动态序列做标定；
2. 建立 `pressure → Rload → firing response` 的 LUT 或代理模型；
3. 在 Python 中批量生成 `[240,1,16,16]` 编码结果；
4. 随机抽取不同压力、不同 taxel 和不同动态片段回到 LTspice 验证；
5. 对比不使用 BGATE 与使用 BGATE 的脉冲数、尾迹和最终分类性能。

## 13. 当前确定事项

1. 原始帧级仿真采用 240 帧、2 秒、每帧 8.333333 ms；
2. 数据主要范围是 75–255，缓存绝对范围是 64–255；
3. 小于等于 75 的压力统一视为无压力；
4. 无压力时压敏电阻应接近开路，不再使用“75 映射到 40 kΩ”作为最终主方案；
5. 当前主映射为电导—压力线性，对应 `Rload=900k/(p-75)`；
6. LTspice 用 1 TΩ 近似无穷大；
7. 2.5 V 下约从 ADC 96 开始进入持续振荡区；
8. U1 已使用新可变模型，U2 保持固定 10 kΩ；
9. LTspice 已使用行为电阻而不是时间相关 `.param`；
10. 已实现从真实 HDF5 taxel 到两秒零阶保持 PWL 的生成流程；
11. 当前版本不含 BGATE，保留真实 RC 尾迹；
12. LTspice 已成功完成一次动态运行且日志无错误，但最终保存版本仍需要再运行一次形成严格回归。

## 14. 后续待办

1. 重新执行并保存 `pressure_pwl_generation.ipynb`，使单元格输出路径与当前 `neurophic_system_model_var_Rload` 源代码一致；
2. 对当前最终保存的 `System_with_TIA.asc` 再运行一次，更新严格对应的 `.log/.raw`；
3. 完成固定压力 75、90、96、120、255 的 LTspice 回归并记录实际频率；
4. 对 `AT_A_1`、taxel `(7,9)` 的两秒 RAW 波形定量统计 U1 和 U2 事件数、首次发放时间及 RC 尾迹；
5. 检查 TIA 是否饱和、`VDRIVE` 是否达到 U2 启振条件；
6. 比较有无 BGATE 时的输出尾迹、突触状态和最终事件数；
7. 根据仿真结果决定是否调整 `Rmin=5k`、压力基线 75 或电导映射指数；
8. 将 Notebook 中的单 taxel 生成逻辑整理为可重复调用的脚本或函数；
9. 建立可供完整数据集批量使用的编码 LUT/代理模型；
10. 随机抽取不同样本、不同 taxel 和不同压力区间进行 LTspice 回归验证。
