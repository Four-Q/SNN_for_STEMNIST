# Log

## 数据

`data.py`

`make_dataloaders`函数：

- 功能：生成训练集、验证集和测试集的数据加载器。
- 参数：
  - `time_steps`：时间步数，默认值为 240。
  - `batch_size`：批量大小，默认值为 64。
  - `train_workers`：训练集 DataLoader 的工作线程数，默认值为 4。
  - `eval_workers`：验证集 DataLoader 的工作线程数，默认值为 2。
  - `pin_memory`：是否将数据加载到固定内存中，默认值为 True。
  - `force_rebuild`：是否强制重建数据缓存，默认值为 False。
- 返回值：`train_loader`、`val_loader`、`test_loader`。

loader中的值：
- x：形状为(B, T, C, H, W) -> (64, 240, 1, 16, 16)
- y：形状为(B,) -> (64,)
- record['sample_id']

**注意：**
1. 在训练循环中需要将x转为float类型
```python
x = x.to(device, non_blocking=True)
x = x.float()   # 在GPU上转换类型
# 或者
x = x.to(
    device,
    dtype=torch.float32,
    non_blocking=True,
)
```

2. 输入到SNN中时，需要将 DataLoader 输出的张量从 (B, T, C, H, W) 交换为 (T, B, C, H, W)，即时间步维度放在最前面。
例如：
```python
    # 进入 SpikingJelly 多步网络前，只交换一次 N 和 T：
    x_snn = x.permute(1, 0, 2, 3, 4).contiguous()
```

3. 总体训练循环中：
```python
device = torch.device("cuda")

for x, y, global_indices in train_loader:
    # x 此时是 CPU 上的 uint8
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    # 在 GPU 上转换类型
    x = x.float()

    # [N,T,C,H,W] -> [T,N,C,H,W]
    x = x.permute(
        1, 0, 2, 3, 4
    ).contiguous()

    output = model(x)

    # 后续 loss、backward 等
```


## 模型

