# jkq-bishe

这是一个时间序列预测（Transformer）的起步实现。

## 项目内容

- `timeseries_transformer.py`：
  - `TimeSeriesTransformer` 模型（Encoder-only Transformer）
  - 正弦位置编码 `PositionalEncoding`
  - 滑动窗口数据集 `SlidingWindowDataset`
  - 超参数配置 `ModelConfig`
- `train.py`：
  - 生成带噪声的正弦波样例数据
  - 用滑动窗口构造监督学习样本
  - 训练并输出最后一个样本的预测结果

## 环境准备

建议 Python 3.10+，并安装 PyTorch：

```bash
pip install torch
```

## 运行示例

```bash
python train.py --epochs 10 --batch-size 64 --lr 1e-3
```

你会看到每个 epoch 的 MSE 损失，以及最后一个时间点的真实值和预测值。

## 下一步建议

1. 将 `build_sine_series` 替换为你的真实业务数据读取逻辑（如 CSV / 数据库）。
2. 对特征做标准化，并在预测后做反标准化。
3. 将输出维度从 `1` 扩展到多变量/多步预测。
4. 增加验证集、早停、学习率调度、模型保存和可视化。
