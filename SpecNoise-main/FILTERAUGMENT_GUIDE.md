# FilterAugment 基线方法使用指南

## 📋 更新概述

本次更新移除了老旧的 **Mixup** 数据增强方法，替换为更先进的 **FilterAugment** 方法。

### 🔧 修改的文件
1. **新建**：`data_aug/filterAugment.py` - FilterAugment 官方实现适配版
2. **修改**：`train_mel.py` - 移除 mixup，添加 filteraugment
3. **修改**：`train_stft.py` - 移除 mixup，添加 filteraugment
4. **修改**：`train_mfcc.py` - 移除 mixup，添加 filteraugment
5. **修改**：`train_multi_feature.py` - 移除 mixup，添加 filteraugment

---

## 🎯 FilterAugment 原理

**FilterAugment**（ICASSP 2022）通过在频率维度上对频谱图应用随机增益来增强数据：

1. **频带划分**：将频谱图在频率维度上随机分成 3-6 个频带
2. **增益应用**：对每个频带应用不同的 dB 增益（-6 到 +6 dB）
3. **滤波类型**：
   - **线性滤波**（默认）：频带间增益线性插值，模拟自然的频率响应变化
   - **步进滤波**：每个频带内增益恒定，模拟更剧烈的频率变化

这种方法模拟了真实声学环境中不同频率范围的音量变化，增强模型对频谱变化的鲁棒性。

---

## 🚀 运行指令

### 使用 FilterAugment（新的基线方法）

```powershell
# Mel 频谱图
python train_mel.py -i "./images_dataset" -b 32 -t 100 -a filteraugment -c "./check-point"

# STFT 频谱图
python train_stft.py -i "./images_dataset" -b 32 -t 100 -a filteraugment -c "./check-point"

# MFCC 频谱图
python train_mfcc.py -i "./images_dataset" -b 32 -t 100 -a filteraugment -c "./check-point"

# 多特征融合
python train_multi_feature.py -i "./images_dataset" -b 32 -t 100 -a filteraugment -c "./check-point"
```

### 无增强（基线对照）

```powershell
python train_mel.py -i "./images_dataset" -b 32 -t 100 -a none -c "./check-point"
```

### 其他增强方法

```powershell
# SpecAugment
python train_mel.py -i "./images_dataset" -b 32 -t 100 -a specaugment -c "./check-point"

# SpecNoise（你的核心算法）
python train_mel.py -i "./images_dataset" -b 32 -t 100 -a RandomReplacementSpecAugmentation -c "./check-point"

# SpecMix
python train_mel.py -i "./images_dataset" -b 32 -t 100 -a specmix -c "./check-point"
```

---

## 📊 建议的实验顺序

### 第一阶段：新基线实验（FilterAugment）
```powershell
python train_mel.py -i "./images_dataset" -b 32 -t 100 -a filteraugment -c "./check-point"
python train_stft.py -i "./images_dataset" -b 32 -t 100 -a filteraugment -c "./check-point"
python train_mfcc.py -i "./images_dataset" -b 32 -t 100 -a filteraugment -c "./check-point"
python train_multi_feature.py -i "./images_dataset" -b 32 -t 100 -a filteraugment -c "./check-point"
```

### 第二阶段：SpecNoise 实验（你的核心算法）
```powershell
python train_mel.py -i "./images_dataset" -b 32 -t 100 -a RandomReplacementSpecAugmentation -c "./check-point"
python train_stft.py -i "./images_dataset" -b 32 -t 100 -a RandomReplacementSpecAugmentation -c "./check-point"
python train_mfcc.py -i "./images_dataset" -b 32 -t 100 -a RandomReplacementSpecAugmentation -c "./check-point"
python train_multi_feature.py -i "./images_dataset" -b 32 -t 100 -a RandomReplacementSpecAugmentation -c "./check-point"
```

### 第三阶段：对比分析
打开 `check-point` 文件夹下的 `history_training_*_filteraugment_时间戳.json` 和 `history_training_*_RandomReplacementSpecAugmentation_时间戳.json`，对比：
- `test_accuracy`：测试集准确率
- 计算提升幅度：`SpecNoise 准确率 - FilterAugment 准确率`

---

## ⚙️ FilterAugment 参数配置

当前配置（在四个 `train_*.py` 中）：

```python
filter_augmenter = FilterAugment(
    db_range=[-6, 6],     # dB 增益范围：-6 到 +6 dB
    n_band=[3, 6],        # 频带数量范围：3 到 6 个频带
    min_bw=6,             # 最小频带宽度：6 个频率 bin
    filter_type="linear"  # 滤波类型：线性插值
)
```

如需调整参数，直接修改 `train_*.py` 文件中的上述代码。

---

## 📁 输出文件命名

运行后会在 `check-point` 目录下生成：
- `output_mel_filteraugment_20260409-123456.pth` - 模型权重
- `history_training_mel_filteraugment_20260409-123456.json` - 训练记录

---

## ✅ 验证安装

运行以下命令测试 FilterAugment 是否正常工作：

```powershell
python -c "from data_aug.filterAugment import FilterAugment; print('FilterAugment 导入成功！')"
```

如果输出 `FilterAugment 导入成功！`，说明安装正确。

---

## 📚 参考论文

**FilterAugment**:  
- Paper: https://arxiv.org/abs/2107.03649 (ICASSP 2022)
- Official Code: https://github.com/DCASE-REPO/DESED_task

**SpecNoise**（你的方法）：
- 详见项目根目录的 `SpecNoise__Enhancing_Bird_Sound_Classification...pdf`
