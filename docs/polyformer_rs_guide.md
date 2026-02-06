# PolyFormer → 遥感指代理解复现指南（三数据集统一版）

> 📌 **入口文档**：[`docs/README.md`](./README.md) | **实时进度**：[`docs/autodl_setup_guide.md`](./autodl_setup_guide.md)

> **唯一总目标**：复现 **PolyFormer 在三个遥感数据集上的结果**，回答"**迁移到遥感领域**"的核心问题

> **工程边界**：只做 **数据 / 训练设定 / 推理口径 / 评估口径** 的严格对齐与诊断闭环；**不引入任何新网络模块**

---

## ✅ 验收 Gate（三数据集统一）

### Gate-1：RIS 指标 (RefDIOR + RRSIS-D)

| 指标 | RefDIOR | RRSIS-D | 容忍 |
|------|---------|---------|------|
| P@0.5 | 37.77 | TBD | ±0.5 |
| P@0.6 | 31.95 | TBD | ±0.5 |
| P@0.7 | 23.59 | TBD | ±0.5 |
| P@0.8 | 12.72 | TBD | ±0.5 |
| P@0.9 | 2.01 | TBD | ±0.5 |
| oIoU | 60.39 | TBD | ±0.3 |
| mIoU | 34.27 | TBD | ±0.3 |
| **Sum** | **202.70** | TBD | ±1.0 |

### Gate-2：VG 指标 (RefDIOR + OPT-RSVG)

| 指标 | RefDIOR | OPT-RSVG | 容忍 |
|------|---------|----------|------|
| P@0.5 | 55.39 | TBD | ±0.5 |
| P@0.6 | 48.64 | TBD | ±0.5 |
| P@0.7 | 40.04 | TBD | ±0.5 |
| P@0.8 | 27.45 | TBD | ±0.5 |
| P@0.9 | 6.09 | TBD | ±0.5 |
| oIoU | 71.22 | TBD | ±0.3 |
| mIoU | 49.09 | TBD | ±0.3 |
| **Sum** | **297.92** | TBD | ±1.0 |

---

## 三数据集任务对齐

| 数据集 | 任务 | 输入 | 输出 | Loss 开关 |
|--------|------|------|------|-----------|
| **RefDIOR** | RRSECS (联合) | 遥感图 + 指代表达 | box + mask | `L_box + L_poly` |
| **OPT-RSVG** | RSVG (定位) | 遥感图 + 指代表达 | box | `L_box` only |
| **RRSIS-D** | RRSIS (分割) | 遥感图 + 指代表达 | mask | `L_poly + (opt)L_box` |

---

## 已知硬事实

* **坐标监督**：[0,1] 归一化 → × (num_bins-1) → floor/ceil 量化
* **训练入口**：`train_polyformer_b.sh` → fairseq `--user-dir=polyformer_module`
* **统一超参**：α=2.0 (微小目标重聚焦), Nmax=80 (polygon 点数上限)
* **⚠️ RefDIOR 数据身份**：当前 DIOR-RSVG 是 VG-only（无 mask），不是论文 RefDIOR

---

## Phase 概览

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 0 | 环境配置 + Demo 验证 | ✅ |
| Phase 1 | 读代码定口径 | ✅ |
| Phase 2 | **三数据集适配** | 🔴 阻塞 |
| Phase 3 | 训练与评估闭环 | ⏳ |
| Phase 4 | **三数据集画像分析** | ⏳ |
| Phase 5 | 实验矩阵 + **迁移结论** | ⏳ |

---

# Phase 2：三数据集适配

## 2.1 数据集获取

### RefDIOR (DIOR-RSVG)

| 项目 | 状态 | 说明 |
|------|------|------|
| 来源 | ✅ | https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_ |
| 图像 | ✅ | JPEGImages.zip (17,402 张) |
| Box 标注 | ✅ | Annotations/*.xml |
| Mask 标注 | ❌ | **无！需另外获取** |

### OPT-RSVG

| 项目 | 状态 | 说明 |
|------|------|------|
| 来源 | ✅ | [GitHub](https://github.com/like413/OPT-RSVG) / 百度网盘 提取码: sjoe |
| 论文 | TGRS 2024 | "Language-Guided Progressive Attention for Visual Grounding in RS Images" |
| 规模 | — | 48,952 samples, 25,452 images |
| Split | — | Train 19,580 / Val 4,895 / Test 24,477 |
| 特点 | — | 城市场景、干扰多、14个类别、**只有 box** |

### RRSIS-D

| 项目 | 状态 | 说明 |
|------|------|------|
| 来源 | ✅ | [GitHub (RMSIN)](https://github.com/Lsan2401/RMSIN) / 百度网盘 提取码: sjoe |
| 论文 | — | "Rotated Multi-Scale Interaction Network for Referring RS Image Segmentation" |
| 规模 | — | **17,402** image-caption-mask triplets |
| 特点 | — | 边界细节、复杂形状、**✅ 含 mask 标注！** |

## 2.2 标注统一层

### 统一内部格式 (JSONL)

```json
{
  "image_path": "path/to/image.jpg",
  "expression": "the red car on the left",
  "box": [x1, y1, x2, y2],
  "mask_path": "path/to/mask.png",
  "polygon": [[x1,y1,x2,y2,...], ...],
  "dataset": "refdior|opt_rsvg|rrsis_d",
  "split": "train|val|test"
}
```

### 缺失监督补齐策略

| 数据集 | 缺失项 | 策略 |
|--------|--------|------|
| RefDIOR | mask | 从作者获取，或 SAM2 生成 |
| OPT-RSVG | mask | **不补齐**，只训练 box |
| RRSIS-D | box | mask → MBR (最小外接矩形) |

## 2.3 转换脚本（每数据集一个）

```bash
# RefDIOR
python tools/refdior/convert_refdior.py \
  --root datasets/refdior/raw \
  --out datasets/refdior/processed/refdior_{split}.jsonl

# OPT-RSVG
python tools/opt_rsvg/convert_opt_rsvg.py \
  --root datasets/opt_rsvg/raw \
  --out datasets/opt_rsvg/processed/opt_rsvg_{split}.jsonl

# RRSIS-D
python tools/rrsis_d/convert_rrsis_d.py \
  --root datasets/rrsis_d/raw \
  --out datasets/rrsis_d/processed/rrsis_d_{split}.jsonl
```

## 2.4 验证（Stop condition）

每个数据集必须满足：

1. **Split 规模硬校验**
   - RefDIOR: 26,824 / 3,832 / 7,664
   - OPT-RSVG: 39,162 / — / 9,790
   - RRSIS-D: 12,181 / 1,740 / 3,481

2. **可视化硬校验**：至少 20 张 overlay 正确

3. **Polygon fidelity**：mask→polygon→mask 的 IoU > 0.9

---

# Phase 3：训练与评估闭环

## 3.1 训练策略

### 方案 A：先联合后专精（推荐）

```
1. RefDIOR 训练联合任务 (box+mask) 打底  [50 epochs]
2. OPT-RSVG 微调 box 分支                [20 epochs]
3. RRSIS-D 微调分割分支                  [20 epochs]
```

### 方案 B：三数据集混训

```python
# 采样比例
sampling_ratio = {
    "refdior": 0.5,    # 联合任务占主导
    "opt_rsvg": 0.25,  # 定位强化
    "rrsis_d": 0.25    # 分割强化
}

# Loss 开关
if batch.dataset == "refdior":
    loss = L_box + L_poly
elif batch.dataset == "opt_rsvg":
    loss = L_box
elif batch.dataset == "rrsis_d":
    loss = L_poly
```

## 3.2 评估报表（三数据集统一格式）

```bash
# RefDIOR (RIS + VG)
python tools/common/eval_metrics.py \
  --dataset refdior --task ris --preds outputs/exp_xxx/preds/refdior_ris.jsonl
python tools/common/eval_metrics.py \
  --dataset refdior --task vg --preds outputs/exp_xxx/preds/refdior_vg.jsonl

# OPT-RSVG (VG only)
python tools/common/eval_metrics.py \
  --dataset opt_rsvg --task vg --preds outputs/exp_xxx/preds/opt_rsvg_vg.jsonl

# RRSIS-D (RIS only)
python tools/common/eval_metrics.py \
  --dataset rrsis_d --task ris --preds outputs/exp_xxx/preds/rrsis_d_ris.jsonl
```

---

# Phase 4：三数据集画像分析

## 4.1 分析项（三数据集统一）

### 目标尺度分析 (Micro / Non-micro)

```python
r = Area(obj) / Area(img)
micro_threshold = 0.1  # r < 0.1 为微小目标
```

**输出**：
- r 分布直方图
- micro 样本占比
- micro 在类别上的分布

### 形状复杂度分析 (Complex / Regular)

```python
compactness = 4 * pi * Area / Perimeter^2  # 越小越复杂
```

**输出**：
- compactness 分布
- 复杂形状占比
- 复杂形状集中的类别

### 文本表达分析

**属性提取**：Class / Location / Size

**输出**：
- 属性覆盖率
- location 词分布
- 文本长度分布

## 4.2 画像脚本

```bash
python tools/common/dataset_profile.py \
  --dataset refdior \
  --jsonl datasets/refdior/processed/refdior_train.jsonl \
  --out outputs/rs_datasets/profile/refdior_profile.json

python tools/common/dataset_profile.py \
  --dataset opt_rsvg \
  --jsonl datasets/opt_rsvg/processed/opt_rsvg_train.jsonl \
  --out outputs/rs_datasets/profile/opt_rsvg_profile.json

python tools/common/dataset_profile.py \
  --dataset rrsis_d \
  --jsonl datasets/rrsis_d/processed/rrsis_d_train.jsonl \
  --out outputs/rs_datasets/profile/rrsis_d_profile.json
```

---

# Phase 5：实验矩阵 + 迁移结论

## 5.1 分层评估维度

每个数据集的评估报表必须包含：

| 维度 | 分层 | 意义 |
|------|------|------|
| **尺度** | micro vs non-micro | 微小目标定位难度 |
| **形状** | complex vs regular | 边界拟合难度 |
| **类别** | class-wise | 类别泛化能力 |
| **文本** | 有/无 location | 空间理解能力 |

## 5.2 迁移结论模板

最终需要回答：

1. **定位迁移**：RefDIOR → OPT-RSVG 的 VG 指标变化
   - 城市干扰对定位的影响
   - 多尺度目标的定位准确性

2. **边界迁移**：RefDIOR → RRSIS-D 的 RIS 指标变化
   - 复杂形状的边界拟合
   - 微小目标的分割精度

3. **文本迁移**：跨数据集的 location 词理解
   - 不同场景下的空间描述差异
   - 类别词的泛化

---

## 附录 A：统一超参设定

| 参数 | 值 | 说明 |
|------|-----|------|
| image_size | 512×512 | 统一 resize |
| num_bins | 64 | 坐标量化 |
| α | 2.0 | 微小目标重聚焦 |
| Nmax | 80 | polygon 点数上限 |
| max_lang_len | 20 | 文本长度 |
| batch_size | 32 | 全局 |
| lr | 5e-5 | polynomial decay |
| warmup_ratio | 0.1 | |
| epochs | 50 | strict reproduction |
| seed | 3407 | |

---

## 附录 B：工具脚本清单

### 通用工具 (`tools/common/`)

| 脚本 | 功能 |
|------|------|
| `label_adapter.py` | 标注统一层 (mask↔polygon, mask→box) |
| `dataset_profile.py` | 数据集画像分析 |
| `eval_metrics.py` | 统一评估指标 (P@/oIoU/mIoU/Sum) |
| `vis_samples.py` | 可视化 (支持 overlay box/mask/polygon) |

### 数据集专用

| 目录 | 脚本 |
|------|------|
| `tools/refdior/` | `convert_refdior.py`, `sniff_refdior.py` |
| `tools/opt_rsvg/` | `convert_opt_rsvg.py` |
| `tools/rrsis_d/` | `convert_rrsis_d.py` |

---

## 附录 C：Gate 失败排查树

### 优先级顺序

0. **数据集身份错误**：检查 split 统计是否匹配论文
1. **口径错误**：检查坐标归一化/量化流程
2. **数据错误**：检查 mask/polygon 转换
3. **训练不足**：检查 loss 曲线是否收敛
4. **随机性**：检查 seed 和 deterministic 设置

---

> 最后更新：2026-02-05 (三数据集统一框架)

