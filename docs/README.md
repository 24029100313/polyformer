# PolyFormer → 遥感指代理解复现项目

> **⚠️ 重要：每次对话必须先读这个文档，然后按指示查看子文档**

---

## 🎯 项目目标

**复现 PolyFormer 在三个遥感指代理解数据集上的结果**，实现"迁移到遥感领域"的完整验证：

| 数据集 | 任务类型 | 输出 | 对应论文 Table |
|--------|---------|------|---------------|
| **RefDIOR** | RRSECS (联合) | box + mask | Table I (RIS) + Table III (VG) |
| **OPT-RSVG** | RSVG (定位) | box only | Table III (VG) |
| **RRSIS-D** | RRSIS (分割) | mask only | Table I (RIS) |

**不是**复现 SeeFormer 模型本身，只做数据/训练/评估的严格对齐。

---

## 📊 三数据集任务地图

```
遥感指代理解 (Remote Sensing Referring Understanding)
├── RRSECS (RefDIOR) ──── 联合任务 ──── box + mask/polygon
├── RSVG (OPT-RSVG) ───── 定位任务 ──── box only
└── RRSIS (RRSIS-D) ───── 分割任务 ──── mask only
```

### 数据集基本信息

| 数据集 | 样本数 | 标注类型 | Train/Val/Test | 特点 |
|--------|--------|---------|----------------|------|
| **RefDIOR** | 38,320 | image-expr-box-mask | 26,824/3,832/7,664 | 联合学习核心 |
| **OPT-RSVG** | 48,952 | image-expr-box | 39,162/—/9,790 | 城市场景、干扰多、多尺度 |
| **RRSIS-D** | 17,402 | image-expr-mask | 12,181/1,740/3,481 | 边界细节、复杂形状 |

---

## 📚 核心文档结构

| 文档 | 用途 | 更新频率 |
|------|------|---------|
| [`polyformer_rs_guide.md`](./polyformer_rs_guide.md) | **全局工程指南**：Phase 0-5、验收 Gate、三数据集统一流程 | 稳定 |
| [`autodl_setup_guide.md`](./autodl_setup_guide.md) | **实时进度追踪**：当前状态、操作记录、问题解决 | **每次操作后即时更新** |
| [`../outputs/rs_datasets/fact_sheet.md`](../outputs/rs_datasets/fact_sheet.md) | **事实记录**：代码口径、数据格式、关键发现 | Phase 1 后必须 |

### 为什么需要三个数据集？

1. **RefDIOR** 是"联合学习"的核心 → 验证 box+mask 同时输出
2. **OPT-RSVG** 强化"定位能力" → 验证城市干扰下的 box 定位
3. **RRSIS-D** 强化"分割能力" → 验证复杂边界的 mask 精度

分开评估可以回答：**到底是定位迁移难、边界迁移难、还是文本对齐迁移难**

---

## 🔄 工作流程（必须遵守）

### 1. 每次新对话开始时

```
请先查看以下文档了解项目状态：
1. docs/README.md（本文档 - 入口）
2. docs/autodl_setup_guide.md（当前进度）
3. docs/polyformer_rs_guide.md（全局指南的相关 Phase）
4. outputs/rs_datasets/fact_sheet.md（事实记录）

然后告诉我当前状态和下一步操作。
```

### 2. 每次操作后

AI 助手必须：

1. **即时输出结果**：
   ```
   [Step X.Y] 操作描述
   - 数据集: RefDIOR / OPT-RSVG / RRSIS-D / 全部
   - 命令: <执行的命令>
   - 状态: ✅ 成功 / ❌ 失败 / 🔄 进行中
   - 输出: <关键输出摘要>
   - 下一步: <下一步操作>
   ```

2. **即时更新 `autodl_setup_guide.md`**

3. **提供可供 GPT-Pro 检查的摘要**

---

## 📊 当前项目状态速查

> ⚠️ 此处为快速参考，详细信息请查看 `autodl_setup_guide.md`

### 当前 Phase

| Phase | 状态 | 说明 |
|-------|------|------|
| Phase 0 | ✅ 完成 | 环境配置 + Demo 验证 |
| Phase 1 | ✅ 完成 | 读代码定口径 (fact_sheet.md) |
| Phase 2 | 🔴 阻塞 | 三数据集适配 (RefDIOR 无 mask，需获取/生成) |
| Phase 3 | ⏳ 待开始 | 训练与评估闭环 |
| Phase 4 | ⏳ 待开始 | 三数据集画像分析 |
| Phase 5 | ⏳ 待开始 | 实验矩阵 + 迁移结论 |

### 数据集获取状态

| 数据集 | 图像 | Box 标注 | Mask 标注 | 状态 |
|--------|------|---------|----------|------|
| RefDIOR (DIOR-RSVG) | ✅ | ✅ | ❌ 无 | 🔴 需获取 mask |
| OPT-RSVG | ⏳ | ⏳ | N/A | ⏳ 待下载 |
| RRSIS-D | ⏳ | ⏳ | ⏳ | ⏳ 待下载 |

---

## 🎯 验收 Gate 目标（写死）

### Gate-1：RIS 指标 (RefDIOR + RRSIS-D)

| 指标 | RefDIOR 目标 | RRSIS-D 目标 | 容忍 |
|------|-------------|--------------|------|
| P@0.5 | 37.77 | TBD | ±0.5 |
| P@0.6 | 31.95 | TBD | ±0.5 |
| P@0.7 | 23.59 | TBD | ±0.5 |
| P@0.8 | 12.72 | TBD | ±0.5 |
| P@0.9 | 2.01 | TBD | ±0.5 |
| oIoU | 60.39 | TBD | ±0.3 |
| mIoU | 34.27 | TBD | ±0.3 |
| **Sum** | **202.70** | TBD | ±1.0 |

### Gate-2：VG 指标 (RefDIOR + OPT-RSVG)

| 指标 | RefDIOR 目标 | OPT-RSVG 目标 | 容忍 |
|------|-------------|---------------|------|
| P@0.5 | 55.39 | TBD | ±0.5 |
| P@0.6 | 48.64 | TBD | ±0.5 |
| P@0.7 | 40.04 | TBD | ±0.5 |
| P@0.8 | 27.45 | TBD | ±0.5 |
| P@0.9 | 6.09 | TBD | ±0.5 |
| oIoU | 71.22 | TBD | ±0.3 |
| mIoU | 49.09 | TBD | ±0.3 |
| **Sum** | **297.92** | TBD | ±1.0 |

---

## 🛠️ 统一超参设定（遥感域默认配置）

| 参数 | 值 | 说明 |
|------|-----|------|
| 图像尺寸 | 512×512 | 统一 resize |
| α (微小目标重聚焦) | 2.0 | 论文验证最佳 |
| Nmax (polygon 采样点) | 80 | 超过 80 收益很小 |
| max_lang_len | 20 | RRSECS 统一设定 |
| seed | 3407 | 可复现 |
| batch_size | 32 | 全局 batch |

---

## 📁 项目目录结构（三数据集统一）

```
polyformer/
├── docs/
│   ├── README.md                    # 本文档（入口）
│   ├── polyformer_rs_guide.md       # 全局工程指南（三数据集）
│   └── autodl_setup_guide.md        # 实时进度追踪
├── tools/
│   ├── common/                      # 通用工具
│   │   ├── dataset_profile.py       # 数据集画像分析（通用）
│   │   ├── label_adapter.py         # 标注统一层
│   │   └── eval_metrics.py          # 统一评估指标
│   ├── refdior/                     # RefDIOR 专用
│   ├── opt_rsvg/                    # OPT-RSVG 专用
│   └── rrsis_d/                     # RRSIS-D 专用
├── datasets/
│   ├── refdior/
│   │   ├── raw/                     # 原始数据
│   │   └── processed/               # 转换后 JSONL
│   ├── opt_rsvg/
│   │   ├── raw/
│   │   └── processed/
│   └── rrsis_d/
│       ├── raw/
│       └── processed/
└── outputs/
    └── rs_datasets/
        ├── fact_sheet.md            # 三数据集事实记录
        ├── profile/                 # 数据集画像输出
        │   ├── refdior_profile.json
        │   ├── opt_rsvg_profile.json
        │   └── rrsis_d_profile.json
        └── exp_<ID>/                # 实验输出
            ├── args.json
            ├── train.log
            ├── eval/
            │   ├── refdior_ris.json
            │   ├── refdior_vg.json
            │   ├── opt_rsvg_vg.json
            │   └── rrsis_d_ris.json
            └── vis/
```

---

## ⚠️ 关键约束（不可违反）

1. **不改模型结构**：只做数据/训练/评估的对齐
2. **三数据集统一流程**：同一套分析/评估脚本，输出同规格报表
3. **标注统一层**：mask↔polygon、mask→box 的转换逻辑一致
4. **Loss 开关策略**：
   - RefDIOR: `L_box + L_poly/mask`
   - OPT-RSVG: `L_box` only
   - RRSIS-D: `L_poly/mask + (可选)L_box_from_mask`
5. **分层评估**：每个数据集都输出 micro/complex/text 三维分层

---

## 🔗 相关链接

- PolyFormer 原仓库：https://github.com/amazon-science/polygon-transformer
- SeeFormer 论文：[待补充]
- 数据集：
  - RefDIOR (DIOR-RSVG): https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_
  - **OPT-RSVG**: [GitHub](https://github.com/like413/OPT-RSVG) / 百度网盘 提取码: sjoe
  - **RRSIS-D**: [GitHub (RMSIN)](https://github.com/Lsan2401/RMSIN) / 百度网盘 提取码: sjoe ✅ **含 mask 标注！**

---

> 最后更新：2026-02-05 (扩展到三数据集统一框架)
