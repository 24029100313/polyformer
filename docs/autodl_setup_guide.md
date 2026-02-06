# PolyFormer + 遥感指代理解 AutoDL 配置指南

> 📌 **入口文档**：[`docs/README.md`](./README.md)
> 
> 最后更新：2026-02-06
> 当前状态：**🟡 Phase 2 基本完成 — 三数据集远程 JSONL 已生成，RRSIS-D 可进入训练，RefDIOR 缺 mask 仍阻塞**

---

## 🔄 工作流程约定（重要）

> ⚠️ 详细工作流程请查看 [`docs/README.md`](./README.md)

### 每次对话/操作必做：

1. **检查三个核心文档：**
   - [`docs/README.md`](./README.md)（入口文档 - 总体说明）
   - `docs/autodl_setup_guide.md`（本文档 - 实时进度追踪）
   - [`docs/polyformer_refdior_guide.md`](./polyformer_refdior_guide.md)（全局工程指南）

2. **即时输出每一步结果：**
   - 命令执行后立即输出结果
   - 方便用户给 GPT-Pro 检查是否需要调整策略
   - 格式：`[Step X.Y] 操作描述 → 结果/状态`

3. **即时更新进度文档：**
   - 每完成一个步骤，立即更新本文档的状态
   - 记录遇到的问题和解决方案
   - 保持"当前状态"字段实时准确

### 输出格式模板：

```
[Step X.Y] 操作描述
- 命令: <执行的命令>
- 状态: ✅ 成功 / ❌ 失败 / 🔄 进行中
- 输出: <关键输出摘要>
- 下一步: <下一步操作>
```

---

## 📋 总体目标

在 AutoDL 远程服务器上配置 PolyFormer 环境，复现三个遥感指代理解数据集上的结果：

| 数据集 | 任务 | 输出 | 状态 |
|--------|------|------|------|
| **RefDIOR** | RRSECS (联合) | box + mask | 🔴 缺 mask |
| **OPT-RSVG** | RSVG (定位) | box | ⏳ 待下载 |
| **RRSIS-D** | RRSIS (分割) | mask | ⏳ 待下载 |

**📚 详细工程指南：** 请参阅 [`docs/polyformer_rs_guide.md`](./polyformer_rs_guide.md)

---

## 🗂️ Phase 概览

| Phase | 内容 | 状态 | 详细指南 |
|-------|------|------|---------|
| Phase 0 | 环境配置 + Demo 验证 | ✅ 完成 | 本文档 |
| Phase 1 | 理解 PolyFormer 数据流 | ✅ 完成 | [指南 Phase 1](./polyformer_rs_guide.md#phase-1) |
| Phase 2 | **三数据集适配** | 🔴 阻塞 | [指南 Phase 2](./polyformer_rs_guide.md#phase-2) |
| Phase 3 | 训练评估闭环 | ⏳ 待开始 | [指南 Phase 3](./polyformer_rs_guide.md#phase-3) |
| Phase 4 | **三数据集画像分析** | ⏳ 待开始 | [指南 Phase 4](./polyformer_rs_guide.md#phase-4) |
| Phase 5 | 实验矩阵 + **迁移结论** | ⏳ 待开始 | [指南 Phase 5](./polyformer_rs_guide.md#phase-5) |

### 📊 三数据集获取与转换状态

| 数据集 | 图像 | Box | Mask | Split 验证 | JSONL 转换 | 状态 |
|--------|------|-----|------|-----------|-----------|------|
| **RefDIOR** | ✅ 解压 | ✅ | ❌ 无 | ⚠️ 不匹配 | ✅ 38,320 (bbox-only) | 🔴 需获取 mask |
| **OPT-RSVG** | ✅ 解压 | ✅ | N/A | ⚠️ 与文档不匹配 | ✅ 48,952 英文 ✅ | 🟡 Split问题待确认 |
| **RRSIS-D** | ✅ 解压 | ✅ | ✅ RLE | ✅ 17,399/17,402 | ✅ mask→polygon | ✅ **可训练** |

**数据集下载链接**：

| 数据集 | GitHub | 百度网盘 |
|--------|--------|---------|
| **OPT-RSVG** | [like413/OPT-RSVG](https://github.com/like413/OPT-RSVG) | 提取码: sjoe |
| **RRSIS-D** | [Lsan2401/RMSIN](https://github.com/Lsan2401/RMSIN) | 提取码: sjoe |

> ⚠️ **重要发现**：RRSIS-D 数据集包含完整的 **mask 标注** (17,402 image-caption-mask triplets)！

---

## ✅ 当前完成状态总览

| 项目 | 状态 | 备注 |
|------|------|------|
| AutoDL 实例 | ✅ | PyTorch 1.12 / CUDA 11.3 |
| VSCode Remote-SSH | ✅ | |
| 克隆仓库 | ✅ | /root/autodl-tmp/projects/polyformer |
| venv 虚拟环境 | ✅ | Python 3.8 |
| 安装依赖 | ✅ | 所有包已安装 |
| PYTHONPATH 配置 | ✅ | fairseq 路径已添加 |
| Backbone 权重 | ✅ | Swin-base/large, BERT |
| PolyFormer 权重 | ✅ | 6个 .pt 文件 |
| RefCOCO 标注 | ✅ | refcoco/refcoco+/refcocog 已解压 |
| COCO train2014 图像 | ✅ | 82783 张图像 |
| COCO val2014 图像 | ✅ | 已合并到 train2014（共 123287 张） |
| 符号链接 | ✅ | refer/data/images/mscoco |
| 环境验证 | ✅ | 所有模块导入成功 |
| 存储扩容 | ✅ | 已扩容（finetune TSV 文件较大） |
| **生成微调数据** | ✅ | create_finetuning_data.py 完成 |
| **Demo 测试** | ✅ | 端到端推理成功 (45步, 36.97% mask) |

---

## 📝 即时进度追踪（最新操作记录）

> 此部分记录每次操作的即时结果，方便 GPT-Pro 检查和调整策略。

### 最近操作记录

| 时间 | Step | 操作 | 状态 | 结果摘要 |
|------|------|------|------|---------|
| 2026-02-05 | 0.12 | 生成微调数据 | ✅ | create_finetuning_data.py 完成 |
| - | - | 下载 val2014 | ✅ | 40504 张图像 |
| - | - | 合并 val2014 到 train2014 | ✅ | rsync 完成，共 123287 张 |
| - | - | 存储扩容 | ✅ | 解决空间不足问题 |
| 2026-02-05 15:21 | 0.13 | BERT 权重本地化 | ✅ | 修改代码使用本地 bert-base-uncased |
| 2026-02-05 15:25 | 0.14 | **Demo 测试** | ✅ | **端到端推理成功！45步，mask 36.97%** |
| 2026-02-05 15:26 | - | **Phase 0 完成** | ✅ | 准备进入 Phase 1 |
| **2026-02-05** | **1.1-1.5** | **Phase 1: 读代码定口径** | ✅ | **fact_sheet.md 已生成** |
| 2026-02-05 17:00 | 2.1 | 上传 DIOR-RSVG 数据 | ✅ | Annotations.zip + split txt 已上传 |
| 2026-02-05 17:15 | 2.2 | Sniff RefDIOR 格式 | ✅ | **XML 格式，17402 图像，38320 样本，无 mask** |
| 2026-02-05 17:30 | 2.3 | 创建 convert_refdior.py | ✅ | 脚本已上传到远程 |
| 2026-02-05 20:16 | 2.4 | 上传 JPEGImages.zip | ✅ | **5.0 GB 上传成功，Python zipfile 可读** |
| 2026-02-05 21:00 | 2.5 | **⚠️ 数据集身份问题** | 🔴 | **确认 DIOR-RSVG ≠ RefDIOR（无 mask）** |
| **2026-02-06** | **2.6** | **RRSIS-D Sniff + 转换** | ✅ | **17,399/17,402 mask→polygon, fidelity=0.897** |
| 2026-02-06 | 2.7 | **OPT-RSVG 转换** | ✅ | **48,952 bbox→polygon, 英文表达 ✅, split 不匹配** |
| 2026-02-06 | 2.8 | **label_adapter.py 修复** | ✅ | **内嵌 poly_utils 避免 fairseq 导入链** |
| 2026-02-06 | 2.9 | **远程解压全部数据集** | ✅ | **3 数据集 ZIP 全部解压，已删除 ZIP 释放空间** |
| 2026-02-06 | 2.10 | **远程运行三数据集转换** | ✅ | **DIOR-RSVG 38320 + OPT-RSVG 48952(英文✅) + RRSIS-D 17399** |

### 待确认项目（需要用户反馈）

- [x] `create_finetuning_data.py` 是否运行完成？ → ✅ 已完成
- [x] 生成的 TSV 文件大小和数量？ → ✅ 已生成
- [x] Demo 测试结果？ → ✅ 端到端推理成功！

### 🔴 当前 Blocker（必须解决 - 三数据集）

#### Blocker 1：RefDIOR 缺少 mask（仍然阻塞）

| 对比项 | DIOR-RSVG（当前数据） | RefDIOR（论文期望） |
|--------|----------------------|-------------------|
| Split | 26,991/3,829/7,500 | 26,824/3,832/7,664 |
| Mask | ❌ 无 | ✅ 有 |
| 可复现 RIS？ | ❌ 不可能 | ✅ |

**解决方案**：
1. **方案 A（推荐）**：联系 SeeFormer/RRSECS 作者获取 gt mask
2. **方案 B**：用 SAM2 生成 pseudo mask
3. **方案 C（临时）**：先只跑 VG 验证 pipeline

#### Blocker 2：OPT-RSVG Split 不匹配（需确认）

**Split 数量不匹配**：

| 来源 | Train | Val | Test | Total |
|------|-------|-----|------|-------|
| **文档期望** | 39,162 | — | 9,790 | 48,952 |
| **实际 split.zip** | 19,580 | 4,895 | 24,477 | 48,952 |

Total 一致 (48,952) 但分配完全不同。需要确认 split.zip 是否为正确版本，或文档数字是否有误。

**~~问题 2 - 中文表达~~ → 已解决 (2026-02-06)**：
- ✅ 已替换为正确的英文版 `Annotations.zip`（100% 英文描述）
- 之前项目中的 `Annotations.zip` 被错误替换为中文版（与 `Annotations_CN.zip` 相同）
- 现已确认 48,952 条描述全部为英文，BERT-base-uncased 可正常处理

#### ~~Blocker 3：RRSIS-D 图像未解压~~ → 已解决 (2026-02-06)

- ✅ `JPEGImages.zip` 已解压（17,402 张图像）
- ✅ 转换脚本已在远程运行，JSONL 已生成
- ✅ ZIP 文件已删除释放空间

#### 三数据集统一下一步

| 数据集 | 下一步 | 优先级 |
|--------|--------|--------|
| RefDIOR | 获取 mask 或生成 pseudo mask | 🔴 高 |
| OPT-RSVG | 确认 split（语言问题已解决 ✅） | 🟡 中 |
| RRSIS-D | ✅ 数据就绪，进入 Phase 3 训练 | 🟢 可开始 |

### ✅ Phase 0 验证结果

```
✅ 推理成功！
- 模型类型: PolyFormerModel
- 设备: cuda:0
- 推理步数: 45 步
- 输出图像形状: (512, 512, 3)
- 输出mask形状: (512, 512)
- mask 非零像素比例: 36.97%
```

---

## 🖥️ Phase 0：环境配置详细步骤

### Step 0.1：AutoDL 实例创建 ✅ 已完成

**配置选择：**
- 镜像：`PyTorch 1.12.0 / Python 3.8 / CUDA 11.3`
- 可以选择「无卡模式」配置环境，节省费用
- GPU：训练时选择 RTX 3090 或更好

---

### Step 0.2：VSCode Remote-SSH 连接 ✅ 已完成

1. 安装 VSCode 扩展：`Remote - SSH`
2. 配置 `~/.ssh/config`：
```
Host autodl-polyformer
    HostName region-X.autodl.com
    User root
    Port XXXXX
```
3. 连接后在终端操作

---

### Step 0.3：开启学术加速 ✅ 已完成

```bash
# 连接后首先执行
source /etc/network_turbo

# 配置 pip 国内源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### Step 0.4：克隆仓库 ✅ 已完成

```bash
cd /root/autodl-tmp
mkdir -p projects && cd projects
git clone https://github.com/amazon-science/polygon-transformer.git polyformer
cd polyformer
```

---

### Step 0.5：创建虚拟环境 ✅ 已完成

由于 conda 在 AutoDL 上有问题，使用 Python venv：

```bash
cd /root/autodl-tmp/projects/polyformer
python -m venv venv
source venv/bin/activate

# 配置 pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade pip
```

**激活环境命令（每次连接都要执行）：**
```bash
cd /root/autodl-tmp/projects/polyformer
source venv/bin/activate
```

---

### Step 0.6：安装依赖 ✅ 已完成

按以下顺序安装（已验证可行）：

```bash
# 1. 安装 PyTorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# 2. 安装 numpy 和 cython（pycocotools 需要）
pip install numpy==1.23.5 cython==0.29.36

# 3. 安装 pycocotools
pip install pycocotools --no-build-isolation

# 4. 降级 pip（解决 omegaconf 兼容性）
pip install pip==23.3.2

# 5. 安装 fairseq（项目内置）
cd /root/autodl-tmp/projects/polyformer/fairseq
pip install -e .
cd ..

# 6. 安装其他依赖
pip install opencv-python timm ftfy==6.0.3 tensorboardX==2.4.1 einops scikit-image tensorboard

# 7. 安装兼容版本的 pytorch_lightning
pip install pytorch_lightning==1.9.5

# 8. 安装 tokenizers（指定版本避免构建问题）
pip install tokenizers==0.13.3

# 9. 安装其他
pip install datasets rouge_score

# 10. 降级 protobuf（解决 tensorboardX 兼容性）
pip install protobuf==3.20.0
```

---

### Step 0.7：设置 PYTHONPATH ✅ 已完成

```bash
# 永久添加到 venv 激活脚本
echo 'export PYTHONPATH="/root/autodl-tmp/projects/polyformer/fairseq:$PYTHONPATH"' >> /root/autodl-tmp/projects/polyformer/venv/bin/activate

# 重新激活
source venv/bin/activate
```

---

### Step 0.8：验证环境 ✅ 已完成

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA compiled:', torch.version.cuda)

from models.polyformer import PolyFormerModel
from bert.tokenization_bert import BertTokenizer
from tasks.refcoco import RefcocoTask
print('✓ 所有核心模块导入成功')
"
```

**预期输出：**
```
PyTorch: 1.12.1+cu113
CUDA compiled: 11.3
✓ 所有核心模块导入成功
```

---

### Step 0.9：下载/上传权重文件 ✅ 已完成

#### 服务器下载 Backbone 权重
```bash
cd /root/autodl-tmp/projects/polyformer
mkdir -p pretrained_weights weights
cd pretrained_weights

source /etc/network_turbo

# Swin-base
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth

# Swin-large
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth

# BERT-base
wget https://hf-mirror.com/bert-base-uncased/resolve/main/pytorch_model.bin -O bert-base-uncased-pytorch_model.bin
```

#### 从本地上传 PolyFormer 权重

**本地资源路径：** `D:\science\polyformer\polyformer_assets\`

通过 AutoDL JupyterLab 上传以下文件到对应目录：

| 本地文件 | 远程目标 |
|---------|---------|
| `weights/polyformer/*.pt` | `/root/autodl-tmp/projects/polyformer/weights/` |
| `datasets/annotations/*` | `/root/autodl-tmp/projects/polyformer/datasets/annotations/` |
| `datasets/refcoco/*.zip` | `/root/autodl-tmp/projects/polyformer/datasets/refcoco/` |

**上传状态：** ✅ 全部完成

---

### Step 0.10：下载 COCO train2014 ✅ 已完成

由于本地上传的文件损坏，在服务器直接下载：

```bash
cd /root/autodl-tmp/projects/polyformer/datasets/coco

# 安装 aria2（多线程下载更稳定）
apt-get update && apt-get install -y aria2

# 下载（约 13GB）
source /etc/network_turbo
aria2c -x 16 -s 16 -k 1M http://images.cocodataset.org/zips/train2014.zip

# 如果下载中断，重新执行同样的命令即可续传
```

**下载状态：** ✅ 已完成（14GB）

---

### Step 0.11：解压数据集 ✅ 已完成

```bash
cd /root/autodl-tmp/projects/polyformer

# 解压 RefCOCO 标注
unzip -o datasets/refcoco/refcoco.zip -d refer/data/
unzip -o datasets/refcoco/refcoco+.zip -d refer/data/
unzip -o datasets/refcoco/refcocog.zip -d refer/data/

# 解压 COCO 图像（train2014 + val2014）
mkdir -p datasets/images/mscoco
unzip datasets/coco/train2014.zip -d datasets/images/mscoco/
unzip datasets/coco/val2014.zip -d datasets/images/mscoco/

# 合并 val2014 到 train2014（RefCOCO 需要两者的图像）
rsync -av --remove-source-files datasets/images/mscoco/val2014/ datasets/images/mscoco/train2014/
rmdir datasets/images/mscoco/val2014

# 创建符号链接
mkdir -p refer/data/images
ln -sf /root/autodl-tmp/projects/polyformer/datasets/images/mscoco refer/data/images/mscoco
```

**解压状态：**
- ✅ refcoco.zip → refer/data/refcoco/
- ✅ refcoco+.zip → refer/data/refcoco+/
- ✅ refcocog.zip → refer/data/refcocog/
- ✅ train2014.zip → datasets/images/mscoco/train2014/ (82783 张图像)
- ✅ val2014.zip → 已合并到 train2014/ (40504 张图像)
- ✅ 合并后总计 123287 张图像
- ✅ 符号链接已创建

**⚠️ 重要：** RefCOCO 数据集引用的图像来自 COCO train2014 和 val2014，必须下载两者！

---

### Step 0.12：生成微调数据 🔄 进行中

**问题记录：**
1. 无卡模式内存不足被 Killed → 切换到有卡模式
2. 缺少 val2014 图像导致 FileNotFoundError → 下载并合并 val2014
3. 磁盘空间不足 → 扩容存储

**当前状态：** 正在运行

```bash
cd /root/autodl-tmp/projects/polyformer
source venv/bin/activate

# 生成微调用的 tsv 文件（包含 base64 编码的图像，文件较大）
python data/create_finetuning_data.py
```

**预期输出文件：**
- `datasets/finetune/refcoco/refcoco_train.tsv`
- `datasets/finetune/refcoco/refcoco_val.tsv`
- `datasets/finetune/refcoco/refcoco_testA.tsv`
- `datasets/finetune/refcoco/refcoco_testB.tsv`
- `datasets/finetune/refcoco+/...`
- `datasets/finetune/refcocog/...`
- `datasets/finetune/refcoco+g_train_shuffled.tsv`

**⚠️ 注意：** TSV 文件包含 base64 编码的图像，总大小可能达到 **30-50GB+**，请确保有足够存储空间！

---

### Step 0.13：运行 Demo 测试 ⏳ 待完成

**需要有卡模式**

```bash
cd /root/autodl-tmp/projects/polyformer
source venv/bin/activate

# 确认 GPU 可用
nvidia-smi

# 运行 demo
python demo.py
```

---

## 📁 远程目录结构（当前状态）

```
/root/autodl-tmp/projects/polyformer/
├── venv/                           # Python 虚拟环境 ✅
├── pretrained_weights/             # Backbone 预训练权重
│   ├── swin_base_patch4_window12_384_22k.pth    ✅ (430 MB)
│   ├── swin_large_patch4_window12_384_22k.pth   ✅ (886 MB)
│   └── bert-base-uncased-pytorch_model.bin      ✅ (420 MB)
├── weights/                        # PolyFormer 权重
│   ├── polyformer_b_pretrain.pt                 ✅ (105 MB)
│   ├── polyformer_b_refcoco.pt                  ✅ (104 MB)
│   ├── polyformer_b_refcoco+.pt                 ✅ (112 MB)
│   ├── polyformer_l_pretrain.pt                 ✅ (106 MB)
│   ├── polyformer_l_refcoco.pt                  ✅ (258 MB)
│   └── polyformer_l_refcoco+.pt                 ✅ (164 MB)
├── datasets/
│   ├── annotations/                             ✅
│   │   ├── instances.json                       (73 MB)
│   │   ├── ix_to_token.pkl
│   │   ├── token_to_ix.pkl
│   │   └── word_emb.npz
│   ├── coco/
│   │   └── train2014.zip                        ✅ (14 GB)
│   ├── images/
│   │   └── mscoco/
│   │       └── train2014/                       ✅ (70825 张图像)
│   ├── refcoco/                                 ✅ (zip 文件)
│   └── finetune/                                🔄 待生成
├── refer/
│   └── data/
│       ├── refcoco/                             ✅ 已解压
│       │   ├── instances.json
│       │   ├── refs(google).p
│       │   └── refs(unc).p
│       ├── refcoco+/                            ✅ 已解压
│       │   ├── instances.json
│       │   └── refs(unc).p
│       ├── refcocog/                            ✅ 已解压
│       │   ├── instances.json
│       │   ├── refs(google).p
│       │   └── refs(umd).p
│       └── images/
│           └── mscoco -> (符号链接)             ✅
├── fairseq/                        # fairseq 源码 ✅
├── models/                         # PolyFormer 模型 ✅
├── data/                           # 数据处理代码 ✅
├── tasks/                          # 任务定义 ✅
├── outputs/                        # 输出目录
│   └── env/                        # 环境信息
└── docs/                           # 文档
```

---

## 🔧 常用命令速查

### 每次连接后激活环境
```bash
cd /root/autodl-tmp/projects/polyformer
source venv/bin/activate
source /etc/network_turbo  # 开启加速（可选）
```

### 完整环境验证
```bash
python -c "
import os
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)

from models.polyformer import PolyFormerModel
from bert.tokenization_bert import BertTokenizer
from tasks.refcoco import RefcocoTask
from data.refcoco_dataset import RefcocoDataset
print('✓ 所有模块导入成功')

# 检查关键文件
files = [
    'pretrained_weights/swin_base_patch4_window12_384_22k.pth',
    'weights/polyformer_b_pretrain.pt',
    'refer/data/refcoco/instances.json',
    'datasets/images/mscoco/train2014',
]
for f in files:
    status = '✓' if os.path.exists(f) else '✗'
    print(f'{status} {f}')
"
```

### 检查文件
```bash
ls -lh pretrained_weights/
ls -lh weights/
ls -lh datasets/annotations/
ls datasets/images/mscoco/train2014/ | wc -l
ls -la refer/data/
```

---

## ⚠️ 已解决的问题记录

| 问题 | 解决方案 |
|------|---------|
| conda create 失败 | 使用 Python venv 替代 |
| pycocotools 编译失败 | `pip install pycocotools --no-build-isolation` |
| puccinialin 找不到 | 降级 pip 到 23.3.2 |
| omegaconf metadata 无效 | 降级 pip 到 23.3.2 |
| pytorch_lightning 升级 torch | 安装 `pytorch_lightning==1.9.5` |
| tokenizers 构建失败 | 安装 `tokenizers==0.13.3` |
| protobuf 版本冲突 | 安装 `protobuf==3.20.0` |
| fairseq metrics 导入失败 | 设置 PYTHONPATH 到 fairseq 目录 |
| train2014.zip 损坏 | 使用 aria2 重新下载 |
| 生成数据时 Killed | 需要有卡模式（更多内存） |
| **FileNotFoundError: 图像不存在** | RefCOCO 需要 train2014 + val2014 两者的图像，必须下载并合并 |
| **mv: Argument list too long** | 使用 `rsync -av --remove-source-files` 替代 `mv *` |
| **No space left on device** | 扩容 AutoDL 存储空间（TSV 文件含 base64 图像，很大） |
| **demo.py 权重不存在** | 修改 demo.py 使用 `polyformer_l_refcoco.pt`（没有 refcocog 版本） |
| **BERT 权重下载失败** | 手动下载到本地目录，修改代码使用 `/root/autodl-tmp/projects/polyformer/weights/bert-base-uncased` |
| **gdown Google Drive 限制** | 从本地上传 PolyFormer 权重（VSCode 拖拽上传） |
| **JPEGImages.zip 上传损坏** | VSCode 拖拽上传大文件易损坏，改用 `scp -P 端口` 命令上传 |
| **RefDIOR 缺少 Annotations.zip** | 从 [Google Drive](https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_) 补充下载 |

---

## 📝 下一步待办

### ✅ Phase 0 完成！

所有环境配置和验证已完成：
- ✅ 切换到有卡模式
- ✅ 生成微调数据 (`create_finetuning_data.py`)
- ✅ BERT 权重本地化
- ✅ Demo 端到端推理测试通过（45步，mask 36.97%）

### ✅ Phase 1 完成！

读代码定口径完成，关键发现已记录到 `outputs/refdior/fact_sheet.md`：
- ✅ 定位 task/model/criterion 注册点
- ✅ 定位 normalize→quantize 代码位置（`data/refcoco_dataset.py:111-133`）
- ✅ 定位 polygon 序列规则（`data/poly_utils.py`：顺时针+起点距原点最近）
- ✅ 定位 tokenizer 使用（⚠️ 当前没有设置 max_length=20）
- ✅ 定位损失项（L1 回归 + label smoothed CE，det_weight/cls_weight）

**关键发现汇总**：
- **num_bins = 64**（不是默认的 1000）
- **坐标流程**：[0,1] 归一化 → ×63 → floor/ceil（双线性插值）
- **polygon canonical 已实现**：is_clockwise + reorder_points
- **tokenizer 需要修改**：当前 `batch_encode_plus(padding="longest")` 没有 truncation

### 🔄 当前：Phase 2 进行中

**RefDIOR (DIOR-RSVG) 数据 Sniff 结果：**

| 项目 | 结果 |
|------|------|
| 数据格式 | XML (Pascal VOC 风格) |
| 图像数量 | 17,402 张 |
| 总样本数 | **38,320** ✅ |
| 图像尺寸 | 800×800 (固定) |
| Train | 26,991 (70.4%) |
| Val | 3,829 (10.0%) |
| Test | 7,500 (19.6%) |
| 类别数 | 20 (遥感目标类型) |
| **Mask/Segmentation** | ⚠️ **无！只有 bbox** |

**关键发现：**
- DIOR-RSVG 原始数据**只有 bbox + expression，没有 mask**
- XML 字段：filename, size, object(name, bndbox, description)
- `description` 字段即为 referring expression
- 将用 **bbox 作为矩形 polygon** 进行训练

**当前进度：**
- ✅ Annotations.zip 已解压
- ✅ convert_refdior.py 脚本已创建并上传
- 🔄 JPEGImages.zip 正在上传中 (5.0 GB)
- ⏳ 等待图像上传完成后运行转换

**下一步：**
1. 等待 JPEGImages.zip 上传完成
2. 解压图像：`unzip JPEGImages.zip`
3. 运行转换：`python tools/refdior/convert_refdior.py --root datasets/DIOR-RSVG --out-dir datasets/refdior/processed`
4. 可视化 sanity check

详见工程指南各 Phase 的详细操作步骤。

---

## 📚 参考信息

### PolyFormer 项目信息
- 论文：CVPR 2023
- GitHub：https://github.com/amazon-science/polygon-transformer
- 坐标范围：[0, 1] 归一化
- Num bins：64×64
- 图像尺寸：512×512
- Token types：0=坐标, 1=分隔符(SEP), 2=EOS

### 依赖版本（已验证）
```
torch==1.12.1+cu113
torchvision==0.13.1+cu113
pytorch_lightning==1.9.5
fairseq==1.0.0a0+69fc728
numpy==1.23.5
protobuf==3.20.0
tokenizers==0.13.3
pip==23.3.2
```

### 本地资源路径
```
D:\science\polyformer\polyformer_assets\
├── datasets\
│   ├── annotations\
│   ├── coco\
│   └── refcoco\
└── weights\
    ├── backbone\
    └── polyformer\
```

---

## 📞 继续对话时的快速恢复

如果需要在新对话中继续，请告诉 AI：

> "我在配置 PolyFormer + RefDIOR 项目，请查看 `docs/autodl_setup_guide.md` 和 `docs/polyformer_refdior_guide.md` 了解当前进度和详细工程指南。"

---

## 🔄 快速恢复命令

### 激活环境（每次连接必做）
```bash
cd /root/autodl-tmp/projects/polyformer
source venv/bin/activate
nvidia-smi  # 确认 GPU（有卡模式）
```

### 如果微调数据生成中断，继续运行
```bash
# 检查已生成的文件
ls -lh datasets/finetune/*/

# 如果需要重新运行（会覆盖）
python data/create_finetuning_data.py
```

### 运行 Demo（微调数据生成完成后）
```bash
# 修改权重路径（只需执行一次）
sed -i "s/polyformer_l_refcocog.pt/polyformer_l_refcoco.pt/g" demo.py

# 运行 demo
python demo.py
```

### 检查存储空间
```bash
df -h /root/autodl-tmp
du -sh /root/autodl-tmp/projects/polyformer/*
```
