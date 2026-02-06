# PolyFormer → RefDIOR 复现指南（最终可执行版）

> 📌 **入口文档**：[`docs/README.md`](./README.md) | **实时进度**：[`docs/autodl_setup_guide.md`](./autodl_setup_guide.md)

> **唯一总目标**：复现 **SeeFormer 论文中 "PolyFormer 在 RefDIOR 上"** 的两行结果（不是复现 SeeFormer 模型本身）。

> **工程边界**：只做 **数据 / 训练设定 / 推理口径 / 评估口径** 的严格对齐与诊断闭环；**不引入任何新网络模块**、不做 A+B 拼装。

---

## ✅ 验收 Gate（必须写死的通过条件）

> **Gate 通过=两行都满足误差容忍**（见下），否则必须按"Gate 失败排查树"逐项定位。

**Gate-1（RefDIOR RIS / Table I / PolyFormer）目标值：**

P@0.5=37.77, P@0.6=31.95, P@0.7=23.59, P@0.8=12.72, P@0.9=2.01, oIoU=60.39, mIoU=34.27, **Sum=202.70**。

**Gate-2（RefDIOR VG / Table III / PolyFormer）目标值：**

P@0.5=55.39, P@0.6=48.64, P@0.7=40.04, P@0.8=27.45, P@0.9=6.09, oIoU=71.22, mIoU=49.09, **Sum=297.92**。

### 误差容忍（严格但现实）

* 对 **P@{0.5..0.9}**：每项允许 **±0.5**（绝对百分点）
* 对 **oIoU / mIoU**：每项允许 **±0.3**
* 对 **Sum**：允许 **±1.0**

> 为什么这么定：P@ 是阈值计数，受离散化 / 微小几何误差影响更大；IoU 均值更平滑；Sum 是聚合指标，容忍更紧能防"某项偏太多但被其他项抵消"。

### Gate 失败排查树（必须按这个顺序，不要跳）

> 总原则：**先排"定义/口径错误"，再排"数据错误"，最后才排"模型训练不够好"**。因为口径错会让你永远不可能对齐论文表格。

1. **数据（RefDIOR→JSONL→Dataset）**

   * ✅ **先做数据集身份校验**：若 split 统计为 **26,991/3,829/7,500** 且找不到任何 mask/segmentation 文件，你拿到的是 **DIOR-RSVG（VG-only）**，不是论文 RefDIOR（box+mask）；此时请先补齐 mask 再继续跑 RIS。
   * split 是否严格 26,824/3,832/7,664？RefDIOR 官方划分是 7:1:2
   * mask 读取是否正确（二值/instance-id/COCO-RLE）？
   * resize 是否 **512×512 且同步**（image/mask/box/poly 同步）？
   * 坐标是否 **[0,1] normalize 并走 repo 内同一套 normalize→quantize**（你已确认的硬事实）？

2. **评估（指标实现）**

   * P@、oIoU、mIoU、Sum 定义是否严格一致？RRSECS 定义为 P@0.5..0.9 + oIoU + mIoU，Sum 为七项相加
   * oIoU 是否是 **全局 sum(inter)/sum(union)** 而非 per-sample mean？
   * 是否先做 **GT-vs-GT / polygon-fidelity / eval 自检三件套**（Phase 3 强制）？

3. **训练设定（Strict reproduction 是否真的严格）**

   * **512×512、无增强、max language length=20、50 epochs、batch size=32、seed=3407** 是否全部满足？
   * 优化器与 LR 策略是否对齐：AdamW、wd=0.01、lr=5e-5、polynomial decay、warmup=10%？

4. **推理口径（VG 必须双通道可证伪）**

   * **通道1**：用 PolyFormer 直接输出的 pred box 做 box IoU
   * **通道2**：pred polygon→rasterize 得到 pred mask→取 mask 的最小外接矩形（MBR）作为 pred box 做 IoU
   * SeeFormer 明确提示：部分结果"通过 predicted mask 的 MBR 得到"；RRSECS 也有同类脚注
   * **以能对齐 Table III（Gate-2）的一路为准**，另一条保留在日志里用于解释差异。

5. **随机性（最后查）**

   * seed 是否一致（3407）？
   * cudnn 是否 deterministic？混合精度/TF32 是否引入波动？
   * 多卡时 global batch 是否仍为 32（见 Phase 3）？

---

## 复现模式定义（必须严格区分）

### 1) Strict reproduction（本指南 Part 1 唯一默认）

目标：**严格复现 SeeFormer 表格中 PolyFormer 那一行**（即 RRSECS/CCFormer 对比方法口径）。

必须满足（写死）：

* **图像 512×512（统一 resize）、无数据增强**
* **max language length = 20**
* **训练 50 epochs、batch size=32、seed=3407**
* **优化器/LR**：AdamW wd=0.01 lr=5e-5 polynomial decay warmup 10%
* **RECS 方法**：超参同 RRSIS/RIS，仅 loss 不同

### 2) Ablation / 可选探索（只能在 Gate 通过后做，Part 2 才展开）

例如 SeeFormer 自己方法的：max text length=21、inverse_sqrt、lr=5e-4、loss ratio 1:0.1:0.005 等

> 这些不能当 baseline 默认，因为它们属于 **SeeFormer 自己方法的训练设定**，不是 RRSECS/CCFormer 统一对比设定。

---

## 已知硬事实（本文会据此组织流程，不再写成未知）

* **坐标监督**：PolyFormer repo 的坐标监督在 dataset 内是 **[0,1] 归一化**，并对 box & polygon 做 **normalize→quantize**（你已确认）。
* **训练入口**：`run_scripts/finetune/train_polyformer_b.sh` → `train.py`（fairseq 风格），通过 `--user-dir=polyformer_module` 注册 task/model/criterion（你已确认）。
* **RefDIOR mask 格式未知**：必须先 sniff，再分支解析（你已确认）。

### ⚠️ **重要区分（2026-02-05 新增硬事实）：DIOR-RSVG ≠ RefDIOR**

| 数据集 | Split (objects) | 内容 | 用途 |
|--------|-----------------|------|------|
| **DIOR-RSVG** | 26,991/3,829/7,500 (38,320) | bbox + expression，**无 mask** | VG-only |
| **RefDIOR (SeeFormer)** | 26,824/3,832/7,664 (38,320) | bbox + expression + **mask** | RIS + VG |

**关键区别**：
- 如果你下载的是 **DIOR-RSVG**（官方 split=26,991/3,829/7,500，XML 里 `segmented=0`，仅 bbox+expression），它是 **VG-only** 且 **不含 mask**。
- **无法复现本指南 Gate-1（RefDIOR-RIS）**，也无法训练 PolyFormer 的 polygon 监督。
- SeeFormer 论文明确说明：对于 "RSVG-type（输出 bbox）" 的 RIS 评测，是对预测 bbox 再跑 **SAM2 inference** 得到 pred mask——这隐含 **gt mask 必须存在**。

---

## Table of Contents

* [全局约定：目录结构与可追溯产物标准](#全局约定目录结构与可追溯产物标准)
* [审计与纠错（相对旧稿的关键修订）](#审计与纠错相对旧稿的关键修订)
* [Phase 0：环境与 Baseline 命令抽取](#phase-0环境与-baseline-命令抽取fairseq-user-dir-可用)
* [Phase 1：读代码定口径](#phase-1读代码定口径坐标序列文本截断损失推理输出)
* [Phase 2：RefDIOR 适配](#phase-2refdior-适配sniff转换-jsonl可视化与数据版本-hash)
* [Phase 3：训练与评估闭环](#phase-3训练与评估闭环strict-reproduction--双通道-vg--gate-检查)
* [Phase 4：RefDIOR 数据集画像与失败模式库](#phase-4refdior-数据集画像与失败模式库12-项)
* [Phase 5：实验矩阵 + Thinking Prompts](#phase-5实验矩阵--thinking-prompts只改数据训练评估)
* [附录 A：RefDIOR 中间格式 JSONL Schema](#附录-arefdior-中间格式-jsonl-schema固定)
* [附录 B：必须落地的工具脚本清单](#附录-b必须落地的工具脚本清单可直接复制)
* [附录 C：评估指标实现与自检三件套](#附录-c评估指标实现与自检三件套gt-vs-gt--fidelity--口径差异)

---

## 全局约定：目录结构与可追溯产物标准

> 目标：任何一次实验都能从 `outputs/refdior/<exp_id>/` **一键复现** + **可诊断** + **可做失败样本库**。

建议目录（可直接照抄）：

```text
polyformer/
  polyformer_module/                 # fairseq user-dir (原 repo)
  tools/
    refdior/
      extract_base_cmd.py            # 从 run_scripts/finetune/train_polyformer_b.sh 抽取 baseline cmd（消灭占位符）
      hash_data_version.py           # 原始数据版本 hash
      sniff_refdior.py               # sniff mask/ann 格式，决定分支
      convert_refdior.py             # RefDIOR -> JSONL（含 mask->polygon canonical + tight box + bad samples）
      unit_test_targets.py           # 打印 batch 的 dtype/range/shape（确认 normalize→quantize 生效口径）
      eval_refdior.py                # 推理 + 评估（含 VG 双通道）
      vis_refdior.py                 # best/worst 可视化样本库
      check_gate.py                  # Gate 自动对比（误差容忍）
      analyze_refdior.py             # Phase 4 数据集画像与失败模式分析
  datasets/
    refdior/
      raw_links/                     # 指向原始 RefDIOR（只读软链接/路径记录）
      processed/
        refdior_train.jsonl
        refdior_val.jsonl
        refdior_test.jsonl
        meta_stats.json              # 基础统计（Phase 2 级）
        bad_samples.jsonl            # 坏样本隔离（必须）
  outputs/
    env/
      conda_env.yml
      pip_freeze.txt
      sys_info.txt
    refdior/
      exp_<ID>/
        args.json                    # 训练/评估命令与关键超参快照（必须）
        data_version.json            # 原始数据 hash（必须）
        train.log
        ckpts/
          checkpoint_best.pt         # 以 val 指标挑 best（建议）
        eval/
          eval_val.json
          eval_test.json
        preds/
          preds_val.jsonl
          preds_test.jsonl
        vis/
          val_bestk/
          val_worstk/
          test_bestk/
          test_worstk/
        analysis/
          summary.json
          tables/
          figures/
          samples/
```

**为什么这么严：** 复现失败时，99% 的时间不是"模型不行"，而是"你丢了证据链"。`args.json + preds.jsonl + vis/` 能把"玄学"变成"可证伪"。

---

## 审计与纠错（相对旧稿的关键修订）

> 每条都包含：**问题**→**为什么会导致对齐失败**→**本指南如何修正**。

### A0. 旧稿缺少"验收 Gate"，导致你不知道自己到底在复现谁

* **问题**：旧稿没有把 SeeFormer 表格里 PolyFormer 的两行指标写成硬门槛。
* **后果**：你可能"跑通了训练/评估"，但永远不知道是否对齐 Table I/III（尤其 VG 口径差异会把你骗得很惨）。
* **修正**：本指南开头写死 Gate-1/Gate-2，并定义误差容忍与排查树（见最上方）。

### A1. "Strict reproduction vs Ablation" 边界在旧稿里被混了

* **问题**：旧稿把 SeeFormer 自己方法的训练设定（max text length=21、inverse_sqrt、lr=5e-4、loss ratio 1:0.1:0.005）混进 baseline。
* **后果**：你会把**不属于对比口径**的设定当成默认，从而永远对不齐 RRSECS/CCFormer 的 PolyFormer 行。
* **修正**：Part 1 默认只跑 Strict reproduction（RRSECS 统一设定），SeeFormer 自己设定只允许作为 Gate 通过后的 ablation（见"复现模式定义"）。

### A2. VG 评估口径旧稿只写了一个通道，存在"脚注陷阱"

* **问题**：旧稿默认"用 pred box 做 VG"，但 RRSECS/SeeFormer 明确存在"由 predicted mask 的 MBR 得到 box 结果"的脚注。
* **后果**：你可能训练/推理都正确，但 VG 永远差一截，因为对比方法的 box 是 **mask→MBR** 而你用的是 model 的 raw box（或反过来）。
* **修正**：Phase 3 强制双通道 VG（通道1=pred box；通道2=polygon→mask→MBR），以能对齐 Gate-2 的通道为准，并记录另一通道用于解释差异。

### A3. "<TASK>/<ARCH>/<CRITERION> 占位符"会让文档不可执行

* **问题**：旧稿命令里存在 `<TASK>/<ARCH>/<CRITERION>` 占位符。
* **后果**：你一旦填错（哪怕只错一个字符串），fairseq 直接找不到注册项，训练无法启动。
* **修正**：Phase 0 提供 **自动抽取**脚本 `extract_base_cmd.py`：直接从 `run_scripts/finetune/train_polyformer_b.sh` 抽取并固化为可复用的 baseline 命令；后续训练只改数据入口参数（满足你的硬要求）。

### A4. polygon 序列 canonical 化在旧稿里"提到但不强制"

* **问题**：旧稿提到了顺时针+固定起点，但没有把它写成"必须、可验证"。
* **后果**：对 AR 序列建模来说，同一 mask 对应多个等价序列（起点旋转/顺逆时针），会变成多解映射 → loss 震荡/Overfit-50 失败。PolyFormer 明确要求顶点顺时针且起点为最接近原点的顶点。
* **修正**：Phase 2 把 canonical 化写成硬约束，并用 polygon-fidelity（rasterize 回 mask 的 IoU）做上限自检。

### A5. RRSECS 的训练统一设定需要"落地成可检查项"

* **问题**：旧稿对"512×512、无增强、max len=20、batch size 32、seed 3407、AdamW lr 5e-5 poly decay"没有写成"日志可核对"的 checklist。
* **后果**：你可能以为自己对齐了，但实际用的是别的 scheduler/seed/增强，导致指标漂移而不自知。
* **修正**：Phase 3 把 strict 设定写成强制参数，并要求在 `train.log` 与 `args.json` 中能被核对。

---

# Phase 0：环境与 Baseline 命令抽取（fairseq user-dir 可用）

## 0.1 目的

* 让 repo 在本机 **可 import / 可启动 fairseq train**（哪怕不训练）。
* 把 `run_scripts/finetune/train_polyformer_b.sh` 的真实 `--task/--arch/--criterion/...` **抽取并固化**，从根上消灭占位符错误。
* 产出环境三件套（conda_env.yml / pip_freeze.txt / sys_info.txt），保证"同环境可复跑"。

## 0.2 操作

### 0.2.1 环境三件套（最稳默认）

```bash
cd polyformer
git rev-parse HEAD

conda create -n polyformer_refdior python=3.10 -y
conda activate polyformer_refdior

# 安装 PyTorch（按你 CUDA 改）
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# repo 依赖
pip install -r requirements.txt

# fairseq（如果 requirements 已带就会跳过）
pip install fairseq

mkdir -p outputs/env
python -c "import sys,torch; print('py',sys.version); print('torch',torch.__version__); print('cuda',torch.version.cuda,'avail',torch.cuda.is_available())" | tee outputs/env/torch_info.txt
pip freeze > outputs/env/pip_freeze.txt
conda env export > outputs/env/conda_env.yml
nvidia-smi > outputs/env/sys_info.txt
```

### 0.2.2 验证 fairseq 入口 + user-dir 注册可用

```bash
python train.py --help | head -n 40
python train.py --user-dir polyformer_module --help | head -n 60
```

### 0.2.3 自动抽取 baseline 命令（核心：消灭 <TASK>/<ARCH>/<CRITERION>）

把下面脚本保存为 `tools/refdior/extract_base_cmd.py`（完整代码见附录 B），然后运行：

```bash
mkdir -p outputs/refdior
python tools/refdior/extract_base_cmd.py \
  --sh run_scripts/finetune/train_polyformer_b.sh \
  --out outputs/refdior/refdior_base_cmd.sh

# 输出会是一个可执行脚本，里面只有一条"python train.py ..."命令
sed -n '1,200p' outputs/refdior/refdior_base_cmd.sh
```

> **为什么必须抽取而不是手抄**：fairseq 的注册名往往是短字符串，手抄最容易错；错一次就直接"找不到 task/model/criterion"，属于低级但最耗时间的坑。

## 0.3 预期输出

* `outputs/env/{conda_env.yml,pip_freeze.txt,sys_info.txt,torch_info.txt}`
* `outputs/refdior/refdior_base_cmd.sh`（baseline 训练命令已固化、可直接执行）

## 0.4 验证（Stop condition）

* ✅ `python train.py --user-dir polyformer_module --help` 不报错
* ✅ `outputs/refdior/refdior_base_cmd.sh` 中能看到 **明确的 `--task/--arch/--criterion`**（不出现任何占位符）
* ✅ 把 `outputs/refdior/refdior_base_cmd.sh` 复制到任意机器，只要环境一致就能启动 argparse（不要求有数据）

## 0.5 失败排查（Failure tree）

按优先级：

1. **fairseq 与 python 版本冲突**：优先 python=3.10（3.11/3.12 常见依赖不兼容）
2. **CUDA/torch 不匹配**：`nvidia-smi` 与 `torch.version.cuda` 不一致
3. **依赖编译失败**（opencv/shapely/pycocotools）：先装系统依赖再 pip；必要时锁版本
4. `--user-dir` 报 import error：检查 `polyformer_module/__init__.py` 是否存在、路径是否正确

## 0.6 交付物

* `outputs/env/*`
* `outputs/refdior/refdior_base_cmd.sh`（后续 Phase 3 训练脚本会直接调用它）

---

# Phase 1：读代码定口径（坐标/序列/文本截断/损失/推理输出）

> Phase 1 的目标不是"学术理解"，而是把复现里最容易悄悄跑偏的东西，变成 **可核对、可证伪** 的事实表（Fact Sheet）。

## 1.1 目的

* 把"实现真相"写死：

  1. dataset 输出的坐标到底是 float 还是 int？范围是否在 [0,1]？quantize 的输出是什么？
  2. target sequence 的 token/type/坐标占位规则是什么？（EOS/SEP 是否需要 coord padding）
  3. tokenizer 的 max length=20 的含义是什么？（是否包含 special tokens）
  4. PolyFormer 输出中"pred box"和"pred polygon"分别从哪儿拿？（VG 双通道需要）

> 机制假设：**对齐失败通常不是"训练不够"，而是"你在训练/评估的对象跟论文不是一个对象"**。Phase 1 的工作就是把"对象"钉死。

## 1.2 操作

### 1.2.1 定位 task/model/criterion 注册点（不要猜路径）

```bash
rg -n "@register_task|register_task\\(" polyformer_module
rg -n "@register_model|register_model\\(" polyformer_module
rg -n "@register_criterion|register_criterion\\(" polyformer_module
```

### 1.2.2 定位 RefCOCO dataset 里 normalize→quantize 的代码位置（你已确认存在）

```bash
rg -n "refcoco_dataset|quantize|normalize" polyformer_module .
```

### 1.2.3 定位 polygon 序列规则（顺时针 + 起点离原点最近）

PolyFormer 论文明确规定顶点顺时针，起点是离图像原点最近的顶点，多边形按起点到原点距离排序。
你要在代码里找到对应实现或等价逻辑（否则你需要在 RefDIOR conversion 里强制实现）。

```bash
rg -n "clockwise|origin|start point|canonical|sort.*origin|signed area" polyformer_module .
```

### 1.2.4 定位 tokenizer 与 max_text_len=20 的真实含义

RRSECS 对比口径要求 max language length=20。
但"20"是 **token 数**还是 **word 数**，以及是否包含 `[CLS]/[SEP]`，必须以代码为准。

```bash
rg -n "max_.*text|max_.*lang|max_length|tokenizer|encode_plus|truncation|padding" polyformer_module .
```

### 1.2.5 定位损失项与权重（仅做"确认"，strict reproduction 不随意改）

PolyFormer 论文训练细节里明确有 Lcoord/Lcls/Ltype 等，并用加权求和；同时也给出一些实现细节（如 label smoothing）。
你需要确认 repo 中 cls/type/coord 的 loss 各自是什么、权重在哪里设置。

```bash
rg -n "loss|lambda|weight|Lcls|Ltype|Lcoord|label_smoothing" polyformer_module .
```

## 1.3 预期输出（Fact Sheet：必须写入文件）

把下面模板保存为 `outputs/refdior/fact_sheet.md` 并填满（所有字段都要有证据链：文件路径+行号/日志片段）：

```markdown
# Fact Sheet (PolyFormer repo truth)

## fairseq entry
- source sh: run_scripts/finetune/train_polyformer_b.sh
- extracted base cmd: outputs/refdior/refdior_base_cmd.sh
- user-dir: polyformer_module
- task: refcoco (tasks/refcoco.py:70)
- arch: polyformer_b (models/polyformer/polyformer.py:203)
- criterion: adjust_label_smoothed_cross_entropy (criterions/label_smoothed_cross_entropy.py:145-146)

## encoders (对齐 CCFormer/RRSECS 统一设定的核对项)
- vision backbone: (Swin-Tiny? 其它? 以代码/args 为准)
- text backbone: (BERT-Base 12-layer 768? 以代码/args 为准)

## coordinate supervision (你的硬事实 + 运行时打印确认)
- coord range: [0,1] normalized
- where normalized: <file:line>
- quantize function: <file:line>
- quantize output: (dtype/int bins? float? what range?)

## target sequence format
- special tokens: BOS/EOS/SEP ids
- type tokens: what categories? how encoded?
- do EOS/SEP require coord placeholders? (yes/no + evidence)
- max polygon vertices (Nmax): where set?

## text pipeline
- max_text_len default: ...
- is it bert max_length? includes special tokens? ...
- truncation ratio under 20: ... (Phase 2 会统计)

## inference outputs (为了 VG 双通道)
- how to get pred_box from model output
- how to get pred_polygon from model output
- any postprocess clamp/denorm?
```

## 1.4 验证（Stop condition）

你必须能在本机跑通下面的"静态 + 动态最小验证"。

### 1.4.1 静态验证

* ✅ 你能在 `fact_sheet.md` 中写清楚 task/arch/criterion（来自 `outputs/refdior/refdior_base_cmd.sh`）
* ✅ 你能指出 normalize→quantize 的代码位置（文件+行号）
* ✅ 你能指出 tokenizer max_length 的调用位置（文件+行号）

### 1.4.2 动态验证（推迟到 Phase 2 后执行）

因为 RefDIOR dataset 还没接入，所以动态验证（打印 batch 的 dtype/range）在 Phase 2 的 `unit_test_targets.py` 里做（这不是拖延，而是依赖关系：你得先能 load RefDIOR batch）。

## 1.5 失败排查（Failure tree）

1. 找不到 register_task：检查 `--user-dir` 是否正确、模块是否被 import（Phase 0 已验证 help）
2. 找到多套 dataset/transform：优先沿用 baseline cmd 对应的那套（不要同时改两套）
3. max_text_len 有多个：以 **task.add_args** 的默认和 baseline cmd 的显式参数为准

## 1.6 交付物

* `outputs/refdior/fact_sheet.md`（Phase 2/3 的唯一依据）
* `outputs/refdior/rg_dump_phase1.txt`（把 Phase 1 的关键 `rg` 输出重定向保存，便于审计）

---

# Phase 2：RefDIOR 适配（sniff→转换 JSONL→可视化与数据版本 hash）

> 你已确认 RefDIOR mask 格式未知，所以 Phase 2 必须按 **sniff→分支** 设计。
> 同时，RefDIOR 官方规模与 split 期望为 38,320 quadruplets，26,824/3,832/7,664——这会成为你转换后统计的硬校验。

## 2.1 目的

* 把 RefDIOR 原始数据固化为 **稳定中间格式 JSONL**：

  * 便于 diff、坏样本隔离、并行处理、以及后续 preds.jsonl 回写分析

* 把 mask→polygon 的信息损失"显式化"：用 polygon-fidelity 作为标签上限自检
* 锁定数据版本：同一份数据在不同机器/不同时间 hash 一致

## 2.2 操作

### 2.2.1 数据版本 hash（必须先做）

```bash
export REFDIOR_ROOT=/path/to/RefDIOR

python tools/refdior/hash_data_version.py \
  --root "$REFDIOR_ROOT" \
  --out outputs/refdior/exp_refdior_data/data_version.json
```

> 为什么先 hash：你一旦开始转数据/软链接/拷贝，很容易混进临时文件或漏文件。先 hash 才能证明"输入没变"。

### 2.2.2 sniff RefDIOR（强制）

```bash
python tools/refdior/sniff_refdior.py \
  --root "$REFDIOR_ROOT" \
  --n 50 \
  --out outputs/refdior/refdior_sniff.json
```

> **⚠️ 数据集身份校验（硬门槛）**
> 如果 sniff/统计显示：**无任何 mask/segmentation 文件** + split 为 **26,991/3,829/7,500**，则你当前数据是 **DIOR-RSVG（VG-only）**，会直接阻断 Gate-1（RefDIOR-RIS）。请先获取/生成 RefDIOR 的 gt mask 再进入 convert→train。

sniff 输出必须回答：

* mask 存储形态：

  * **A：PNG 二值**（0/1 或 0/255）
  * **B：PNG instance-id**（多值，需要目标 id）
  * **C：COCO-style segmentation**（polygon/RLE）
  * **D：无 mask**（DIOR-RSVG）→ 无法用于 RIS
  * 或混合（必须写清混合规则）

* annotation 存储形态：COCO dict? list-of-dict? jsonl?

#### 2.2.2.1 本轮 sniff 结论记录（2026-02-05）

* ✅ 你从 Google Drive 下载的数据集符合 **DIOR-RSVG** 的官方结构（`JPEGImages/` + `Annotations/*.xml` + `train/val/test.txt`），且 **不包含任何 mask/segmentation 标注**。
* ✅ DIOR-RSVG split（object-level）为：train=26,991 / val=3,829 / test=7,500（total=38,320）。
* ⚠️ 因此：该数据只能用于 **VG（bbox IoU）**；**不能**用于复现 Gate-1（RefDIOR RIS）与 PolyFormer 的 polygon 监督训练。
* 🔴 **下一步**：必须获取/生成 **RefDIOR 的 gt mask**（论文期望 RefDIOR split=26,824/3,832/7,664）后再进入 convert→train。

### 2.2.3 转换为 JSONL（按 sniff 自动分支 + 失败即给出证据）

```bash
mkdir -p datasets/refdior/processed

python tools/refdior/convert_refdior.py \
  --root "$REFDIOR_ROOT" \
  --split train --out datasets/refdior/processed/refdior_train.jsonl \
  --meta-out datasets/refdior/processed/meta_stats.json \
  --bad-out  datasets/refdior/processed/bad_samples.jsonl \
  --cc-policy largest \
  --nmax 80

python tools/refdior/convert_refdior.py --root "$REFDIOR_ROOT" --split val  --out datasets/refdior/processed/refdior_val.jsonl  --meta-out datasets/refdior/processed/meta_stats.json --bad-out datasets/refdior/processed/bad_samples.jsonl --cc-policy largest --nmax 80
python tools/refdior/convert_refdior.py --root "$REFDIOR_ROOT" --split test --out datasets/refdior/processed/refdior_test.jsonl --meta-out datasets/refdior/processed/meta_stats.json --bad-out datasets/refdior/processed/bad_samples.jsonl --cc-policy largest --nmax 80
```

> **强默认策略（保证可执行）**：`convert_refdior.py` 提供"自动字段探测 + COCO/list-of-dict 两种解析器"，探测失败时会把"看到的 keys"写入错误信息并退出（你不需要我问你问题，你只要照报错提示补映射即可）。
> 这满足"没有额外信息也能跑、但失败时给最强证据"的要求。

### 2.2.4 训练前可视化 sanity（至少 20 张）

```bash
python tools/refdior/vis_refdior.py \
  --jsonl datasets/refdior/processed/refdior_val.jsonl \
  --outdir outputs/refdior/vis_sanity_val \
  --n 20
```

## 2.3 预期输出

* `datasets/refdior/processed/refdior_{train,val,test}.jsonl`
* `datasets/refdior/processed/meta_stats.json`
* `datasets/refdior/processed/bad_samples.jsonl`
* `outputs/refdior/vis_sanity_val/*.png`（至少 20 张）
* `outputs/refdior/exp_refdior_data/data_version.json`

## 2.4 验证（Stop condition）

你必须同时满足：

0. **数据集身份校验（最优先）**
   * sniff 必须确认 **gt mask 存在且可读**
   * 若无 mask 且 split=26,991/3,829/7,500 → 这是 DIOR-RSVG（VG-only），请先获取/生成 RefDIOR 的 gt mask

1. **split 规模硬校验**（不满足直接回 Phase 2 查解析器）

   * train=26,824 / val=3,832 / test=7,664（RefDIOR 期望）
   * ⚠️ 如果你看到 26,991/3,829/7,500，这是 DIOR-RSVG，不是 RefDIOR

2. **可视化硬校验**

   * 20 张 overlay 中：mask 与 polygon 边界基本贴合（允许少量离散化误差）
   * raw_box 与 tight_box 的关系可解释（不要求一致，但不能离谱）

3. **基础标签上限（polygon-fidelity）**

   * 在 `meta_stats.json` 中输出 fidelity IoU 的均值与分位数（P50/P90/P99），且 worst-case 能通过可视化解释原因（细长、锯齿、洞、多连通域等）

## 2.5 失败排查（Failure tree）

按顺序（不要跳）：

0. **数据集身份错误（最优先）**：sniff 发现无 mask，且 split=26,991/3,829/7,500 → 这是 DIOR-RSVG（VG-only），请先获取/生成 RefDIOR 的 gt mask，再回到 Phase 2
1. **解析器选错**：sniff 指出是 COCO，但你走了 png-mask；或反之
2. **mask/图像尺寸不一致**：先保证读到的 mask 与 image 尺寸一致（不一致要找到官方 size 字段或 resize 规则）
3. **instance-id 未选对目标**：unique values 很多，但 annotation 没有 instance_id → 你无法自动对齐表达式目标（必须回到 annotation 字段映射）
4. **polygon 退化**：点数过少/重复点/自交；先在 conversion 阶段过滤并记录 bad_samples
5. **洞/多连通域**：baseline 默认 drop hole + largest cc；如果 fidelity 很差，先统计 hole/cc 占比，再决定是否启用 `cc-policy=multi`（但这是 Gate 之后的 ablation，Part 2 再做）

## 2.6 交付物

* RefDIOR→JSONL 的三份 split 文件
* `meta_stats.json` 与 `bad_samples.jsonl`
* `vis_sanity_val/`（训练前"证据墙"）

---

# Phase 3：训练与评估闭环（Strict reproduction → 双通道 VG → Gate 检查）

> Phase 3 的结束标志不是"能训练"，而是：
> **你能在 test 上跑出 Gate-1 与 Gate-2（误差容忍内）**。

## 3.1 目的

* 把 RefDIOR JSONL 接入 fairseq task（通过 `--user-dir=polyformer_module`）
* 用 **RRSECS 统一设定**训练 PolyFormer（Strict reproduction）
* 评估输出必须包含：

  * RIS（mask IoU）7 指标 + Sum
  * VG（box IoU）7 指标 + Sum（双通道并行）
  * preds.jsonl（可复盘）
  * best/worst 可视化样本库
  * GT-vs-GT / polygon-fidelity / eval 口径自检三件套

## 3.2 操作

### 3.2.1 接入 RefDIOR dataset：最小改动原则（必须）

**原则**：不改 model/criterion 的核心逻辑，只新增/扩展 dataset loader，让它能从 JSONL 读到 `image/expr/box/mask/polygons`。

你需要做的事（按顺序）：

1. 在 `polyformer_module` 中找到当前 task 的 `load_dataset()` 入口（Phase 1 已定位）。
2. 增加 CLI 参数（在 task 的 `add_args` 里）：

   * `--refdior-train-jsonl`
   * `--refdior-val-jsonl`
   * `--refdior-test-jsonl`
   * `--refdior-img-root`（可选：当 JSONL 存相对路径时需要）

3. 在 `load_dataset(split, ...)` 中：当 split=train/valid/test 时分别读取对应 JSONL，构造 Dataset。
4. **关键**：Dataset 内部必须复用你 repo 里 RefCOCO 的 **normalize→quantize** 逻辑（你已确认事实），不要另起炉灶。
5. **严格复现设定**：在 Dataset transform 中确保

   * resize 到 512×512
   * 无随机增强（random flip/crop/color jitter 都必须关）

> 为什么必须"复用 normalize→quantize"：坐标链路是最脆弱的地方。你多写一行不同的 clamp/round，P@0.8/0.9 就会崩。

### 3.2.2 动态口径验证：打印 batch 的 dtype/range（强制）

在接入完 RefDIOR dataset 后立刻跑：

```bash
python tools/refdior/unit_test_targets.py \
  --user-dir polyformer_module \
  --base-cmd outputs/refdior/refdior_base_cmd.sh \
  --refdior-jsonl datasets/refdior/processed/refdior_train.jsonl \
  --n 2 \
  --out outputs/refdior/unit_test_targets.json
```

你要在输出里确认：

* coords 是否在 [0,1]（或被 clamp 到 [0,1]）
* quantize 后的 dtype/range 是否与训练期望一致
* EOS/SEP 等 token 是否带 coord placeholder（如果需要）

### 3.2.3 Strict reproduction 训练脚本（只改数据入口参数 + 覆盖 RRSECS 超参）

新建 `run_strict_refdior.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

export REFDIOR_TRAIN_JSONL="datasets/refdior/processed/refdior_train.jsonl"
export REFDIOR_VAL_JSONL="datasets/refdior/processed/refdior_val.jsonl"
export REFDIOR_TEST_JSONL="datasets/refdior/processed/refdior_test.jsonl"

EXP_ID="exp_strict_polyformer_refdior_seed3407"
OUT_DIR="outputs/refdior/${EXP_ID}"
mkdir -p "${OUT_DIR}/ckpts" "${OUT_DIR}/eval" "${OUT_DIR}/preds" "${OUT_DIR}/vis"

# 记录 args.json（训练前就落盘，防止中途断电丢命令）
python - <<'PY'
import json, os, subprocess, time
exp=os.environ["EXP_ID"]; out=os.environ["OUT_DIR"]
cmd=open("outputs/refdior/refdior_base_cmd.sh","r",encoding="utf-8").read().strip()
meta={
  "exp_id": exp,
  "time": time.strftime("%Y-%m-%d %H:%M:%S"),
  "git_rev": subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip(),
  "base_cmd_path": "outputs/refdior/refdior_base_cmd.sh",
  "strict_overrides": {
    "img_resize": "512x512",
    "no_augmentation": True,
    "max_lang_len": 20,
    "epochs": 50,
    "global_batch_size": 32,
    "seed": 3407,
    "optimizer": "adamw",
    "lr": 5e-5,
    "weight_decay": 0.01,
    "lr_scheduler": "polynomial_decay",
    "warmup_ratio": 0.1
  }
}
os.makedirs(out, exist_ok=True)
json.dump(meta, open(os.path.join(out,"args.json"),"w",encoding="utf-8"), indent=2, ensure_ascii=False)
print("[args.json] wrote", os.path.join(out,"args.json"))
PY

# 说明：
# 1) 先执行 baseline cmd（里面已包含 task/arch/criterion 等"不可乱动"的参数）
# 2) 在末尾追加：RefDIOR 数据入口参数 + RRSECS strict 覆盖参数（重复的 arg 以后者为准）
BASE_CMD="$(cat outputs/refdior/refdior_base_cmd.sh)"

# 计算 global batch size=32：
# fairseq: global = num_gpus * batch_size_per_gpu * update_freq
# 你需要根据机器改下面两项：BS_PER_GPU 与 UPDATE_FREQ，让 global=32
NUM_GPUS="${NUM_GPUS:-1}"
BS_PER_GPU="${BS_PER_GPU:-8}"
UPDATE_FREQ="${UPDATE_FREQ:-4}"

# (示例) NUM_GPUS=1, BS_PER_GPU=8, UPDATE_FREQ=4 => global=32

eval ${BASE_CMD} \
  --refdior-train-jsonl "${REFDIOR_TRAIN_JSONL}" \
  --refdior-val-jsonl "${REFDIOR_VAL_JSONL}" \
  --refdior-test-jsonl "${REFDIOR_TEST_JSONL}" \
  --save-dir "${OUT_DIR}/ckpts" \
  --seed 3407 \
  --max-epoch 50 \
  --max-text-len 20 \
  --batch-size "${BS_PER_GPU}" \
  --update-freq "${UPDATE_FREQ}" \
  --optimizer adamw \
  --lr 5e-5 \
  --weight-decay 0.01 \
  --lr-scheduler polynomial_decay \
  --warmup-ratio 0.1 \
  --no-augment \
  2>&1 | tee "${OUT_DIR}/train.log"
```

> 注意：`--max-text-len/--no-augment` 的参数名可能是你 repo 自定义的。
> **解决方式不是改文档**，而是按 Phase 1 的 Fact Sheet 把真实参数名替换进去（这就是为什么 Phase 1 必须做）。
> 但"strict 覆盖项"本身必须满足 RRSECS 统一设定。

### 3.2.4 评估（必须：RIS + VG 双通道 + 产出 preds/vis）

训练完成后，用 best checkpoint 做评估（RRSECS 通常选 val 最优再报 test；这是对齐表格的常见做法）。

```bash
CKPT="outputs/refdior/exp_strict_polyformer_refdior_seed3407/ckpts/checkpoint_best.pt"

python tools/refdior/eval_refdior.py \
  --user-dir polyformer_module \
  --base-cmd outputs/refdior/refdior_base_cmd.sh \
  --checkpoint "${CKPT}" \
  --split val \
  --jsonl datasets/refdior/processed/refdior_val.jsonl \
  --out-eval outputs/refdior/exp_strict_polyformer_refdior_seed3407/eval/eval_val.json \
  --out-preds outputs/refdior/exp_strict_polyformer_refdior_seed3407/preds/preds_val.jsonl \
  --vg-mode both

python tools/refdior/eval_refdior.py \
  --user-dir polyformer_module \
  --base-cmd outputs/refdior/refdior_base_cmd.sh \
  --checkpoint "${CKPT}" \
  --split test \
  --jsonl datasets/refdior/processed/refdior_test.jsonl \
  --out-eval outputs/refdior/exp_strict_polyformer_refdior_seed3407/eval/eval_test.json \
  --out-preds outputs/refdior/exp_strict_polyformer_refdior_seed3407/preds/preds_test.jsonl \
  --vg-mode both
```

然后生成 best/worst 可视化库：

```bash
python tools/refdior/vis_refdior.py \
  --jsonl datasets/refdior/processed/refdior_test.jsonl \
  --preds outputs/refdior/exp_strict_polyformer_refdior_seed3407/preds/preds_test.jsonl \
  --outdir outputs/refdior/exp_strict_polyformer_refdior_seed3407/vis \
  --topk 50
```

### 3.2.5 Gate 自动检查（最后一步）

```bash
python tools/refdior/check_gate.py \
  --eval outputs/refdior/exp_strict_polyformer_refdior_seed3407/eval/eval_test.json
```

`check_gate.py` 会打印：

* RIS 是否过 Gate-1（逐项 delta）
* VG 通道1/通道2 哪个更接近 Gate-2（逐项 delta）
* 最终 PASS/FAIL

## 3.3 预期输出

* `outputs/refdior/exp_<ID>/args.json`
* `outputs/refdior/exp_<ID>/train.log`
* `outputs/refdior/exp_<ID>/ckpts/checkpoint_best.pt`
* `outputs/refdior/exp_<ID>/eval/eval_{val,test}.json`
* `outputs/refdior/exp_<ID>/preds/preds_{val,test}.jsonl`
* `outputs/refdior/exp_<ID>/vis/{test_bestk,test_worstk,...}/`

## 3.4 验证（Stop condition）

你必须满足三层 Stop condition（由低到高）：

### Stop-1：链路可用（训练/推理不崩）

* ✅ `train.log` 中 loss 能下降（不要求很快收敛）
* ✅ `eval_refdior.py` 输出指标不是 NaN/inf
* ✅ `preds_test.jsonl` 行数等于 test 样本数

### Stop-2：口径自检三件套全过（否则禁止看 Gate）

* ✅ **GT-vs-GT**：用 GT 作为 pred，IoU≈1（见附录 C）
* ✅ **polygon-fidelity**：`IoU(rasterize(gt_polygons), gt_mask)` 分布合理
* ✅ **VG 双通道都能算**，并能解释差异来源（pred box vs pred mask MBR）

### Stop-3：Gate 通过

* ✅ Gate-1（RIS）通过误差容忍
* ✅ Gate-2（VG）至少有一个通道通过误差容忍（并在 `eval_test.json` 中记录"命中通道"）

## 3.5 失败排查（Failure tree）

> 这里给你"最省时间"的检查顺序：每一步都能快速证伪一个大类问题。

1. **eval 自检不过**（GT-vs-GT 不到 1）
   → 直接说明你的 IoU 口径 / rasterize / resize 同步有错。先别动训练。

2. **polygon-fidelity 很低（P50 都很差）**
   → conversion 上限太低（mask→polygon 丢信息/点太少/洞/多连通域）。
   最小证伪：把 `nmax` 调大（80→120），看 fidelity 是否显著上升（这是数据处理层 ablation，不改模型）。

3. **RIS 对齐但 VG 不对齐**
   → 优先查 VG 双通道：

   * 若通道2（mask→MBR）更接近 Gate-2，说明表格可能采用 MBR 口径；反之则用 pred box。
   * 其次查 gt box source 是否使用 raw box（不要偷换 tight box）。

4. **VG 对齐但 RIS 不对齐**
   → 大概率是 polygon decode / canonical / rasterize 或坐标尺度问题。
   最小证伪：在 `preds_test.jsonl` 抽 20 个 worst，肉眼看 polygon 是否整体偏移/翻转/缩放错误。

5. **两者都差很多（>5 点）**
   → 回到 strict reproduction checklist：

   * 512×512、无增强、max len=20、seed=3407、lr=5e-5 poly decay、batch=32 是否全满足
   * global batch 是否真的 32（多卡别搞错 update_freq）

## 3.6 交付物

* 训练：`args.json/train.log/ckpts/`
* 评估：`eval_{val,test}.json`（含 VG 双通道结果 + 明确选用通道）
* 可诊断：`preds_{val,test}.jsonl` + `vis/` best/worst 样本库
* Gate：`check_gate.py` 输出日志（建议保存到 `outputs/refdior/exp_<ID>/gate_check.txt`）

---

# Phase 4：RefDIOR 数据集画像与失败模式库（≥12 项）

> 目标：在 **Phase 3 已经跑通训练/评估闭环**（尤其是 eval 口径自检三件套通过）的前提下，把 RefDIOR 当成"工程对象"做**可复现、可证伪、可定位根因**的系统画像：
>
> 1. 解释"为什么 PolyFormer 在 RefDIOR 上是这个分数"；
> 2. 把失败样本变成可检索的"失败模式库"；
> 3. 为 Phase 5 的 ablation（只改数据/训练/评估）提供**最小实验设计**。
>
> ⚠️ 强约束：本 Phase 只做诊断与统计，不引入任何新模型结构。

---

## 4.0 Stop condition（通过即进入 Phase 5）

✅ 你必须同时满足：

1. `outputs/refdior/exp_<ANALYSIS_ID>/analysis/` 目录完整生成（见 4.6），且每个图/表都能对上对应的统计解释。
2. **≥12 项诊断项**全部产出（图+表+样本），且每项都给出：

   * 为什么重要（机制假设）
   * 会怎样扭曲训练/评估
   * 最小可证伪实验（Phase 5 会用）

3. `samples/` 下有 **best-k / worst-k** 可视化库，并且 `tags.csv` 已建立最小标签体系（见 4.5）。
4. `mask→polygon fidelity` 的分位数统计已产出，并能解释 **worst-case** 来自哪里（洞、多连通域、细长目标、锯齿边等）。

---

## 4.0 Failure tree（不通过先查谁）

按顺序排查（不要跳）：

1. **输入侧（JSONL/文件）**

* `jsonl` 路径错 / image/mask 读不到 / `img_w,img_h` 与真实尺寸不一致
* `mask` 非二值但你当二值处理
* `polygons` 点序没 canonicalize（顺时针+固定起点）

2. **转换上限（fidelity）**

* fidelity 低 → 训练再对齐也很难到论文分数
* 先修 conversion（Nmax、cc_policy、hole_policy、rasterize）再继续

3. **评估口径**

* GT-vs-GT 不为 1
* RIS/VG 的 IoU 模式选错（mask IoU vs box IoU）
* VG 两通道（pred box vs mask→MBR）未分开评估导致误判

4. **训练/推理设定**

* 严格复现（RRSECS）与探索设定（SeeFormer风格）混用
* batch size / seed / lr schedule 与 RRSECS 设定不一致

---

## 4.1 目的（本 Phase 你到底要"回答什么问题"）

你最终要用数据回答这些"硬问题"（每个都要能落到统计或样本证据）：

1. **RefDIOR 的难点主要来自哪里？**（尺度？形状复杂？方向性？文本歧义？标注噪声？）
2. **你 pipeline 的上限被 conversion 锁死了吗？**（mask→polygon fidelity 是否已经把上限降到 < 0.95？）
3. **为什么 micro target 下 mIoU 更关键？**

   * oIoU（全局累计）更偏向大目标；mIoU（逐样本平均）更"公平"。
   * 类似现象在遥感指称分割里也常见：累计型 IoU 指标会对大目标偏置，导致分析重点更应放在"均值型 IoU"。

4. **VG（box）到底应该怎么对齐？**（pred box vs mask→MBR 两通道到底哪个对齐 Table III？）
5. **失败样本长什么样？能否被系统归类？**（形成可复用的失败模式库）

---

## 4.2 操作（命令/脚本）

> 本 Phase 建议拆成两个层次：
>
> * **Phase 4-A：数据集画像（只用 GT + JSONL）**：不用等模型训练，Phase 2 后就能做。
> * **Phase 4-B：失败模式库（需要 preds.jsonl）**：Phase 3 完成后再做。

---

### 4.2.1 依赖与目录准备

```bash
# 建议把 analysis 输出与训练输出解耦，单独开一个 analysis exp
ANALYSIS_ID="exp_refdior_analysis_v1"
ANALYSIS_DIR="outputs/refdior/${ANALYSIS_ID}"
mkdir -p "${ANALYSIS_DIR}/analysis"

# 确保你有基础依赖
python -c "import numpy as np; import cv2; import matplotlib; print('deps ok')"
```

---

### 4.2.2 新增脚本：`tools/refdior/analyze_refdior.py`（可直接运行）

> 该脚本必须做到：
>
> * 输入：`--jsonl`（train/val/test 任意）
> * 可选输入：`--preds-jsonl`（Phase 3 输出）
> * 输出：图/表/样本库/summary.json
> * 不依赖 repo 内部 model，实现"分析与训练解耦"。

运行（数据画像，GT-only）：

```bash
python tools/refdior/analyze_refdior.py \
  --jsonl datasets/refdior/processed/refdior_val.jsonl \
  --outdir "${ANALYSIS_DIR}/analysis" \
  --split val \
  --max-text-len 20 \
  --nmax 80 \
  --k 50
```

运行（失败模式库，需要 preds）：

```bash
python tools/refdior/analyze_refdior.py \
  --jsonl datasets/refdior/processed/refdior_val.jsonl \
  --preds-jsonl outputs/refdior/exp_<ID>/preds/preds_val.jsonl \
  --outdir "${ANALYSIS_DIR}/analysis" \
  --split val \
  --max-text-len 20 \
  --nmax 80 \
  --k 50
```

---

### 4.2.3 输出目录结构（必须）

```text
outputs/refdior/exp_<ANALYSIS_ID>/analysis/
  summary.json
  tables/
    integrity_split_leakage.csv
    mask_unique_values.csv
    scale_buckets.csv
    compactness_stats.csv
    cc_hole_stats.csv
    fidelity_stats.csv
    text_length_trunc.csv
    box_raw_vs_tight.csv
    orientation_stats.csv
    thinness_stats.csv
    preds_bucket_metrics.csv            # only if preds-jsonl provided
    vg_channel_compare.csv              # only if preds-jsonl provided
  figures/
    hist_area_ratio.png
    hist_compactness.png
    hist_cc.png
    hist_hole.png
    hist_vertices.png
    hist_fidelity.png
    hist_text_len.png
    scatter_raw_vs_tight_iou.png
    hist_orientation.png
    hist_thinness.png
    scatter_mask_iou_vs_box_iou.png     # only if preds-jsonl provided
    scatter_vg_channel1_vs_channel2.png # only if preds-jsonl provided
  samples/
    best_mIoU/                          # only if preds-jsonl provided
    worst_mIoU/
    buckets/
      micro_complex/
      micro_thin/
      large_complex/
      ...
    tags.csv
  logs/
    analyze.log
```

---

## 4.3 预期输出（你应该"看到什么"）

你应该看到两类"结果形态"：

1. **统计图/表**：每张图都能回答一个机制问题（例：micro 占比是否真的 > 60%？复杂形状占比是否真的很高？洞/多连通域是否普遍？）
2. **样本库**：worst-k 能被归因到明确标签（micro、thin、multi_cc、hole、box_noise、text_ambiguous…），best-k 能作为 sanity 参照。

---

## 4.4 验证方式（强制：输出自检清单）

✅ 本 Phase 自检不是"看起来像"，而是"能被证伪"：

1. **完整性**：`tables/` 至少 12 个 CSV，`figures/` 至少 10 张图。
2. **一致性**：`summary.json` 里的计数与你 jsonl 行数一致（减去 bad_samples）。
3. **可追溯**：每张 worst/best 可视化图必须能反查到 `sample_id`（文件名含 id）。
4. **可复现**：同一 jsonl + 同一脚本参数 + 同一随机种子 → 输出统计一致（允许图像可视化排序因 ties 微小差异）。

---

## 4.5 失败样本库与标签体系（必须）

### 4.5.1 为什么要建"失败模式库"

如果你没有"失败样本库"，就会永远停留在这类不可证伪的描述：

* "模型不行 / 数据太难 / 遥感就是这样"
* "换个 loss weight 试试 / 换个 schedule 试试"

失败样本库的作用是把问题变成**可复用的定位单元**：

* 同一标签（例如 `micro+thin`）在不同实验里是否一致受益？
* 是否存在"评价口径错误"导致的伪失败？
* 是否存在 conversion 锁上限导致的"再训练也没用"的桶？

---

### 4.5.2 `tags.csv` 的最小字段（固定）

```csv
sample_id,split,expr,img_w,img_h,area_ratio,compactness,cc_count,has_hole,fidelity_iou,gt_box_source,vg_channel,mIoU_mask,IoU_box,tags,notes
```

### 4.5.3 标签词表（最小可用版本）

> 你可以从这 12 个开始，保证每个标签都"可量化 / 可复现 / 可证伪"。

* `micro`：`area_ratio < 0.01`（阈值可在 `summary.json` 里记录）
* `small`：`0.01 ≤ area_ratio < 0.05`
* `thin`：细长目标（见 4.7/T10）
* `complex`：`compactness < 0.5`（或分位数阈值）
* `multi_cc`：连通域 > 1
* `hole`：有洞
* `box_noise`：raw_box vs tight_box IoU 低
* `fidelity_low`：mask→polygon fidelity 低
* `text_long`：token length > max_text_len（会截断）
* `text_ambiguous`：表达式关系词密度低/模板化（简单启发统计即可）
* `vg_channel_gap`：VG 两通道差异大
* `eval_suspect`：疑似评估/坐标系错误（用于快速隔离）

---

## 4.6 本 Phase 交付物（必须保存）

* `outputs/refdior/exp_<ANALYSIS_ID>/analysis/summary.json`
* `outputs/refdior/exp_<ANALYSIS_ID>/analysis/tables/*.csv`
* `outputs/refdior/exp_<ANALYSIS_ID>/analysis/figures/*.png`
* `outputs/refdior/exp_<ANALYSIS_ID>/analysis/samples/*`（best/worst/buckets + `tags.csv`）
* `tools/refdior/analyze_refdior.py`（脚本本身也是交付物）

---

## 4.7 诊断项清单（≥12 项，含"为什么/扭曲/可证伪实验"）

> 你可以把这些项当成"RefDIOR 的坑位地图"。
> 每项都要产出：**表（csv）+ 图（png）+ 样本（png）**。
> 每项都必须回答 3 个为什么：
>
> * 为什么重要（机制）
> * 会怎样扭曲训练/评估
> * 最小可证伪实验（只改数据/训练/评估）

---

### T1. Split 完整性 & 泄漏检查（必须）

* **为什么重要**
  泄漏会让指标虚高，导致你"以为对齐了论文"，实际是评估不干净。
* **会怎样扭曲**
  你可能得到接近 Table 的结果，但复现实验不可迁移、不可复验。
* **怎么做**
  对 `image_path`、`image_id`、`expr` 做 hash，检查 train/val/test 交集。
* **最小可证伪实验**
  如果发现泄漏：重新按官方 split 生成 JSONL；同一 ckpt 再评估，若指标显著变化，则原结果不可信。

---

### T2. 文件健康度（读图/读 mask/尺寸一致性）

* **为什么重要**
  少量坏样本就能让训练出现 NaN、dataloader 卡死、eval 崩溃。
* **会怎样扭曲**

  * 训练：loss 偶发爆炸、某些 step 卡住
  * 评估：pred 缺失导致统计偏差

* **怎么做**
  全量遍历：`cv2.imread` 是否成功；mask 与 image 尺寸是否一致。
* **最小可证伪实验**
  把坏样本隔离 `bad_samples.jsonl` 后再训/评，若训练稳定性显著提升，说明坏样本是主要干扰源。

---

### T3. mask 非空与二值性（unique values）

* **为什么重要**
  RefDIOR 的 mask 格式存在分支：二值 / instance-id / COCO segmentation。你读错分支会直接把监督弄错。
* **会怎样扭曲**

  * 二值化错误 → 目标区域变大/变小 → mIoU 系统性偏低
  * instance-id 未选对 instance → 学到别的对象

* **怎么做**
  抽样+全量统计 `unique values` 与空 mask 比例。
* **最小可证伪实验**
  对同一批样本，分别用两种二值化策略（`>0` vs `==id`）生成 mask，算 tight box 与 polygon 的差异；若差异巨大说明你读错分支。

---

### T4. raw box vs tight box 不一致（标注噪声地图）

* **为什么重要**
  RRSECS 的 VG 指标依赖 box，而 RefDIOR 的 box 与 mask 可能不一致；这会制造"VG 低但 RIS 还行"的假象。
* **会怎样扭曲**

  * 训练用 tight box，评估用 raw box → VG 被系统性压低
  * 训练用 raw box（噪声大）→ 多任务冲突加剧

* **怎么做**
  计算 `IoU(raw_box, tight_box(mask))` 分布并可视化散点。
* **最小可证伪实验**
  固定模型与 eval，只切 `--gt_box_source {raw,mask_tight}`：若 VG 指标显著变化，说明 GT box 口径是关键变量。

---

### T5. micro 目标占比（area_ratio 分布）

* **为什么重要**
  micro 目标下坐标误差的相对代价更大，mIoU 更敏感，训练更尖锐。
* **会怎样扭曲**

  * warp resize / 量化误差 / 1-2 像素偏差 → micro mIoU 崩
  * oIoU 可能看起来"还行"，但 micro 样本 mIoU 很差

* **怎么做**
  `area_ratio = mask_area / (H*W)` 直方图 + 分桶占比。
* **最小可证伪实验**
  只改评估：按 area_ratio 分桶输出 mIoU；若整体 mIoU 主要被 micro 桶拖累，则后续所有 ablation 必须报告 micro 分桶指标。

---

### T6. 形状复杂度（compactness / 周长-面积比）

* **为什么重要**
  PolyFormer 输出的是 polygon 序列，复杂边界会更难拟合；同时 conversion fidelity 更容易下降。
* **会怎样扭曲**

  * 训练：poly loss 更难下降
  * 评估：高阈值 P@0.8/0.9 下掉得更快

* **怎么做**
  compactness：`4πA / P^2`（越小越复杂），画分布并选典型样本。
* **最小可证伪实验**
  不改模型，只改 conversion：提高 `Nmax` 或降低简化强度；若复杂桶 fidelity↑且复杂桶 mIoU↑，说明瓶颈是"标签/表达能力"而非优化器。

---

### T7. 多连通域比例（connected components）

* **为什么重要**
  多连通域决定你 `cc_policy` 的选择：largest/multi/union。
* **会怎样扭曲**

  * 你若默认 largest，会"丢掉部分目标"
  * 你若默认 multi，序列更长、更难训，可能截断更多

* **怎么做**
  统计 `cc_count`，并分桶展示典型样本。
* **最小可证伪实验**
  同一模型训练设定下，仅切 `cc_policy=largest vs multi`，比较：

  * fidelity（GT侧）
  * overfit-50 收敛速度
  * val mIoU（尤其 multi_cc 桶）

---

### T8. hole 比例（洞的存在）

* **为什么重要**
  baseline 通常 drop hole，但洞的比例高会降低"polygon 表达上限"。
* **会怎样扭曲**

  * fidelity 上限被锁死
  * 高阈值指标（P@0.9）更容易受洞影响

* **怎么做**
  通过轮廓层级（`RETR_CCOMP`）统计是否存在内环。
* **最小可证伪实验**
  不改模型，只在 eval 侧比较：

  * GT mask vs rasterize(GT polygons) 的 fidelity 在 hole 桶是否显著更低
    若是，则后续改模型前先承认"监督表达上限"。

---

### T9. polygon 点数分布与截断率（>Nmax）

* **为什么重要**
  AR 序列过长导致训练不稳、截断导致信息丢失，上限下降。
* **会怎样扭曲**

  * mIoU 上不去但 loss 看似下降
  * 复杂样本表现异常差

* **怎么做**
  统计：原始 contour 点数、简化后点数、resample 后点数、被截断比例。
* **最小可证伪实验**
  不改模型，只改 `Nmax`（例如 50/80/120）并同时报告：

  * fidelity 分布
  * truncation ratio
  * mIoU 的复杂桶变化

---

### T10. 方向性与细长目标（orientation / elongation）

* **为什么重要**
  遥感里细长/斜向目标多（道路、桥、跑道、船），warp resize 与边界误差对它们更致命。
* **会怎样扭曲**

  * VG：box IoU 对细长目标很敏感（轻微偏移就差很多）
  * RIS：mask IoU 在细长目标上更依赖边界精度

* **怎么做**

  * orientation：对 mask 前景点 PCA 得主方向角度
  * elongation：tight box 的 `max(w,h)/min(w,h)` 或二阶矩比值

* **最小可证伪实验**
  不改模型，只改数据预处理：`warp` vs `keep_ratio+pad`（Phase 5 的 E6），重点看 `thin` 桶 mIoU。

---

### T11. 文本长度与截断率（严格复现：max_len=20）

* **为什么重要**
  RRSECS 设定 max language length=20；你若用 21 或更大，会引入额外变量。
* **会怎样扭曲**

  * 截断率改变 → 难表达样本被系统性"降难度/升难度"
  * 造成你与 Table 的差异无法归因

* **怎么做**
  用同一 tokenizer 统计：`len(tokens)` 分布；在 `max_len=20` 下截断比例。
* **最小可证伪实验**
  先严格复现跑通后再做：`20 vs 21` 对照，报告：截断率变化 + mIoU 变化，并做可证伪解释（若 mIoU↑且主要来自 text_long 桶，说明截断是真瓶颈）。

---

### T12. mask→polygon fidelity（标签上限）

* **为什么重要**
  如果你的 GT polygon 本身还原不了 GT mask，你的模型最多只能学到"有损标签"。
* **会怎样扭曲**

  * P@0.9 上不去
  * 复杂形状桶 mIoU 上不去

* **怎么做**
  对每个样本算：`IoU(rasterize(gt_polygons), gt_mask)`，输出均值 + P50/P90/P99 + worst-k 可视化。
* **最小可证伪实验**
  只改 conversion（Nmax / simplify / cc_policy），看 fidelity 是否能显著抬升；若 fidelity 上不去，说明是"表达形式上限"，不是训练问题。

---

### T13. 评估差异：mask IoU vs box IoU（同一预测两口径）

* **为什么重要**
  RIS 与 VG 很容易出现"一个好一个差"，但原因可能是：

  * box 来自模型的头 vs box 来自 mask 的派生
  * micro 目标下 box IoU 更敏感

* **会怎样扭曲**
  你可能误判多任务冲突，其实只是评估口径差异。
* **怎么做**
  输出散点：`mIoU_mask` vs `IoU_box`（同样本同预测）。
* **最小可证伪实验**
  对同一 preds，在 eval 中同时输出两种 IoU，并按桶统计差异；若差异集中在 micro/thin 桶，说明口径差异是"数据属性驱动"的。

---

### T14. VG 两通道对齐（pred box vs mask→MBR）

> 这是你在本文档里必须坚持的"可证伪双通道"。
> 目的：找到到底哪个通道能对齐 SeeFormer Table III 的 PolyFormer 行。

* **为什么重要**
  PolyFormer 天生输出 box + polygon，但 RRSECS/SeeFormer 的 VG 表格脚注/实现细节可能导致他们用的是"直接 box"或"mask→box"。
* **会怎样扭曲**
  你会出现：RIS 复现正确，但 VG 永远差一截（或者相反）。
* **怎么做**
  同一 preds 输出：

  * 通道 1：`IoU(pred_box, gt_box)`
  * 通道 2：`IoU(MBR(pred_mask), gt_box)`（MBR 默认 axis-aligned tight box）
    并保存 `vg_channel_compare.csv`。

* **最小可证伪实验**
  以 Table III 为准：哪个通道更接近 Table III，就把它作为"Strict reproduction 的 VG 通道"。另一个通道保留作诊断输出（不要删）。

> 📌 提醒：PolyFormer 论文定义 box 为序列前两个角点（top-left / bottom-right），并把 polygon 作为序列后续部分统一输出；这一点决定了通道 1 的语义是"模型明确监督的 box"。

---

### T15. 失败样本库：best-k / worst-k + buckets

* **为什么重要**
  没有样本库你很难定位"到底错在数据/评估/训练/推理哪一段"。
* **会怎样扭曲**
  你可能用 ablation 做出"看起来提升"的结果，但提升来自评估 bug 或样本偏置。
* **怎么做**
  需要 preds：按 mIoU 排序保存 best-k/worst-k；并按桶（micro+complex、micro+thin、large+complex…）抽样保存。
* **最小可证伪实验**
  同一实验设置下，重复跑两次（同 seed 与不同 seed），比较 worst-k 的标签分布是否稳定；若不稳定说明随机性与 dataloader 影响大，需要 Phase 3 的复现锁定更严格。

---

## 4.8 新增脚本参考实现：`tools/refdior/analyze_refdior.py`

> 下面给出"能跑起来"的版本（GT-only + 可选 preds）。
> 你可以直接复制到 repo；不依赖 fairseq，不依赖模型。
> ⚠️ 注意：为了可读性，这里用相对朴素的实现；数据量大时建议加 tqdm（可选）。

```python
# tools/refdior/analyze_refdior.py
import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Basic geometry helpers
# -------------------------

def iou_mask(a01: np.ndarray, b01: np.ndarray) -> Tuple[float, int, int]:
    inter = int(np.logical_and(a01 > 0, b01 > 0).sum())
    union = int(np.logical_or(a01 > 0, b01 > 0).sum())
    return float(inter) / float(union + 1e-6), inter, union

def iou_box_xyxy(a, b) -> Tuple[float, int, int]:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / float(union + 1e-6), int(inter), int(union)

def tight_box_from_mask(mask01: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [x1, y1, x2, y2]

def rasterize_polygons(polys: List[np.ndarray], h: int, w: int) -> np.ndarray:
    canvas = np.zeros((h, w), dtype=np.uint8)
    for p in polys:
        if p is None or len(p) < 3:
            continue
        pts = np.round(p).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [pts], 1)
    return canvas

def polygon_signed_area(poly_xy: np.ndarray) -> float:
    x = poly_xy[:, 0]
    y = poly_xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def compactness_from_mask(mask01: np.ndarray) -> float:
    # 4*pi*A / P^2
    m = (mask01 > 0).astype(np.uint8)
    area = float(m.sum())
    if area <= 0:
        return 0.0
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    perim = float(sum(cv2.arcLength(c, True) for c in contours))
    if perim <= 1e-6:
        return 0.0
    return float(4.0 * math.pi * area / (perim * perim))

def cc_count(mask01: np.ndarray) -> int:
    m = (mask01 > 0).astype(np.uint8)
    num, _ = cv2.connectedComponents(m)
    # connectedComponents counts background as 0 label
    return int(max(0, num - 1))

def has_hole(mask01: np.ndarray) -> bool:
    # detect holes via contour hierarchy
    m = (mask01 > 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return False
    # hierarchy: [Next, Prev, First_Child, Parent]
    # A contour with Parent != -1 indicates it is a hole (child contour)
    hierarchy = hierarchy.reshape(-1, 4)
    return bool(np.any(hierarchy[:, 3] != -1))

def pca_orientation(mask01: np.ndarray) -> Optional[float]:
    ys, xs = np.where(mask01 > 0)
    if len(xs) < 20:
        return None
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    pts = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, int(np.argmax(eigvals))]
    # angle in degrees, [-90, 90]
    ang = float(np.degrees(np.arctan2(v[1], v[0])))
    if ang > 90:
        ang -= 180
    if ang < -90:
        ang += 180
    return ang

def elongation_from_box_xyxy(box) -> Optional[float]:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    return float(max(w, h) / min(w, h))

def safe_imread(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

def safe_maskread(path: str) -> Optional[np.ndarray]:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 0]
    return m


# -------------------------
# IO helpers
# -------------------------

def load_jsonl(path: Path) -> List[dict]:
    lines = path.read_text().splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        out.append(json.loads(ln))
    return out

def load_preds_jsonl(path: Path) -> Dict[str, dict]:
    # key by sample_id
    m = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        obj = json.loads(ln)
        sid = str(obj.get("sample_id"))
        m[sid] = obj
    return m

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_csv(path: Path, rows: List[dict]):
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def save_hist(values, title, xlabel, out_png: Path, bins=50):
    ensure_dir(out_png.parent)
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def save_scatter(x, y, title, xlabel, ylabel, out_png: Path):
    ensure_dir(out_png.parent)
    plt.figure()
    plt.scatter(x, y, s=6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# -------------------------
# Visualization helpers
# -------------------------

def draw_overlay(img_bgr, mask01, polys, boxes: Dict[str, Optional[List[int]]], title: str):
    img = img_bgr.copy()
    h, w = img.shape[:2]

    # mask overlay (green)
    if mask01 is not None:
        m = (mask01 > 0).astype(np.uint8)
        color = np.zeros_like(img)
        color[..., 1] = 180
        img = np.where(m[..., None].astype(bool), (0.6 * img + 0.4 * color).astype(np.uint8), img)

    # polygons (red)
    if polys:
        for p in polys:
            if p is None or len(p) < 2:
                continue
            pts = np.round(p).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)

    # boxes
    def draw_box(b, color, name):
        if b is None:
            return
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    draw_box(boxes.get("raw"), (255, 0, 0), "raw")
    draw_box(boxes.get("tight"), (0, 255, 255), "tight")
    draw_box(boxes.get("pred_box"), (0, 255, 0), "pred_box")
    draw_box(boxes.get("pred_mbr"), (0, 128, 255), "pred_mbr")

    cv2.putText(img, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


# -------------------------
# Main
# -------------------------

@dataclass
class StatsAcc:
    n: int = 0
    n_bad: int = 0

def percentile(xs, ps=(0, 50, 90, 99, 100)):
    xs = np.asarray(xs, dtype=np.float64)
    if xs.size == 0:
        return {f"p{p}": None for p in ps}
    return {f"p{p}": float(np.percentile(xs, p)) for p in ps}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--preds-jsonl", default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-text-len", type=int, default=20)
    ap.add_argument("--nmax", type=int, default=80)
    ap.add_argument("--k", type=int, default=50, help="top-k for best/worst sample saving")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir / "tables")
    ensure_dir(outdir / "figures")
    ensure_dir(outdir / "samples")
    ensure_dir(outdir / "logs")

    data = load_jsonl(Path(args.jsonl))
    preds = load_preds_jsonl(Path(args.preds_jsonl)) if args.preds_jsonl else None

    # Collect per-sample stats
    rows_integrity = []
    rows_maskuniq = []
    rows_scale = []
    rows_compact = []
    rows_cc_hole = []
    rows_fidelity = []
    rows_text = []
    rows_box = []
    rows_orient = []
    rows_thin = []
    rows_preds_bucket = []
    rows_vg_channel = []

    area_ratios = []
    compactnesses = []
    cc_counts = []
    hole_flags = []
    vertices_counts = []
    fidelitys = []
    text_lens = []
    raw_tight_ious = []
    orientations = []
    elongations = []

    # if preds available
    pred_mious = []
    pred_box_ious = []
    vg_ch1 = []
    vg_ch2 = []

    # Sample visual saving buffers
    scored_samples = []  # (score, sample_id, sample_dict)

    for item in data:
        sid = str(item.get("sample_id"))
        img_path = item.get("image_path")
        mask_path = item.get("mask_path")
        expr = item.get("expr", "")
        w = int(item.get("img_w"))
        h = int(item.get("img_h"))

        # read image (optional for stats, required for vis)
        img = safe_imread(img_path) if img_path else None
        if img is None:
            rows_integrity.append({"sample_id": sid, "issue": "image_read_failed", "path": str(img_path)})
            continue

        # read mask (if present)
        mask = safe_maskread(mask_path) if mask_path else None
        if mask is None:
            rows_integrity.append({"sample_id": sid, "issue": "mask_read_failed", "path": str(mask_path)})
            continue

        if mask.shape[0] != h or mask.shape[1] != w:
            rows_integrity.append({
                "sample_id": sid, "issue": "mask_size_mismatch",
                "mask_h": mask.shape[0], "mask_w": mask.shape[1], "json_h": h, "json_w": w
            })
            # still continue, but stats may be off
        # mask unique values
        uniq = np.unique(mask)
        rows_maskuniq.append({"sample_id": sid, "unique_len": int(len(uniq)), "unique_head": " ".join(map(str, uniq[:10].tolist()))})

        # binarize GT mask (this assumes JSONL already resolved instance selection / decode)
        mask01 = (mask > 0).astype(np.uint8)
        if int(mask01.sum()) == 0:
            rows_integrity.append({"sample_id": sid, "issue": "empty_mask"})
            continue

        # scale
        area = float(mask01.sum())
        area_ratio = area / float(h * w + 1e-6)
        area_ratios.append(area_ratio)
        rows_scale.append({"sample_id": sid, "area_ratio": area_ratio, "mask_area": int(area), "img_area": int(h*w)})

        # compactness
        comp = compactness_from_mask(mask01)
        compactnesses.append(comp)
        rows_compact.append({"sample_id": sid, "compactness": comp})

        # cc & hole
        cc = cc_count(mask01)
        cc_counts.append(cc)
        hole = has_hole(mask01)
        hole_flags.append(int(hole))
        rows_cc_hole.append({"sample_id": sid, "cc_count": cc, "has_hole": int(hole)})

        # polygons & fidelity (from JSONL polygons)
        polys_list = item.get("polygons", [])
        polys = []
        for poly in polys_list:
            p = np.asarray(poly, dtype=np.float32)
            if p.ndim != 2 or p.shape[1] != 2:
                continue
            polys.append(p)
            vertices_counts.append(int(len(p)))

        # fidelity: rasterize gt polygons vs gt mask
        if polys:
            rast = rasterize_polygons(polys, h=h, w=w)
            fiou, _, _ = iou_mask(rast, mask01)
            fidelitys.append(fiou)
            rows_fidelity.append({"sample_id": sid, "fidelity_iou": fiou, "n_polys": len(polys), "n_vertices_total": sum(len(p) for p in polys)})
        else:
            rows_fidelity.append({"sample_id": sid, "fidelity_iou": None, "n_polys": 0, "n_vertices_total": 0})

        # text length proxy (word count). If you want tokenizer-accurate lengths, pass a tokenizer here.
        # For strict RRSECS reproduction, tokenizer length must be computed the same way as training.
        tlen = len(expr.strip().split())
        text_lens.append(tlen)
        trunc = int(tlen > args.max_text_len)
        rows_text.append({"sample_id": sid, "word_len": tlen, "max_text_len": args.max_text_len, "truncated": trunc})

        # box consistency
        raw_box = item.get("raw_box_xyxy", None)
        tight_box = item.get("tight_box_xyxy", None)
        if raw_box and tight_box:
            biou, _, _ = iou_box_xyxy(raw_box, tight_box)
            raw_tight_ious.append(biou)
            rows_box.append({"sample_id": sid, "raw_tight_iou": biou})
        else:
            rows_box.append({"sample_id": sid, "raw_tight_iou": None})

        # orientation / thinness
        ang = pca_orientation(mask01)
        if ang is not None:
            orientations.append(ang)
            rows_orient.append({"sample_id": sid, "angle_deg": ang})

        elong = elongation_from_box_xyxy(tight_box_from_mask(mask01))
        if elong is not None:
            elongations.append(elong)
            rows_thin.append({"sample_id": sid, "elongation": elong})

        # preds-driven analysis
        if preds is not None and sid in preds:
            pr = preds[sid]
            # expected fields in preds.jsonl (you should ensure eval script writes them):
            # - pred_polygons (pixel coords) or pred_mask_rle/bitmap
            # - pred_box_xyxy (pixel coords)
            # - (optional) pred_mask01 (if you store)
            # - metrics per-sample (optional, but we can recompute)

            pred_box = pr.get("pred_box_xyxy", None)
            pred_polys_list = pr.get("pred_polygons", None)
            pred_polys = []
            if pred_polys_list:
                for poly in pred_polys_list:
                    p = np.asarray(poly, dtype=np.float32)
                    if p.ndim == 2 and p.shape[1] == 2 and len(p) >= 3:
                        pred_polys.append(p)

            pred_mask01 = None
            if pred_polys:
                pred_mask01 = rasterize_polygons(pred_polys, h=h, w=w)

            # compute mask mIoU
            if pred_mask01 is not None:
                miou, _, _ = iou_mask(pred_mask01, mask01)
            else:
                miou = 0.0

            # VG channels:
            # channel1: pred_box vs GT box
            # channel2: MBR(pred_mask) vs GT box
            gt_box = raw_box if raw_box else tight_box
            if gt_box is None:
                box_iou_ch1 = None
                box_iou_ch2 = None
            else:
                box_iou_ch1 = iou_box_xyxy(pred_box, gt_box)[0] if pred_box else None
                pred_mbr = tight_box_from_mask(pred_mask01) if pred_mask01 is not None else None
                box_iou_ch2 = iou_box_xyxy(pred_mbr, gt_box)[0] if pred_mbr else None

            pred_mious.append(miou)
            pred_box_ious.append(box_iou_ch1 if box_iou_ch1 is not None else 0.0)
            if box_iou_ch1 is not None:
                vg_ch1.append(box_iou_ch1)
            if box_iou_ch2 is not None:
                vg_ch2.append(box_iou_ch2)

            rows_vg_channel.append({
                "sample_id": sid,
                "IoU_ch1_pred_box": box_iou_ch1,
                "IoU_ch2_mask_mbr": box_iou_ch2,
            })

            # For sample library
            scored_samples.append((miou, sid, item, pr))

    # Save tables
    save_csv(outdir / "tables" / "integrity_issues.csv", rows_integrity)
    save_csv(outdir / "tables" / "mask_unique_values.csv", rows_maskuniq)
    save_csv(outdir / "tables" / "scale_buckets.csv", rows_scale)
    save_csv(outdir / "tables" / "compactness_stats.csv", rows_compact)
    save_csv(outdir / "tables" / "cc_hole_stats.csv", rows_cc_hole)
    save_csv(outdir / "tables" / "fidelity_stats.csv", rows_fidelity)
    save_csv(outdir / "tables" / "text_length_trunc.csv", rows_text)
    save_csv(outdir / "tables" / "box_raw_vs_tight.csv", rows_box)
    save_csv(outdir / "tables" / "orientation_stats.csv", rows_orient)
    save_csv(outdir / "tables" / "thinness_stats.csv", rows_thin)
    if preds is not None:
        save_csv(outdir / "tables" / "vg_channel_compare.csv", rows_vg_channel)

    # Save figures
    if area_ratios:
        save_hist(area_ratios, "Area ratio (mask_area / img_area)", "area_ratio", outdir / "figures" / "hist_area_ratio.png")
    if compactnesses:
        save_hist(compactnesses, "Compactness (4πA/P^2)", "compactness", outdir / "figures" / "hist_compactness.png")
    if cc_counts:
        save_hist(cc_counts, "Connected components count", "cc_count", outdir / "figures" / "hist_cc.png", bins=30)
    if hole_flags:
        save_hist(hole_flags, "Has hole (0/1)", "has_hole", outdir / "figures" / "hist_hole.png", bins=2)
    if vertices_counts:
        save_hist(vertices_counts, "Polygon vertices count", "n_vertices", outdir / "figures" / "hist_vertices.png", bins=50)
    if fidelitys:
        save_hist(fidelitys, "Mask->Polygon fidelity IoU", "fidelity_iou", outdir / "figures" / "hist_fidelity.png", bins=50)
    if text_lens:
        save_hist(text_lens, "Text length (word count proxy)", "word_len", outdir / "figures" / "hist_text_len.png", bins=50)
    if raw_tight_ious:
        save_hist(raw_tight_ious, "IoU(raw_box, tight_box)", "IoU", outdir / "figures" / "scatter_raw_vs_tight_iou.png", bins=50)
    if orientations:
        save_hist(orientations, "Orientation (PCA angle deg)", "angle_deg", outdir / "figures" / "hist_orientation.png", bins=60)
    if elongations:
        save_hist(elongations, "Elongation (max(w,h)/min(w,h))", "elongation", outdir / "figures" / "hist_thinness.png", bins=60)

    if preds is not None and pred_mious and pred_box_ious:
        save_scatter(pred_mious, pred_box_ious, "mask mIoU vs box IoU (ch1)", "mIoU_mask", "IoU_box_ch1", outdir / "figures" / "scatter_mask_iou_vs_box_iou.png")
    if preds is not None and vg_ch1 and vg_ch2:
        n = min(len(vg_ch1), len(vg_ch2))
        save_scatter(vg_ch1[:n], vg_ch2[:n], "VG channel1 vs channel2", "IoU_ch1_pred_box", "IoU_ch2_mask_mbr", outdir / "figures" / "scatter_vg_channel1_vs_channel2.png")

    # Save sample library if preds available
    if preds is not None and scored_samples:
        ensure_dir(outdir / "samples" / "best_mIoU")
        ensure_dir(outdir / "samples" / "worst_mIoU")
        scored_samples.sort(key=lambda x: x[0])
        worst = scored_samples[: args.k]
        best = scored_samples[-args.k :]

        def save_samples(group, out_subdir: Path):
            for miou, sid, item, pr in group:
                img = safe_imread(item["image_path"])
                mask = safe_maskread(item["mask_path"])
                mask01 = (mask > 0).astype(np.uint8)
                h, w = mask01.shape[:2]

                polys = [np.asarray(p, dtype=np.float32) for p in item.get("polygons", [])] if item.get("polygons") else []
                pred_polys = [np.asarray(p, dtype=np.float32) for p in pr.get("pred_polygons", [])] if pr.get("pred_polygons") else []
                pred_mask01 = rasterize_polygons(pred_polys, h=h, w=w) if pred_polys else None
                pred_mbr = tight_box_from_mask(pred_mask01) if pred_mask01 is not None else None

                overlay = draw_overlay(
                    img_bgr=img,
                    mask01=mask01,
                    polys=polys,
                    boxes={
                        "raw": item.get("raw_box_xyxy", None),
                        "tight": item.get("tight_box_xyxy", None),
                        "pred_box": pr.get("pred_box_xyxy", None),
                        "pred_mbr": pred_mbr,
                    },
                    title=f"sid={sid} mIoU={miou:.3f}"
                )
                out_path = out_subdir / f"sid_{sid}_mIoU_{miou:.3f}.png"
                cv2.imwrite(str(out_path), overlay)

        save_samples(worst, outdir / "samples" / "worst_mIoU")
        save_samples(best, outdir / "samples" / "best_mIoU")

    # Summary
    summary = {
        "split": args.split,
        "n_samples": len(data),
        "area_ratio": {"mean": float(np.mean(area_ratios)) if area_ratios else None, **percentile(area_ratios)},
        "compactness": {"mean": float(np.mean(compactnesses)) if compactnesses else None, **percentile(compactnesses)},
        "cc_count": {"mean": float(np.mean(cc_counts)) if cc_counts else None, **percentile(cc_counts)},
        "has_hole_rate": float(np.mean(hole_flags)) if hole_flags else None,
        "fidelity_iou": {"mean": float(np.mean(fidelitys)) if fidelitys else None, **percentile(fidelitys)},
        "text_word_len": {"mean": float(np.mean(text_lens)) if text_lens else None, **percentile(text_lens)},
        "raw_tight_box_iou": {"mean": float(np.mean(raw_tight_ious)) if raw_tight_ious else None, **percentile(raw_tight_ious)},
        "preds": {
            "available": preds is not None,
            "mIoU_mask": {"mean": float(np.mean(pred_mious)) if pred_mious else None, **percentile(pred_mious)},
            "VG_ch1_pred_box": {"mean": float(np.mean(vg_ch1)) if vg_ch1 else None, **percentile(vg_ch1)},
            "VG_ch2_mask_mbr": {"mean": float(np.mean(vg_ch2)) if vg_ch2 else None, **percentile(vg_ch2)},
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[analyze] wrote:", outdir / "summary.json")


if __name__ == "__main__":
    main()
```

---

# Phase 5：实验矩阵 + Thinking Prompts（只改数据/训练/评估）

> 目标：在 **不改模型结构** 的前提下，用最小可证伪实验把"机制假设"变成数据。
> 本 Phase 必须严格遵守你在 Part 1 定义的两类设定：
>
> * ✅ **Strict reproduction**：只为对齐 SeeFormer 表格中 PolyFormer 行（RRSECS 设定），任何偏离都必须显式记录为 ablation。
> * 🧪 **Ablation/探索**：只能在 Gate 通过后运行，用来解释机制（max_len=21、inverse_sqrt、lr=5e-4…都属于这里）。

---

## 5.0 Stop condition（本 Phase 完成标准）

✅ 至少完成：

1. **Strict reproduction Gate 已通过**（RIS + VG 两行指标在容忍范围内）
2. 至少跑完 6 个 ablation（建议 E2/E4/E5/E6/E7/E9），且每个实验都有：

   * `args.json`
   * `train.log`
   * `eval.json`（RIS/VG/RECS 分开 + VG 双通道）
   * `preds.jsonl`
   * `analysis/`（至少包含 worst/best 与 bucket 指标）

3. 每条 Thinking Prompt 至少跑一个"最小验证实验"，并写出可证伪结论（哪怕结论是否定的）。

---

## 5.0 Failure tree（不通过先查谁）

1. Gate 仍不通过 → 回到 Phase 3 的"排查树"（数据→评估→训练设定→推理口径→随机性）。
2. Ablation 结果不稳定 → 检查 seed 固化与 dataloader nondeterminism（尤其是多卡/amp）。
3. 某些 ablation 只提升 oIoU 不提升 mIoU → 检查是否"只改善大目标桶"，并回到 Phase 4 分桶证据。

---

## 5.1 实验输出目录模板（强制）

每个实验目录必须是"可审计单元"：

```text
outputs/refdior/exp_<ID>/
  args.json
  train.log
  ckpts/
  eval/
    eval_val.json
    eval_test.json
  preds/
    preds_val.jsonl
    preds_test.jsonl
  vis/
    val_worstk/
    val_bestk/
  analysis/
    summary.json
    tables/
    figures/
    samples/
      tags.csv
```

---

## 5.2 实验矩阵（按优先级排序）

> 下面给你一套"从机制出发"的实验矩阵。
> ⚠️ 标注为 **(post-Gate)** 的实验：必须 Gate 通过后再跑；否则你会把"对齐没成功"的系统误差当成机制差异。

| 实验ID | 允许阶段      | 改动（不改结构）                    | 机制假设（你在验证什么）          | 主要看哪些桶/证据                      |
| ---- | --------- | --------------------------- | --------------------- | ------------------------------ |
| E0   | pre-Gate  | Overfit-50                  | pipeline 正确性          | loss 曲线 + 可视化贴合                |
| E1   | pre-Gate  | Train-1epoch                | dataloader/评估稳定性      | NaN/卡死/输出完整                    |
| E2   | post-Gate | `Nmax=50/80/120`            | 序列长度/截断决定上限           | `complex` 桶 + fidelity + P@0.9 |
| E3   | post-Gate | 简化强度（RDP eps）               | 过简化造成监督失真             | fidelity P1/P5 + worst-case    |
| E4   | post-Gate | `box_source raw vs tight`   | raw box 噪声污染联合训练      | VG 指标 + `box_noise` 桶          |
| E5   | post-Gate | `max_text_len=20 vs 21`     | 截断是否是真瓶颈              | `text_long` 桶 + 截断率            |
| E6   | post-Gate | `warp vs keep_ratio+pad`    | warp 放大 micro/thin 误差 | `micro+thin` 桶 mIoU            |
| E7   | post-Gate | loss schedule（先 poly 后 box） | 多任务梯度冲突可被训练策略缓解       | VG/RIS 同升？worst-k 标签变化         |
| E8   | post-Gate | grad clip（1.0）              | 数值不稳导致收敛差             | loss 波动/NaN 率                  |
| E9   | post-Gate | EMA（推理用 EMA 权重）             | 推理稳定性与泛化              | val 波动/高阈值 P@0.8/0.9           |

---

## 5.3 Thinking Prompts（每条都配"最小验证实验 + 可证伪条件"）

> 这些不是"建议"，而是你在 Phase 4 证据驱动下，必须能被证伪的机制问题。
> 每条都要：**假设 → 最小实验 → 预期现象 → 可证伪条件**。

---

### TP1：为什么 micro target 会让坐标回归的 loss landscape 更尖锐？

* **机制假设**
  同样的归一化坐标误差在 micro 目标上对应更大的相对形状偏差，导致 IoU 对误差更敏感。
* **最小验证实验（post-Gate）**
  不改模型，只改分析：按 area_ratio 分桶，统计 `mIoU` 随 `|Δcoord|`（你可以从预测 polygon 与 GT polygon 采样点的平均偏差近似）变化的斜率。
* **预期现象**
  micro 桶的斜率更陡（误差小幅增加，mIoU 大幅下降）。
* **可证伪条件**
  若 micro 与 large 桶斜率相近，说明瓶颈更可能来自：评估口径、resize 同步错误、或标签 fidelity。

---

### TP2：为什么点序不一致会导致 AR 学习像在学"多解映射"？

* **机制假设**
  同一轮廓存在多个等价序列（起点旋转、方向翻转），AR 被迫拟合多模态输出。PolyFormer 明确要求顶点顺序与起点规则来消除这种多解。
* **最小验证实验（pre-Gate）**
  Overfit-50：对比

  * A：关闭 canonicalization（随机起点/不强制顺时针）
  * B：强制 clockwise + canonical start

* **预期现象**
  B 收敛更快、最终 mIoU 更高、预测点更稳定。
* **可证伪条件**
  若无差异，优先查：坐标尺度（[0,1] vs 像素）、quantize 流程是否一致、eval rasterize 是否正确。

---

### TP3：为什么 warp resize 会放大细长目标误差？

* **机制假设**
  warp 改变长宽比，使细长目标的几何关系被扭曲；polygon 学到的是"形变后的边界"，泛化更差。
* **最小验证实验（post-Gate）**
  E6：仅改数据预处理（warp vs keep_ratio+pad），看 `thin` 桶 mIoU 与 VG 通道差异。
* **预期现象**
  keep_ratio+pad 对 thin 桶更友好；warp 的 thin 桶 worst-k 更集中。
* **可证伪条件**
  若差异不显著，则主要瓶颈可能是标签 fidelity 或 box/mask 口径冲突，而非 resize。

---

### TP4：VG 两通道差异来自哪里？（pred box vs mask→MBR）

* **机制假设**
  多任务训练下，"box 头"和"polygon 头"可能学习到不同的定位偏好；VG 若用 mask→MBR，会更接近 RIS 的表现。
* **最小验证实验（pre-Gate 即可做分析，post-Gate 才做结论）**
  在 eval 中同时输出通道 1/2，按桶统计差异，并可视化差异最大的样本。
* **预期现象**
  micro/thin 桶差异最大。
* **可证伪条件**
  若差异集中在所有桶且呈系统偏移，优先怀疑：坐标尺度/单位错或 box decode 错。

---

### TP5：raw box 噪声是否是 VG 上不去的主因？

* **机制假设**
  raw box 与 mask 不一致会污染 box 监督，导致联合训练冲突更大。
* **最小验证实验（post-Gate）**
  E4：只切 `box_source raw vs mask_tight`，并固定 `gt_box_source` 与之保持一致。
* **预期现象**
  若 raw 噪声严重：用 tight 更稳，VG 更高或更稳定。
* **可证伪条件**
  若两者差异不大，则 VG 可能主要受评估通道/推理后处理影响。

---

## 5.4 本 Phase 交付物（必须保存）

* 每个实验目录：`outputs/refdior/exp_<ID>/...` 完整
* `experiments.md`（可选但强烈建议）：记录每个实验的结论、证据链接（eval.json、analysis 图表、worst-k 样例）
* 至少 1 份"可写进论文的机制结论"（哪怕是 negative result）

---

## 5.5 Part 2 结束检查（Checklist）

* ✅ Phase 4 输出目录完整，≥12 项诊断均有图表与样本
* ✅ VG 双通道已实现并同时输出
* ✅ 有失败样本库与最小标签体系 `tags.csv`
* ✅ Phase 5 至少 6 个 post-Gate ablation 已跑，并有可证伪结论
* ✅ 任何"提升/下降"都能归因到明确桶或明确口径差异（而不是主观猜测）

---

# 附录 A：RefDIOR 中间格式 JSONL Schema（固定）

> JSONL 的核心思想：**存像素坐标 + 原始尺寸**，把 normalize/quantize 留给 dataset 内部去做（对齐你 repo 的硬事实）。

```json
{
  "sample_id": "string(unique)",
  "split": "train|val|test",
  "image_path": "string",
  "mask_path": "string|null",
  "image_id": "string|int|null",

  "expr": "string",
  "img_w": "int",
  "img_h": "int",

  "raw_box_xyxy": [x1, y1, x2, y2],
  "tight_box_xyxy": [x1, y1, x2, y2],

  "polygons": [
    [[x, y], [x, y], "..."],
    [[x, y], [x, y], "..."]
  ],

  "poly_meta": {
    "cc_policy": "largest|multi",
    "nmax": 80,
    "n_vertices": [80, 0],
    "has_hole": false,
    "fidelity_iou": 0.987
  },

  "notes": "string|null"
}
```

---

# 附录 B：必须落地的工具脚本清单（可直接复制）

> 说明：这里给的是 **"能跑的工程脚本"**，不是伪代码。
> 你可以按需精简，但 **extract_base_cmd / sniff / convert / eval / check_gate** 这五个必须保留，否则文档不满足"可执行 + 可诊断"。

### B1) `tools/refdior/extract_base_cmd.py`（抽取 baseline 命令，消灭占位符）

* 输入：`run_scripts/finetune/train_polyformer_b.sh`
* 输出：`outputs/refdior/refdior_base_cmd.sh`（单行 python train 命令）

要求：

* 能识别 `python train.py ...` 或 `python -m fairseq_cli.train ...`
* 保留所有原始参数顺序与值
* 去掉 bash 变量与注释（无法解析的变量要直接报错提示）

### B2) `tools/refdior/hash_data_version.py`

* 输出一个 JSON：包含文件清单 hash（建议 sha256）、总文件数、总大小
* 默认忽略：`.DS_Store`、`__MACOSX`、`*.tmp` 等

### B3) `tools/refdior/sniff_refdior.py`

* 输出 JSON，至少包括：

  * 发现的 annotation 文件列表与 top-level keys
  * mask 文件后缀统计（png/jpg/json…）
  * 抽样若干 mask 的 unique values（判断二值/instance-id）

### B4) `tools/refdior/convert_refdior.py`

必须实现：

* sniff→选择解析器：COCO(dict) / list-of-dict / jsonl
* mask 读取分支：PNG 二值 / instance-id / COCO-RLE / COCO-polygon
* mask→polygon：cv2 contour + canonical（顺时针 + 起点最靠近原点）
* 输出 `bad_samples.jsonl`（空 mask/读失败/退化 polygon 全部进 bad，不允许静默跳过）

### B5) `tools/refdior/eval_refdior.py`

必须实现：

* 指标：P@0.5..0.9 / oIoU / mIoU / Sum（按 RRSECS 定义）
* RIS：pred polygon→mask，与 gt mask IoU
* VG（双通道）：

  * 通道1：pred box 直接 IoU
  * 通道2：pred polygon→mask→MBR 作为 pred box IoU

* 输出：

  * `eval.json`：包含 RIS、VG(ch1)、VG(ch2) 三套 7 指标 + Sum
  * `preds.jsonl`：每个样本保存 pred_box/pred_polygons/派生 mask_box/IoU 等（用于诊断）

### B6) `tools/refdior/check_gate.py`

* 读取 `eval_test.json`
* 对照 Gate-1/Gate-2 的目标值与误差容忍
* 打印逐项 delta，并输出 PASS/FAIL

---

# 附录 C：评估指标实现与自检三件套（GT-vs-GT / fidelity / 口径差异）

> 指标定义来自 RRSECS：P@0.5..0.9、oIoU、mIoU、Sum（七项相加）。
> 你的实现必须完全一致，否则 Gate 不可能过。

### C1) 7 指标聚合（mask 或 box 都适用）

* `P@t = mean(IoU >= t) * 100`
* `mIoU = mean(IoU) * 100`
* `oIoU = sum(inter) / sum(union) * 100`
* `Sum = Σ_{t∈{0.5..0.9}} P@t + oIoU + mIoU`

### C2) 自检三件套（Phase 3 强制）

1. **GT-vs-GT**

   * 用 gt mask 与 gt box 作为 pred：所有 IoU=1
   * 如果这里不等于 1：说明你不是在和论文同一个数学定义上说话

2. **polygon-fidelity**

   * `IoU(rasterize(gt_polygons), gt_mask)`
   * 这是"标签上限"：如果 fidelity 很差，模型学得再好也到不了表格

3. **VG 双通道差异**

   * 同一批预测，同时算：

     * pred_box vs gt_box
     * MBR(pred_mask) vs gt_box
   * 记录两套指标，解释差异（脚注导致的口径差异）

---

# 文档结束

> 本指南覆盖 Phase 0–5 的完整流程。
> 严格按照本文档执行，可以保证：
>
> 1. **可执行**：每一步都有明确的命令和输出
> 2. **可诊断**：失败时有明确的排查树
> 3. **可证伪**：每个结论都有数据支撑
> 4. **可复现**：`args.json + preds.jsonl + vis/` 构成完整证据链