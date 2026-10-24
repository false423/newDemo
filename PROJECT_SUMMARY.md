# 项目总结 — OUR-V9 语义流形（高光谱图像分类）

## 1. 这个项目做了什么

这是一个**高光谱图像分类（HSIC）**研究项目，实现了一种跨场景 / 跨域小样本迁移学习方法，
代码内部命名为 **「OUR-V9 可迁移语义流形（Transferable Semantic Manifold）」**。

具体任务：

- **源域**（有标签、样本多）：多个高光谱场景，它们来自**不同传感器，光谱波段数不同**
  （异质 / heterogeneous）。
- **目标域**（标签极少）：一个场景，**每类只有 15 个标注样本**（15-shot）。
- **目标**：把源域的知识迁移过来，准确分类目标场景中每一个有标签的像素。

核心约束是：源域与目标域的**光谱通道数不同**，因此模型不能简单地共享固定的第一层权重。
该方法的中心主张是：跨域应该共享的是**语义表示基（语义流形）**，而*不是*一个共同的
特征分布。

## 2. 核心思想

> 共享的对象 = **可迁移的语义流形（基）**，而不是一个共同分布。
> 目标域私有学习 = **仅一个浅层光谱适配器**。

1. 每个域（源或目标）先经过一个小的 **光谱适配器（Spectral Adapter）**（只用 1×1 卷积），
   把各自的原始波段数投影到统一的 `adapter_channels` 宽度。
2. 所有域接着共享一个**冻结的语义编码器**和**语义字典**。字典是一个可学习的基
   `B = {b_k}`；每个样本对基原子做注意力得到系数 `alpha`，最终特征 = 编码器输出与字典重建
   结果的**门控融合**。共享的是「基」，每个样本保留自己的系数。
3. 只有目标域的浅层光谱适配器用 15-shot 标签进行微调；共享编码器和字典保持**冻结**。

这种设计刻意避开了「源域与目标域特征必须对齐到同一个分布」这一常见假设。

## 3. 模型架构

定义在 `test/OUR-Y.py` 与 `test/OUR-Y-draw.py` 中（模型代码一致）。

| 组件 | 类名 | 作用 |
|------|------|------|
| 光谱适配器 | `SpectralAdapter` | 1×1 卷积 + GroupNorm + SiLU，把原始波段映射到 `adapter_channels`（64）。每个源一个，目标域一个。 |
| 共享编码器 | `SharedSemanticEncoder` | CNN 主干（Conv–GN–SiLU、残差块、×2 下采样）→ `AdaptiveAvgPool2d(1)` → Linear → LayerNorm → `feature_dim`（128）。 |
| 语义字典 | `SharedSemanticDictionary` | 可学习原子 `B ∈ R^{num_atoms × feature_dim}`（64 个原子）。Softmax 注意力 `alpha = softmax(h·Bᵀ/τ)`；门控融合 `z = h + gate·proj(alpha·B)`。 |
| 源域分类头 | `nn.Linear(feature_dim, n_classes)` | 每个源域一个分类头。 |
| 顶层模型 | `SemanticManifoldModel` | 把适配器 + 共享编码器 + 字典 + 分类头组合起来。 |

关键超参（默认值）：

```
adapter_channels = 64      num_atoms = 64        feature_dim = 128
dictionary_temperature = 0.10                    dictionary_target_entropy = 0.45
contrast_temperature = 0.10                      hidden_channels = 64
```

## 4. 训练流程（两阶段）

### 阶段一 — 多源可迁移语义流形学习（不使用目标域数据）
训练：源适配器、共享编码器、语义字典、源分类头。
损失 = 源域交叉熵 + 有监督对比（SupCon）+ 字典正则化。

- `lambda_source_cls * L_cls`
- `+ lambda_supcon * L_supcon`
- `+ lambda_dict_diversity * L_diversity`（原子尽量正交）
- `+ lambda_dict_selectivity * L_selectivity`（把每个样本的原子熵推向目标熵）
- `+ lambda_dict_balance * L_balance`（批量内原子使用尽量均匀）

### 阶段二 — 15-shot 目标光谱适配器微调（使用目标标签）
训练：**仅**目标 `SpectralAdapter`（共享编码器 + 字典冻结）。
- 目标适配器通过光谱权值插值从源适配器初始化（`initialize_target_adapter_from_sources`）。
- 小样本片段式训练：每类每片段 `support=5`、`query=10`。
- 损失 = 片段式原型交叉熵 + 锚点损失
  `L_proto + lambda_target_anchor * L_anchor`，其中锚点损失让适配器贴近其「源初始化」，
  以抵抗过拟合。

推理评估用 **1-NN（余弦）** 或 **原型** 分类器在学到的目标特征上进行。评价指标：
**OA**、**AA**、**Kappa**。

## 5. 消融实验（V9）

批处理脚本 `run-y.sh` 跑一组消融实验。核心集合：

| 消融 | 含义 |
|------|------|
| `full` | 所有组件启用 |
| `no_dictionary` | 关闭语义字典 |
| `no_supcon` | 关闭有监督对比损失 |
| `random_target_init` | 目标适配器随机初始化（非源初始化） |
| `no_target_tune` | 完全跳过阶段二（只用源流形） |
| `no_target_anchor` | 关闭锚点正则（`--sm_lambda_target_anchor 0`） |
| `ce_only` | 同时关闭字典和 SupCon（纯交叉熵） |

扩展集合（字典正则化的细粒度消融）：

| 消融 | 含义 |
|------|------|
| `no_dict_diversity` | `--sm_lambda_dict_diversity 0` |
| `no_dict_selectivity` | `--sm_lambda_dict_selectivity 0` |
| `no_dict_balance` | `--sm_lambda_dict_balance 0` |
| `no_dict_regularization` | 三个字典损失全关 |

## 6. 数据集

- **源域**（默认 `--source 0`，定义于 `parameters.py::ChooseSource`）：
  `PaviaU`、`Chikusei`、`IndianPines`、`WHU-Hi-HanChuan`（波段数异质）。
  `ChooseSource` 还定义了单源 / 双源 / 三源的预设（索引 1–14），用于源域消融。
- **目标域**（在 `run-y.sh` 中运行）：`Salinas`、`NC12`、`WHU-Hi-LongKou`。
- 数据从 `/home/t1/DL_Code/DATASET/<name>/` 通过 `test/dataset.py`（`get_dataset`）加载，
  支持约 25 个高光谱数据集（AVIRIS、WHU-Hi、Houston、黄河口 NC 系列、雄安等）。

## 7. 目录 / 文件结构

```
newdemo-dict/
├── run-y.sh                     # 批量消融运行脚本（core + extended 集合）
├── test/
│   ├── OUR-Y.py                 # 主训练/评估入口（精简版）
│   ├── OUR-Y-draw.py            # 完整版：增加可视化 + 过拟合诊断
│   ├── parameters.py            # 参数解析 + ChooseSource 源域组合预设
│   └── dataset.py               # 高光谱数据集加载（get_dataset, HyperX, ...）
├── utils/
│   ├── __init__.py
│   ├── run.py                   # 指标 / 检查点 / 预测图辅助函数
│   ├── dataset.py               # 底层 IO + 采样辅助（sample_gt, build_dataset）
│   ├── dim_reduction.py         # PCA / NMF / TruncatedSVD 封装
│   └── display.py               # 颜色图、t-SNE、混淆矩阵显示、指标
└── batch_results_v9/
    ├── manifest.txt             # 批量运行日志
    ├── ablation_summary.csv     # 每个 target × ablation 的聚合 OA/AA/Kappa
    └── <Target>/<Experiment>/   # per_run_metrics.csv, summary.txt, seed_*/visualizations/
```

存在两个主脚本变体：
- `OUR-Y.py` — 紧凑的训练 + 评估 + 可选分类图 / t-SNE 图。
- `OUR-Y-draw.py` — 模型相同，额外增加了逐 step 追踪：源损失曲线、目标过拟合诊断
  （LOO-1NN 训练精度 vs 验证精度、泛化差距、适配器参数移动）、语义原子使用热力图，
  以及每类原子分布一致性。

## 8. 结果（来自 `batch_results_v9`）

5 次运行聚合（OA% / AA% / Kappa），核心消融：

| Target | full | no_dictionary | no_supcon | random_target_init | no_target_tune |
|--------|------|---------------|-----------|--------------------|----------------|
| Salinas | 92.94 / 95.50 / 0.922 | 92.38 / 95.40 / 0.915 | 92.56 / 95.44 / 0.917 | 91.30 / 94.97 / 0.903 | 78.57 / 86.74 / 0.763 |
| NC12 | 87.91 / 91.13 / 0.853 | 87.12 / 91.21 / 0.844 | 87.12 / 91.00 / 0.844 | 86.43 / 90.96 / 0.835 | 75.92 / 79.22 / 0.713 |
| WHU-Hi-LongKou | 92.44 / 90.79 / 0.902 | 93.34 / 91.20 / 0.914 | 93.21 / 91.07 / 0.912 | 92.59 / 90.02 / 0.904 | 81.67 / 77.73 / 0.767 |

观察：`no_target_tune`（跳过阶段二）性能大幅下降（如 Salinas OA 92.94 → 78.57），
确认目标适配器微调是必需的。`full` 模型整体最优或具有竞争力；字典项和 SupCon 项在
大多数目标上带来小幅但稳定的提升。

## 9. 如何运行

单次实验（在 `test/` 目录内）：

```bash
python OUR-Y.py --source 0 --target Salinas --n_run 5 --epochs 2000
```

消融批处理（所有 target × 所有实验）：

```bash
bash run-y.sh                     # 核心集合
SET=extended bash run-y.sh        # + 细粒度字典消融
```

输出写入 `./checkpoints_v9_semantic_manifold/<target>/<ablation>/`；批处理模式下，
结果归档到 `./batch_results_v9/<target>/<experiment>/`。
