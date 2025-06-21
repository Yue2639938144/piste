# TCR-HLA-Pep 三元互作预测器

TCR-HLA-Pep预测器是一个基于深度学习的计算工具，用于预测T细胞受体(TCR)、人类白细胞抗原(HLA)和抗原肽段(Peptide)之间的三元互作关系。该工具采用创新的模型架构，融合了物理滑动注意力机制和数据驱动注意力机制，能够准确预测TCR-HLA-Pep三元复合物的形成概率，并提供详细的残基互作可视化。

## 目录

- [特性](#特性)
- [系统架构](#系统架构)
- [安装](#安装)
- [使用方法](#使用方法)
- [数据格式](#数据格式)
- [模型架构](#模型架构)
- [训练流程](#训练流程)
- [可视化](#可视化)
- [问题修复](#问题修复)
- [引用](#引用)
- [许可证](#许可证)
- [联系方式](#联系方式)

## 特性

- **全长序列输入**：直接使用TCR、HLA和肽段的全长氨基酸序列作为输入，无需伪序列编码
- **双重注意力机制**：融合物理滑动注意力和数据驱动注意力，提高预测准确性
- **二元模型预训练**：将三元互作拆分为TCR-Pep和HLA-Pep两个二元互作进行预训练
- **联合优化策略**：通过三元组数据反向优化二元组模型，实现整体性能提升
- **残基互作可视化**：提供直观的热力图和交互式可视化，展示关键残基互作位点
- **易用命令行工具**：简洁的命令行接口，支持训练、评估、预测和可视化功能

## 系统架构

TCR-HLA-Pep预测器采用模块化设计，主要包含以下组件：

```
tcr_hla_pep_predictor/
├── src/                      # 源代码目录
│   ├── data/                 # 数据处理模块
│   │   ├── data_processor.py # 数据预处理核心逻辑
│   │   ├── dataloader.py     # 数据加载器
│   │   ├── preprocessing.py  # 序列预处理
│   │   └── clustering.py     # 序列聚类
│   ├── models/               # 模型定义
│   │   ├── tcr_pep_model.py  # TCR-Pep二元模型
│   │   ├── hla_pep_model.py  # HLA-Pep二元模型
│   │   └── trimer_model.py   # TCR-HLA-Pep三元模型
│   └── utils/                # 工具函数
│       ├── config.py         # 配置管理
│       ├── logger.py         # 日志工具
│       ├── metrics.py        # 评估指标
│       └── visualization.py  # 可视化工具
├── scripts/                  # 脚本文件
│   ├── preprocess_data.py    # 数据预处理脚本
│   ├── train_tcr_pep.py      # TCR-Pep模型训练脚本
│   ├── train_hla_pep.py      # HLA-Pep模型训练脚本
│   ├── train_trimer.py       # 三元模型训练脚本
│   ├── evaluate_model.py     # 模型评估脚本
│   └── predict_and_visualize.py # 预测和可视化脚本
├── configs/                  # 配置文件
│   └── default_config.yaml   # 默认配置
├── data/                     # 数据目录
│   ├── raw/                  # 原始数据
│   └── processed/            # 处理后的数据
├── models/                   # 模型保存目录
├── logs/                     # 日志目录
├── results/                  # 结果输出目录
└── cli.py                    # 命令行接口
```

### 数据流程

1. **数据采集**：从实验或公共数据库获取TCR、HLA和肽段序列数据
2. **数据预处理**：清洗、标准化和格式化原始数据
3. **数据拆分**：将数据分为训练集、验证集和测试集
4. **特征提取**：将氨基酸序列转换为数值表示，并添加生化特征
5. **模型训练**：先训练二元模型，再训练三元模型
6. **模型评估**：在测试集上评估模型性能
7. **预测和可视化**：使用训练好的模型进行预测并生成可视化结果

## 安装

### 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA 10.2+ (GPU加速，可选)

### 安装步骤

1. 克隆代码库：

```bash
git clone https://github.com/yourusername/tcr-hla-pep-predictor.git
cd tcr-hla-pep-predictor
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 设置环境变量（可选）：

```bash
# Linux/Mac
export PYTHONPATH=$PYTHONPATH:$(pwd)
export KMP_DUPLICATE_LIB_OK=TRUE  # 解决OpenMP初始化问题

# Windows
set PYTHONPATH=%PYTHONPATH%;%cd%
set KMP_DUPLICATE_LIB_OK=TRUE
```

## 使用方法

TCR-HLA-Pep预测器提供了统一的命令行接口，支持数据预处理、模型训练、评估和预测功能。

### 数据预处理

将原始数据转换为模型可用的格式，并可选择拆分为训练、验证和测试集。

```bash
# 处理TCR-Pep二元数据
python cli.py preprocess --data_dir data/raw/tcr_pep --output_dir data/processed --mode tcr_pep --negative_ratio 1.0

# 处理HLA-Pep二元数据
python cli.py preprocess --data_dir data/raw/hla_pep --output_dir data/processed --mode hla_pep --negative_ratio 1.0

# 处理三元数据
python cli.py preprocess --data_dir data/raw/trimer --output_dir data/processed --mode trimer --negative_ratio 1.0
```

### 模型训练

训练模型有三种模式：TCR-Pep二元模型、HLA-Pep二元模型和TCR-HLA-Pep三元模型。

1. 训练TCR-Pep二元模型：

```bash
python cli.py train --mode tcr_pep --train_data data/processed/tcr_pep/train.csv \
                   --val_data data/processed/tcr_pep/val.csv \
                   --output_dir models/tcr_pep --early_stopping
```

2. 训练HLA-Pep二元模型：

```bash
python cli.py train --mode hla_pep --train_data data/processed/hla_pep/train.csv \
                   --val_data data/processed/hla_pep/val.csv \
                   --output_dir models/hla_pep --early_stopping
```

3. 训练三元模型（使用预训练的二元模型）：

```bash
python cli.py train --mode trimer --train_data data/processed/trimer/train.csv \
                   --val_data data/processed/trimer/val.csv \
                   --pretrained_model models/tcr_pep/best_model.pt,models/hla_pep/best_model.pt \
                   --output_dir models/trimer --early_stopping --joint_optimization
```

### 模型评估

在测试集上评估模型性能，生成详细的评估报告和可视化图表。

```bash
python cli.py evaluate --model models/trimer/best_model.pt \
                      --test_data data/processed/trimer/test.csv \
                      --output_dir evaluation
```

### 预测和可视化

使用训练好的模型进行预测，并可选择生成残基互作可视化。

```bash
python cli.py predict --model models/trimer/best_model.pt \
                     --input data/new_samples.csv \
                     --output_dir results \
                     --visualize --interactive
```

## 数据格式

### 输入数据格式

TCR-HLA-Pep预测器接受CSV格式的输入文件，包含以下列：

#### 三元模型数据格式

- `CDR3`: TCR CDR3β序列
- `HLA`: HLA标识符（如HLA-A*02:01）
- `MT_pep`: 抗原肽段序列
- `Label`: 标签（1表示结合，0表示不结合，仅训练和评估需要）

示例：

```
CDR3,HLA,MT_pep,Label
CASSQDLNRGYTF,HLA-A*02:01,NLVPMVATV,1
CASSLGQAYEQYF,HLA-A*02:01,GILGFVFTL,0
```

#### 二元模型数据格式

TCR-Pep模型：

```
CDR3,MT_pep,Label
CASSQDLNRGYTF,NLVPMVATV,1
CASSLGQAYEQYF,GILGFVFTL,0
```

HLA-Pep模型：

```
HLA,MT_pep,Label
HLA-A*02:01,NLVPMVATV,1
HLA-A*02:01,GILGFVFTL,0
```

### HLA序列数据

HLA序列数据需要单独提供，格式如下：

```
HLA,pseudo_sequence
HLA-A*02:01,YFAMYQENMAHTDANTLYIIYRDYTWAAQAYRWYITAYLEYAAFTYLEGRCVEWLRRYLENGKETLQRA
```

### 目录结构要求

数据目录结构需要按照以下格式组织：

```
data/
├── raw/
│   ├── tcr_pep/
│   │   ├── pos/         # 阳性样本
│   │   └── neg/         # 阴性样本
│   ├── hla_pep/
│   │   ├── pos/
│   │   └── neg/
│   └── trimer/
│       ├── pos/
│       ├── neg/
│       └── hla_sequences.csv  # HLA序列数据
└── processed/           # 预处理后的数据
```

## 模型架构

TCR-HLA-Pep预测器的核心是一个基于注意力机制的深度学习模型，包括以下组件：

### 1. 序列编码层

- **氨基酸嵌入**：将氨基酸序列转换为低维嵌入向量
- **位置编码**：添加位置信息，使模型能够区分序列中的位置
- **生化特征融合**：结合氨基酸的生化特性（如疏水性、电荷等）

### 2. 注意力机制

#### 物理滑动注意力

基于氨基酸残基在空间中的相对位置建模互作关系，具有以下特点：
- 考虑空间邻近性
- 模拟分子对接过程
- 关注局部互作模式

#### 数据驱动注意力

通过自注意力机制学习序列内部和序列间的依赖关系：
- 捕捉全局上下文
- 学习复杂的序列模式
- 自适应权重分配

#### 融合注意力

将物理滑动注意力和数据驱动注意力进行加权融合：
- 结合两种注意力机制的优势
- 自适应调整权重
- 提高预测准确性

### 3. 二元互作编码器

分别对TCR-Pep和HLA-Pep的互作进行编码：
- TCR-Pep编码器：捕捉TCR与肽段的结合模式
- HLA-Pep编码器：捕捉HLA与肽段的结合模式

### 4. 三元互作整合器

整合二元互作特征，预测三元复合物形成概率：
- 特征融合层：整合二元互作特征
- 交叉注意力层：建模三元组件间的相互影响
- 预测头：输出结合概率

### 模型图示

```
输入序列
  │
  ├─── TCR序列 ─── 嵌入层 ─┐
  │                       │
  ├─── 肽段序列 ─── 嵌入层 ─┼─── TCR-Pep
  │                       │    二元编码器 ─┐
  │                       │                │
  ├─── HLA序列 ─── 嵌入层 ─┤                │
  │                       │                │
  │                       │                ├─── 三元互作 ─── 预测
  │                       │                │     整合器
  │                       │                │
  │                       │    HLA-Pep     │
  │                       └─── 二元编码器 ─┘
  │
生化特征
```

## 训练流程

TCR-HLA-Pep预测器采用多阶段训练策略：

### 1. 二元模型预训练

首先分别训练TCR-Pep和HLA-Pep二元模型：
- 使用二元互作数据
- 优化二元预测性能
- 学习基础互作模式

### 2. 三元模型训练

使用预训练的二元模型初始化三元模型：
- 加载预训练的二元模型权重
- 冻结二元模型参数
- 训练三元互作整合器

### 3. 联合优化

最后进行端到端的联合优化：
- 解冻二元模型参数
- 使用三元数据反向优化
- 联合损失函数指导训练

### 训练参数

主要训练参数包括：
- 批大小：64
- 学习率：1e-3（初始）
- 优化器：Adam
- 学习率调度：ReduceLROnPlateau
- 早停策略：验证集F1分数不提升
- 训练轮数：最多200轮

## 可视化

TCR-HLA-Pep预测器提供多种可视化方式：

### 1. 残基互作热力图

展示TCR、HLA和肽段之间的残基互作强度：
- TCR-Pep互作热力图
- HLA-Pep互作热力图
- TCR-HLA互作热力图

### 2. 注意力权重可视化

展示模型注意力机制的关注点：
- 物理滑动注意力权重
- 数据驱动注意力权重
- 融合注意力权重

### 3. 互作网络图

展示氨基酸残基之间的互作网络：
- 节点：氨基酸残基
- 边：互作强度
- 颜色：残基类型

### 4. 交互式3D可视化

结合结构预测，提供交互式3D可视化：
- 三元复合物结构
- 互作界面高亮
- 关键残基标注

## 问题修复

### 三元组处理代码错误："不支持的二元类型: trimer"

**问题描述**：
在处理三元组数据时，`load_binary_data`函数不支持'trimer'类型，导致报错。

**解决方案**：
1. 修改了`load_binary_data`函数，明确将'trimer'添加为支持的类型
2. 增强了`check_data_format`函数，添加对'trimer'类型的特殊处理，特别是对HLA序列的验证

### 训练二元模型时的错误

**问题描述**：
训练二元模型时遇到数据处理和目录结构问题。

**解决方案**：
1. 增强了`preprocess_data`函数的错误处理和日志记录
2. 添加了对简化trimer目录结构的支持
3. 添加了自动创建HLA序列文件的功能
4. 添加了手动创建处理后的三元模型数据文件的功能

### OpenMP初始化问题

**问题描述**：
运行时出现OpenMP初始化错误：
```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

**解决方案**：
1. 在训练脚本中添加了环境变量设置：`os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"`
2. 在命令行运行时也需要设置环境变量：`$env:KMP_DUPLICATE_LIB_OK="TRUE";`

### 数据文件路径问题

**问题描述**：
训练命令中使用的文件路径与实际生成的文件路径不一致。

**解决方案**：
修正了训练命令中的文件路径，使用正确的子目录结构：
```
tcr_hla_pep_predictor/data/processed/tcr_pep/train.csv
```
而不是：
```
tcr_hla_pep_predictor/data/processed/tcr_pep_train.csv
```

### 注意事项

尽管做了这些修改，训练命令仍未成功执行，可能需要进一步调试模型训练脚本。可能的原因包括：

1. 数据集太小，不足以训练模型
2. 模型参数配置不当
3. 训练脚本中可能存在其他错误

建议下一步：
1. 检查训练日志文件
2. 尝试使用更大的数据集
3. 调整模型参数
4. 逐步调试训练脚本

## 引用

如果您在研究中使用了TCR-HLA-Pep预测器，请引用我们的论文：

```
待发表
```

## 许可证

本项目采用MIT许可证。详情请参阅[LICENSE](LICENSE)文件。

## 联系方式

如有问题或建议，请通过以下方式联系我们：

- 电子邮件：2639938144@qq.com
- GitHub Issues：https://github.com/Yue2639938144/piste/issues