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

TCR-HLA-Pep预测器提供了统一的命令行接口，支持数据预处理、模型训练、评估和预测功能。所有功能都可以通过`cli.py`主程序访问。

### 数据准备

在开始使用预测器之前，您需要准备好数据并按照[数据格式](#数据格式)部分所述的结构组织数据。

#### 数据目录结构设置

首先，创建必要的数据目录结构：

```bash
# 创建数据目录结构
mkdir -p data/raw/tcr_pep/{pos,neg}
mkdir -p data/raw/hla_pep/{pos,neg}
mkdir -p data/raw/trimer/{pos,neg}
mkdir -p data/processed
```

#### 准备示例数据

您可以使用以下命令生成示例数据文件：

```bash
# 生成TCR-Pep示例数据
echo "CDR3,MT_pep" > data/raw/tcr_pep/pos/pos_data.csv
echo "CASSLAPGATNEKLFF,GILGFVFTL" >> data/raw/tcr_pep/pos/pos_data.csv

echo "CDR3,MT_pep" > data/raw/tcr_pep/neg/neg_data.csv
echo "CASSLTNSGNTLYF,GILGFVFTL" >> data/raw/tcr_pep/neg/neg_data.csv

# 生成HLA-Pep示例数据
echo "HLA,MT_pep" > data/raw/hla_pep/pos/pos_data.csv
echo "HLA-A*02:01,GILGFVFTL" >> data/raw/hla_pep/pos/pos_data.csv

echo "HLA,MT_pep" > data/raw/hla_pep/neg/neg_data.csv
echo "HLA-A*02:01,KLVALGINAV" >> data/raw/hla_pep/neg/neg_data.csv

# 生成HLA序列数据
echo "HLA,pseudo_sequence" > data/raw/hla_pep/hla_sequences.csv
echo "HLA-A*02:01,YFAMYQENMAHTDANTLYIIYRDYTWAAQAYRWYITAYLEYAAFTYLEGRCVEWLRRYLENGKETLQRA" >> data/raw/hla_pep/hla_sequences.csv

# 生成三元组示例数据
echo "CDR3,HLA,MT_pep" > data/raw/trimer/pos/pos_data.csv
echo "CASSLAPGATNEKLFF,HLA-A*02:01,GILGFVFTL" >> data/raw/trimer/pos/pos_data.csv

echo "CDR3,HLA,MT_pep" > data/raw/trimer/neg/neg_data.csv
echo "CASSLAPGATNEKLFF,HLA-A*02:01,KLVALGINAV" >> data/raw/trimer/neg/neg_data.csv

cp data/raw/hla_pep/hla_sequences.csv data/raw/trimer/
```

### 数据预处理

数据预处理是模型训练的第一步，将原始数据转换为模型可用的格式，并拆分为训练、验证和测试集。

#### 预处理参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | 原始数据目录路径 | 必填 |
| `--output_dir` | 处理后数据的输出目录 | 必填 |
| `--mode` | 数据模式，可选值：`tcr_pep`、`hla_pep`、`trimer` | 必填 |
| `--negative_ratio` | 负样本与正样本的比例 | 1.0 |
| `--train_ratio` | 训练集比例 | 0.7 |
| `--val_ratio` | 验证集比例 | 0.15 |
| `--test_ratio` | 测试集比例 | 0.15 |
| `--random_seed` | 随机种子，用于数据拆分的可重复性 | 42 |
| `--cluster_tcrs` | 是否对TCR序列进行聚类以减少序列相似性 | False |
| `--cluster_threshold` | TCR聚类阈值 | 0.8 |

#### 预处理命令示例

```bash
# 处理TCR-Pep二元数据
python cli.py preprocess --data_dir data/raw/tcr_pep --output_dir data/processed --mode tcr_pep --negative_ratio 1.0 --random_seed 42

# 处理HLA-Pep二元数据
python cli.py preprocess --data_dir data/raw/hla_pep --output_dir data/processed --mode hla_pep --negative_ratio 1.0 --random_seed 42

# 处理三元数据
python cli.py preprocess --data_dir data/raw/trimer --output_dir data/processed --mode trimer --negative_ratio 1.0 --random_seed 42
```

#### 预处理注意事项

1. **数据格式检查**：预处理前确保数据格式正确，特别是列名和文件结构
2. **数据平衡**：通过`--negative_ratio`参数调整正负样本比例
3. **数据拆分**：确保训练、验证和测试集的比例之和为1.0
4. **HLA序列文件**：对于`hla_pep`和`trimer`模式，确保`hla_sequences.csv`文件存在且包含所有HLA的序列信息
5. **错误处理**：如果预处理过程中出现错误，请查看日志文件了解详细信息

### 模型训练

训练模型有三种模式：TCR-Pep二元模型、HLA-Pep二元模型和TCR-HLA-Pep三元模型。建议先训练二元模型，然后使用预训练的二元模型初始化三元模型。

#### 训练参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 训练模式，可选值：`tcr_pep`、`hla_pep`、`trimer` | 必填 |
| `--train_data` | 训练数据文件路径 | 必填 |
| `--val_data` | 验证数据文件路径 | 必填 |
| `--output_dir` | 模型输出目录 | 必填 |
| `--pretrained_model` | 预训练模型路径，用于三元模型训练 | 无 |
| `--batch_size` | 批大小 | 64 |
| `--epochs` | 训练轮数 | 200 |
| `--learning_rate` | 学习率 | 0.001 |
| `--early_stopping` | 是否启用早停策略 | False |
| `--patience` | 早停策略的耐心值 | 10 |
| `--joint_optimization` | 是否进行联合优化（仅三元模型） | False |
| `--embedding_dim` | 嵌入维度 | 128 |
| `--hidden_dim` | 隐藏层维度 | 256 |
| `--dropout` | Dropout比率 | 0.2 |
| `--weight_decay` | 权重衰减系数 | 1e-4 |
| `--attention_heads` | 注意力头数量 | 8 |
| `--save_checkpoint` | 是否保存检查点 | True |
| `--checkpoint_freq` | 检查点保存频率（轮数） | 10 |

#### 训练命令示例

1. 训练TCR-Pep二元模型：

```bash
# 在Windows系统中设置环境变量解决OpenMP错误
$env:KMP_DUPLICATE_LIB_OK="TRUE";

# 训练TCR-Pep二元模型
python cli.py train --mode tcr_pep \
                   --train_data data/processed/tcr_pep/train.csv \
                   --val_data data/processed/tcr_pep/val.csv \
                   --output_dir models/tcr_pep \
                   --batch_size 32 \
                   --learning_rate 0.001 \
                   --early_stopping \
                   --patience 15
```

2. 训练HLA-Pep二元模型：

```bash
# 训练HLA-Pep二元模型
python cli.py train --mode hla_pep \
                   --train_data data/processed/hla_pep/train.csv \
                   --val_data data/processed/hla_pep/val.csv \
                   --output_dir models/hla_pep \
                   --batch_size 32 \
                   --learning_rate 0.001 \
                   --early_stopping \
                   --patience 15
```

3. 训练三元模型（使用预训练的二元模型）：

```bash
# 训练三元模型
python cli.py train --mode trimer \
                   --train_data data/processed/trimer/train.csv \
                   --val_data data/processed/trimer/val.csv \
                   --pretrained_model models/tcr_pep/best_model.pt,models/hla_pep/best_model.pt \
                   --output_dir models/trimer \
                   --batch_size 16 \
                   --learning_rate 0.0005 \
                   --early_stopping \
                   --patience 20 \
                   --joint_optimization
```

#### 训练注意事项

1. **GPU加速**：如果有GPU，系统会自动使用；如果没有，将使用CPU训练（速度较慢）
2. **内存使用**：大数据集训练时，可能需要调整`batch_size`以适应内存限制
3. **预训练模型**：三元模型训练时，确保提供正确的预训练二元模型路径，用逗号分隔
4. **联合优化**：使用`--joint_optimization`参数可以在三元模型训练过程中同时优化二元模型
5. **训练监控**：训练过程中会输出损失和评估指标，可以通过日志文件监控训练进度
6. **模型保存**：训练结束后，最佳模型会保存在指定的输出目录中

### 模型评估

在测试集上评估模型性能，生成详细的评估报告和可视化图表。

#### 评估参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型文件路径 | 必填 |
| `--test_data` | 测试数据文件路径 | 必填 |
| `--output_dir` | 评估结果输出目录 | 必填 |
| `--batch_size` | 批大小 | 64 |
| `--detailed_report` | 是否生成详细报告 | False |
| `--visualize` | 是否生成可视化结果 | False |

#### 评估命令示例

```bash
# 评估三元模型
python cli.py evaluate --model models/trimer/best_model.pt \
                      --test_data data/processed/trimer/test.csv \
                      --output_dir evaluation \
                      --batch_size 32 \
                      --detailed_report \
                      --visualize
```

#### 评估注意事项

1. **评估指标**：评估结果包括准确率、精确率、召回率、F1分数、AUC-ROC等
2. **混淆矩阵**：使用`--detailed_report`参数可生成混淆矩阵和分类报告
3. **阈值调整**：可以通过调整决策阈值优化模型性能
4. **可视化**：使用`--visualize`参数可生成ROC曲线、PR曲线等可视化结果

### 预测和可视化

使用训练好的模型对新数据进行预测，并可选择生成残基互作可视化。

#### 预测参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型文件路径 | 必填 |
| `--input` | 输入数据文件路径 | 必填 |
| `--output_dir` | 预测结果输出目录 | 必填 |
| `--batch_size` | 批大小 | 64 |
| `--threshold` | 预测阈值 | 0.5 |
| `--visualize` | 是否生成可视化结果 | False |
| `--interactive` | 是否生成交互式可视化 | False |
| `--top_k` | 显示前k个最强互作 | 10 |

#### 预测数据格式

预测数据格式与训练数据相似，但不需要`Label`列：

```
CDR3,HLA,MT_pep
CASSLAPGATNEKLFF,HLA-A*02:01,GILGFVFTL
CASSLTNSGNTLYF,HLA-A*02:01,NLVPMVATV
```

#### 预测命令示例

```bash
# 预测并生成可视化结果
python cli.py predict --model models/trimer/best_model.pt \
                     --input data/new_samples.csv \
                     --output_dir results \
                     --threshold 0.6 \
                     --visualize \
                     --top_k 5
```

#### 交互式可视化

如果需要生成交互式可视化，可以使用`--interactive`参数：

```bash
# 生成交互式可视化
python cli.py predict --model models/trimer/best_model.pt \
                     --input data/new_samples.csv \
                     --output_dir results \
                     --visualize \
                     --interactive
```

交互式可视化会生成HTML文件，可以在浏览器中打开查看。

#### 批量预测

对大量数据进行批量预测：

```bash
# 批量预测
python cli.py predict --model models/trimer/best_model.pt \
                     --input data/large_dataset.csv \
                     --output_dir results/batch_prediction \
                     --batch_size 128
```

#### 预测注意事项

1. **输入格式**：确保输入数据格式正确，列名与训练数据一致
2. **阈值调整**：可以通过`--threshold`参数调整预测阈值，影响预测结果
3. **可视化选项**：
   - `--visualize`：生成静态可视化图表
   - `--interactive`：生成交互式可视化（需要额外的依赖）
4. **结果保存**：预测结果会保存为CSV文件，可视化结果保存为PNG或HTML文件
5. **内存使用**：处理大量数据时，可能需要调整`batch_size`以适应内存限制

### 高级用法

#### 配置文件

除了命令行参数外，还可以使用配置文件指定参数：

```bash
# 使用配置文件
python cli.py train --config configs/my_config.yaml
```

配置文件示例（YAML格式）：

```yaml
mode: trimer
train_data: data/processed/trimer/train.csv
val_data: data/processed/trimer/val.csv
pretrained_model: 
  - models/tcr_pep/best_model.pt
  - models/hla_pep/best_model.pt
output_dir: models/trimer
batch_size: 32
learning_rate: 0.0005
early_stopping: true
patience: 15
joint_optimization: true
```

#### 模型调试

在训练过程中启用调试模式，输出更详细的信息：

```bash
# 启用调试模式
python cli.py train --mode tcr_pep --train_data data/processed/tcr_pep/train.csv --val_data data/processed/tcr_pep/val.csv --output_dir models/tcr_pep --debug
```

#### 模型导出

将训练好的模型导出为ONNX格式，便于部署：

```bash
# 导出模型
python cli.py export --model models/trimer/best_model.pt --output models/trimer/model.onnx
```

#### 自定义数据处理

对数据进行自定义预处理：

```bash
# 自定义数据处理
python cli.py preprocess --data_dir data/raw/custom --output_dir data/processed/custom --mode custom --custom_config configs/custom_preprocessing.yaml
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