# TCR-HLA-Pep 三元互作预测器

TCR-HLA-Pep预测器是一个基于深度学习的计算工具，用于预测T细胞受体(TCR)、人类白细胞抗原(HLA)和抗原肽段(Peptide)之间的三元互作关系。该工具采用创新的模型架构，融合了物理滑动注意力机制和数据驱动注意力机制，能够准确预测TCR-HLA-Pep三元复合物的形成概率，并提供详细的残基互作可视化。

## 特性

- **全长序列输入**：直接使用TCR、HLA和肽段的全长氨基酸序列作为输入，无需伪序列编码
- **双重注意力机制**：融合物理滑动注意力和数据驱动注意力，提高预测准确性
- **二元模型预训练**：将三元互作拆分为TCR-Pep和HLA-Pep两个二元互作进行预训练
- **联合优化策略**：通过三元组数据反向优化二元组模型，实现整体性能提升
- **残基互作可视化**：提供直观的热力图和交互式可视化，展示关键残基互作位点
- **易用命令行工具**：简洁的命令行接口，支持训练、评估、预测和可视化功能
- **灵活数据输入**：支持多种数据输入模式，包括二元模型和三元模型数据
- **数据格式验证**：自动检测数据格式错误，确保模型输入的准确性

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
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 使用方法

TCR-HLA-Pep预测器提供了统一的命令行接口，支持数据预处理、模型训练、评估和预测功能。

### 数据预处理

支持多种数据输入模式，可自动验证数据格式并处理为模型可用的格式。

#### 数据输入模式

1. **二元模型数据**：输入包含阳性和阴性集合的文件夹，支持调整阴性集合倍数

```bash
python cli.py preprocess --data_dir data/tcr_pep --output_dir data/processed --mode tcr_pep --negative_ratio 2.0 --split
```

2. **三元模型数据**：输入包含3对阳性阴性集合的文件夹，分别用于训练不同类型的模型

```bash
python cli.py preprocess --data_dir data/trimer_data --output_dir data/processed --mode trimer --split
```

#### 数据格式验证

系统会自动检测并验证以下数据格式：

- **肽段序列**：长度9-12的氨基酸序列
- **TCR序列**：30氨基酸以内的CDR3区序列
- **HLA格式**：形如HLA-A01:02的标准HLA格式

如果检测到格式错误，系统会记录错误行并在日志中输出错误信息。

### 模型训练

训练模型有三种模式：TCR-Pep二元模型、HLA-Pep二元模型和TCR-HLA-Pep三元模型。

1. 训练TCR-Pep二元模型：

```bash
python cli.py train --mode tcr_pep --train_data data/processed/tcr_pep_train.csv \
                   --val_data data/processed/tcr_pep_val.csv \
                   --output_dir models/tcr_pep --early_stopping
```

2. 训练HLA-Pep二元模型：

```bash
python cli.py train --mode hla_pep --train_data data/processed/hla_pep_train.csv \
                   --val_data data/processed/hla_pep_val.csv \
                   --output_dir models/hla_pep --early_stopping
```

3. 训练三元模型（使用预训练的二元模型）：

```bash
python cli.py train --mode trimer --train_data data/processed/trimer_train.csv \
                   --val_data data/processed/trimer_val.csv \
                   --pretrained_model models/tcr_pep/best_model.pt,models/hla_pep/best_model.pt \
                   --output_dir models/trimer --early_stopping --joint_optimization
```

### 模型评估

在测试集上评估模型性能，生成详细的评估报告和可视化图表。

```bash
python cli.py evaluate --model models/trimer/best_model.pt \
                      --test_data data/processed/trimer_test.csv \
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

## 示例

### 输入数据格式

#### 单文件输入格式

TCR-HLA-Pep预测器接受CSV格式的输入文件，包含以下列：

- `tcr_seq`/`CDR3`: TCR CDR3β序列
- `hla_seq`/`HLA`: HLA序列或标识符
- `pep_seq`/`MT_pep`: 抗原肽段序列
- `label`/`Label`: 标签（1表示结合，0表示不结合，仅训练和评估需要）

示例：

```
CDR3,MT_pep,Label
CASSQDLNRGYTF,NLVPMVATV,1
CASSLGQAYEQYF,GILGFVFTL,0
...
```

#### 文件夹结构输入格式

二元模型文件夹结构：
```
data/tcr_pep/
├── pos/                   # 阳性样本文件夹
│   └── positive_samples.csv
└── neg/                   # 阴性样本文件夹
    └── negative_samples.csv
```

三元模型文件夹结构：
```
data/trimer_data/
├── tcr_pep/               # TCR-Pep数据文件夹
│   ├── pos/
│   └── neg/
├── hla_pep/               # HLA-Pep数据文件夹
│   ├── pos/
│   └── neg/
└── trimer/                # TCR-HLA-Pep数据文件夹
    ├── pos/
    └── neg/
```

### 可视化输出

预测命令会生成以下可视化结果：

1. TCR-Pep互作热力图
2. HLA-Pep互作热力图
3. TCR-HLA互作热力图
4. 三元互作综合可视化（交互式HTML）
5. 关键互作残基对列表（CSV格式）

## 模型架构

TCR-HLA-Pep预测器的核心是一个基于注意力机制的深度学习模型，包括以下组件：

1. **序列嵌入层**：将氨基酸序列转换为嵌入向量，融合位置编码和生化特征
2. **物理滑动注意力**：基于氨基酸残基在空间中的相对位置建模互作关系
3. **数据驱动注意力**：通过自注意力机制学习序列内部和序列间的依赖关系
4. **融合注意力**：将物理滑动注意力和数据驱动注意力进行加权融合
5. **二元互作编码器**：分别编码TCR-Pep和HLA-Pep的互作特征
6. **三元互作整合器**：整合二元互作特征，预测三元复合物形成概率

## 引用

如果您在研究中使用了TCR-HLA-Pep预测器，请引用我们的论文：

```
待发表
```

## 许可证

本项目采用MIT许可证。详情请参阅[LICENSE](LICENSE)文件。

## 联系方式

如有问题或建议，请通过以下方式联系我们：

- 电子邮件：your.email@example.com
- GitHub Issues：https://github.com/yourusername/tcr-hla-pep-predictor/issues 