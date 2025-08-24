# 模型训练报告

## 1. 摘要

- **数据集**: `results-v4.csv`
- **训练样本总数**: 5
- **描述符**: MACCS Keys + Gasteiger Charges + MolMR + MMFF Properties
- **报告生成时间**: 2025-08-17 16:12:53

---

## 2. 训练配置

| 参数 | 值 |
| :--- | :--- |
| `hidden_sizes` | [256, 128, 64] |
| `dropout_rate` | 0.3 |
| `learning_rate` | 0.001 |
| `batch_size` | 64 |
| `epochs` | 300 |
| `early_stopping_patience` | 100 |

---

## 3. 模型评估结果

### 总体性能

| 指标 | 值 |
| :--- | :--- |
| **MSE** | 0.025158 |
| **MAE** | 0.151281 |
| **R² Score** | nan |

### 参数Q性能

| 指标 | 值 |
| :--- | :--- |
| **MSE** | 0.039580 |
| **MAE** | 0.198948 |
| **R² Score** | nan |

### 参数e性能

| 指标 | 值 |
| :--- | :--- |
| **MSE** | 0.010736 |
| **MAE** | 0.103615 |
| **R² Score** | nan |

---

## 4. 训练可视化

![训练结果图](training_results.png)
