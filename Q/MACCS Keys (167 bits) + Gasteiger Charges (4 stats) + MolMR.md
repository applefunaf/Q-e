# 模型训练报告 (Q值预测)

## 1. 摘要
- **数据集**: `results-v4.csv`
- **训练目标**: Q值
- **训练样本总数**: 255
- **描述符**: MACCS Keys (167 bits) + Gasteiger Charges (4 stats) + MolMR
- **报告生成时间**: 2025-08-17 16:50:03

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
## 3. 模型评估结果 (Q值)
| 指标 | 值 |
| :--- | :--- |
| **MSE** | 2.052074 |
| **MAE** | 0.859152 |
| **R² Score** | 0.139953 |

---
## 4. 训练可视化
![训练结果图](training_results_q.png)
