# 模型训练报告 (e值预测)

## 1. 摘要
- **数据集**: `results-v4.csv`
- **训练目标**: e值
- **训练样本总数**: 256
- **描述符**: One-Hot Encoding (charset size: 38, max_len: 96)
- **报告生成时间**: 2025-08-17 18:21:34

---
## 2. 训练配置
| 参数 | 值 |
| :--- | :--- |
| `hidden_sizes` | [1024, 512, 256] |
| `dropout_rate` | 0.5 |
| `learning_rate` | 0.0001 |
| `batch_size` | 32 |
| `epochs` | 500 |
| `early_stopping_patience` | 100 |

---
## 3. 模型评估结果 (e值)
| 指标 | 值 |
| :--- | :--- |
| **MSE** | 1.732629 |
| **MAE** | 0.897206 |
| **R² Score** | 0.144224 |

---
## 4. 训练可视化
![训练结果图](training_results_e.png)
