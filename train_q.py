import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem, Crippen
import warnings
import time
import os

warnings.filterwarnings('ignore')

# 设置设备
def get_device():
    """获取最佳计算设备"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"使用 Apple Silicon GPU (MPS): {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 NVIDIA GPU (CUDA): {device}")
    else:
        device = torch.device("cpu")
        print(f"使用 CPU: {device}")
    return device

# 优化CPU性能
def optimize_cpu_performance():
    """优化CPU性能设置"""
    torch.set_num_threads(os.cpu_count())
    print(f"设置PyTorch线程数: {torch.get_num_threads()}")
    torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False

class ChemicalDataset(Dataset):
    """化学数据集类"""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class MLP(nn.Module):
    """多层感知机模型"""
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

def smiles_to_one_hot(smiles, char_to_int, max_len):
    """将SMILES字符串转换为one-hot编码特征"""
    try:
        smiles = str(smiles).strip()
        # 填充
        if len(smiles) < max_len:
            smiles = smiles + 'pad' * (max_len - len(smiles))
        else:
            smiles = smiles[:max_len]

        one_hot = np.zeros((max_len, len(char_to_int)))
        
        for i, char in enumerate(smiles):
            if char in char_to_int:
                one_hot[i, char_to_int[char]] = 1
            else:
                # 如果字符不在字典中，可以忽略或标记为未知
                # 在我们的实现中，字符集是从数据中构建的，所以理论上不会发生
                print(f"警告: 字符 '{char}' 不在字符集中，已忽略。")
        
        # 将2D one-hot矩阵展平为1D向量
        return one_hot.flatten()
    except Exception as e:
        print(f"警告: 处理SMILES '{smiles}' 时发生错误: {e}，已跳过。")
        return None

def load_and_preprocess_data(csv_path):
    """加载和预处理数据"""
    print("正在加载数据...")
    df = pd.read_csv(csv_path)
    print(f"数据集形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    smiles_list = df.iloc[:, 0].astype(str).tolist()
    
    # 创建字符集
    all_chars = set(''.join(smiles_list))
    charset = sorted(list(all_chars))
    charset.append('pad') # 添加填充字符
    char_to_int = {char: i for i, char in enumerate(charset)}
    
    # 获取最大长度
    max_len = max(len(s) for s in smiles_list)
    print(f"字符集大小: {len(charset)}")
    print(f"最大SMILES长度: {max_len}")

    features = []
    valid_indices = []
    for idx, smiles in enumerate(df.iloc[:, 0]):
        feature_vector = smiles_to_one_hot(str(smiles), char_to_int, max_len)
        if feature_vector is not None:
            features.append(feature_vector)
            valid_indices.append(idx)

    if not features:
        raise ValueError("无法从任何SMILES中提取特征")

    X = np.array(features)
    # 只取第二列 (Q值) 作为目标，并重塑为 (n, 1)
    y = df.iloc[valid_indices, 1].values.reshape(-1, 1)

    print(f"特征形状: {X.shape}")
    print(f"目标形状: {y.shape}")
    return X, y, charset, max_len

def train_model(X, y, config):
    """训练模型"""
    device = get_device()
    optimize_cpu_performance()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    train_dataset = ChemicalDataset(X_train_scaled, y_train_scaled)
    test_dataset = ChemicalDataset(X_test_scaled, y_test_scaled)
    
    batch_size = config['batch_size']
    num_workers = min(4, os.cpu_count() // 2)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if device.type != 'cpu' else False, persistent_workers=True if num_workers > 0 else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type != 'cpu' else False, persistent_workers=True if num_workers > 0 else False)
    
    # output_size=y.shape[1] (现在是1)
    model = MLP(input_size=X.shape[1], hidden_sizes=config['hidden_sizes'], output_size=y.shape[1], dropout_rate=config['dropout_rate']).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"开始在 {device} 上训练...")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_q_temp.pth')
        else:
            patience_counter += 1
        
        if epoch == 0:
            torch.save(model.state_dict(), 'best_model_q_temp.pth')
            if not np.isnan(val_loss):
                best_val_loss = val_loss

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{config["epochs"]}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        if patience_counter >= config.get('early_stopping_patience', 100) and (epoch + 1) >= 150:
            print(f"早停触发，在第 {epoch+1} 轮停止训练")
            break
    
    model.load_state_dict(torch.load('best_model_q_temp.pth'))
    total_time = time.time() - start_time
    print(f"训练完成，总用时: {total_time:.2f}秒")
    
    return model, scaler_X, scaler_y, train_losses, val_losses, X_test_scaled, y_test, device

def evaluate_model(model, scaler_X, scaler_y, X_test_scaled, y_test, device):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        predictions_scaled = model(X_test_tensor).cpu().numpy()
        predictions = scaler_y.inverse_transform(predictions_scaled)
    
    # 直接计算Q值的指标
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {'mse': mse, 'mae': mae, 'r2': r2}
    
    print("\n模型评估结果 (Q值):")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    return predictions, metrics

def plot_results(train_losses, val_losses, y_test, predictions):
    """绘制训练结果"""
    # 调整图表布局为 1x3
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 损失曲线
    axes[0].plot(train_losses, label='Training Loss')
    axes[0].plot(val_losses, label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Q参数预测对比
    axes[1].scatter(y_test, predictions, alpha=0.7)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('True Q')
    axes[1].set_ylabel('Predicted Q')
    axes[1].set_title('Q Parameter: Predicted vs True')
    axes[1].grid(True)
    
    # 预测误差分布
    q_error = y_test - predictions
    axes[2].hist(q_error, alpha=0.7, label='Q Error', bins=30)
    axes[2].set_xlabel('Prediction Error')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Prediction Error Distribution for Q')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results_q.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_report_to_md(metrics, config, total_samples, descriptor_info, file_path='training_report_q.md'):
    """将训练报告保存到Markdown文件"""
    report = f"""# 模型训练报告 (Q值预测)

## 1. 摘要
- **数据集**: `results-v4.csv`
- **训练目标**: Q值
- **训练样本总数**: {total_samples}
- **描述符**: {descriptor_info}
- **报告生成时间**: {time.strftime("%Y-%m-%d %H:%M:%S")}

---
## 2. 训练配置
| 参数 | 值 |
| :--- | :--- |
"""
    for key, value in config.items():
        report += f"| `{key}` | {value} |\n"
    
    report += f"""
---
## 3. 模型评估结果 (Q值)
| 指标 | 值 |
| :--- | :--- |
| **MSE** | {metrics['mse']:.6f} |
| **MAE** | {metrics['mae']:.6f} |
| **R² Score** | {metrics['r2']:.6f} |

---
## 4. 训练可视化
![训练结果图](training_results_q.png)
"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n训练报告已保存到 '{file_path}'")

def main():
    """主函数"""
    print("="*60)
    print("多层感知机训练 (Q值预测) - One-Hot编码")
    print("="*60)
    
    config = {
        'hidden_sizes': [1024, 512, 256],
        'dropout_rate': 0.5,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'epochs': 500,
        'early_stopping_patience': 100
    }
    
    print("训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # 使用 'results-v4.csv'
        X, y, charset, max_len = load_and_preprocess_data('results-v4.csv')

        results = train_model(X, y, config)
        model, scaler_X, scaler_y, train_losses, val_losses, X_test_scaled, y_test, device = results

        predictions, metrics = evaluate_model(model, scaler_X, scaler_y, X_test_scaled, y_test, device)

        plot_results(train_losses, val_losses, y_test, predictions)

        descriptor_info = f"One-Hot Encoding (charset size: {len(charset)}, max_len: {max_len})"
        save_report_to_md(metrics, config, X.shape[0], descriptor_info)

        # 保存为新模型
        model_save_path = 'mlp_model_q_onehot_optimized.pth'
        model_save_dict = {
            'model_state_dict': model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'config': config,
            'device_type': device.type,
            'model_architecture': {
                'input_size': X.shape[1],
                'hidden_sizes': config['hidden_sizes'],
                'output_size': y.shape[1],
                'dropout_rate': config['dropout_rate']
            },
            'one_hot_params': {
                'charset': charset,
                'max_len': max_len
            }
        }
        torch.save(model_save_dict, model_save_path)

        if os.path.exists('best_model_q_temp.pth'):
            os.remove('best_model_q_temp.pth')

        print("\n" + "="*60)
        print("训练完成！")
        print(f"模型已保存到 '{model_save_path}'")
        print(f"训练结果图已保存到 'training_results_q.png'")
        print(f"使用的计算设备: {device}")
        print("="*60)

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
