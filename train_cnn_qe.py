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

# 设置设备 - Apple Silicon 优化
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

# 设置线程数优化 - 对CPU计算特别有用
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

class CNN(nn.Module):
    """一维卷积神经网络模型"""
    def __init__(self, input_length, output_size, dropout_rate=0.3):
        super(CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv Layer 1
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            # Conv Layer 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )
        
        # 计算卷积层输出的维度
        self._feature_size = self._get_conv_output_size(input_length)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self._feature_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_size)
        )
        
        self._initialize_weights()

    def _get_conv_output_size(self, input_length):
        """辅助函数，用于计算卷积层输出后的展平维度"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            output = self.conv_layers(dummy_input)
            return int(np.prod(output.size()))

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x的输入形状: (batch_size, feature_length)
        # CNN需要: (batch_size, channels, length)
        x = x.unsqueeze(1)  # 增加一个通道维度 -> (batch_size, 1, feature_length)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x

def smiles_to_features(smiles):
    """将SMILES字符串转换为MACCS键作为特征"""
    try:
        smiles = str(smiles).strip()
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        
        if mol is None:
            print(f"警告: 无法解析SMILES '{smiles}'，已跳过。")
            return None
            
        maccs_keys = MACCSkeys.GenMACCSKeys(mol)
        maccs_features = [int(bit) for bit in maccs_keys.ToBitString()]
        return maccs_features

    except Exception as e:
        print(f"警告: 处理SMILES '{smiles}' 时发生错误: {e}，已跳过。")
        return None

def load_and_preprocess_data(csv_path):
    """加载和预处理数据"""
    print("正在加载数据...")
    df = pd.read_csv(csv_path)
    print(f"数据集形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    features = []
    valid_indices = []
    for idx, smiles in enumerate(df.iloc[:, 0]):
        feature_vector = smiles_to_features(str(smiles))
        if feature_vector is not None:
            features.append(feature_vector)
            valid_indices.append(idx)

    if not features:
        raise ValueError("无法从任何SMILES中提取特征")

    X = np.array(features)
    y = df.iloc[valid_indices, 1:3].values.astype(np.float32)

    # 将 Q 转换为 ln(Q)
    print("将目标 'Q' 转换为 'ln(Q)'")
    y[:, 0] = np.log(y[:, 0])

    print(f"特征形状: {X.shape}")
    print(f"目标形状: {y.shape}")
    return X, y

def train_model(X, y, config):
    """训练CNN模型"""
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if device.type == 'mps' else False, persistent_workers=True if num_workers > 0 else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'mps' else False, persistent_workers=True if num_workers > 0 else False)
    
    model = CNN(
        input_length=X.shape[1],
        output_size=y.shape[1],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    train_losses, val_losses = [], []
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
        
        train_losses.append(train_loss / len(train_loader))
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(test_loader))
        scheduler.step(val_losses[-1])
        
        if not np.isnan(val_losses[-1]) and val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            torch.save(model.state_dict(), 'best_cnn_model_temp.pth')
        else:
            patience_counter += 1
        
        if epoch == 0:
            torch.save(model.state_dict(), 'best_cnn_model_temp.pth')
            if not np.isnan(val_losses[-1]):
                best_val_loss = val_losses[-1]

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{config["epochs"]}], Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        if patience_counter >= config.get('early_stopping_patience', 100):
            print(f"早停触发，在第 {epoch+1} 轮停止训练")
            break
    
    model.load_state_dict(torch.load('best_cnn_model_temp.pth'))
    print(f"训练完成，总用时: {time.time() - start_time:.2f}秒")
    
    return model, scaler_X, scaler_y, train_losses, val_losses, X_test_scaled, y_test, y_test_scaled, device

def evaluate_model(model, scaler_X, scaler_y, X_test_scaled, y_test, y_test_scaled, device):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        predictions_scaled = model(X_test_tensor).cpu().numpy()
        predictions = scaler_y.inverse_transform(predictions_scaled)
    
    metrics = {}
    
    # 总体性能
    overall_mse = mean_squared_error(y_test, predictions)
    overall_mae = mean_absolute_error(y_test, predictions)
    overall_r2 = r2_score(y_test, predictions)
    metrics['overall'] = {'mse': overall_mse, 'mae': overall_mae, 'r2': overall_r2}
    
    print("\n模型总体评估结果:")
    print(f"  Mean Squared Error (MSE): {overall_mse:.6f}")
    print(f"  Mean Absolute Error (MAE): {overall_mae:.6f}")
    print(f"  R² Score: {overall_r2:.6f}")

    # 各参数性能
    for i, param in enumerate(['lnQ', 'e']):
        mse_param = mean_squared_error(y_test[:, i], predictions[:, i])
        mae_param = mean_absolute_error(y_test[:, i], predictions[:, i])
        r2_param = r2_score(y_test[:, i], predictions[:, i])
        metrics[param] = {'mse': mse_param, 'mae': mae_param, 'r2': r2_param}
        print(f"\n{param} 参数性能:")
        print(f"  MSE: {mse_param:.6f}")
        print(f"  MAE: {mae_param:.6f}")
        print(f"  R²: {r2_param:.6f}")
    
    return predictions, metrics

def plot_results(train_losses, val_losses, y_test, predictions):
    """绘制训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].plot(train_losses, label='Training Loss')
    axes[0, 0].plot(val_losses, label='Validation Loss')
    axes[0, 0].set(xlabel='Epoch', ylabel='Loss', title='Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    for i, param in enumerate(['lnQ', 'e']):
        ax = axes[0, 1] if i == 0 else axes[1, 0]
        
        y_true = y_test[:, i]
        y_pred = predictions[:, i]

        ax.scatter(y_true, y_pred, alpha=0.7)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set(xlabel=f'True {param}', ylabel=f'Predicted {param}', title=f'{param} Parameter: Predicted vs True')
        ax.grid(True)
    
    q_error, e_error = y_test[:, 0] - predictions[:, 0], y_test[:, 1] - predictions[:, 1]
    axes[1, 1].hist(q_error, alpha=0.7, label='lnQ Error', bins=20)
    axes[1, 1].hist(e_error, alpha=0.7, label='e Error', bins=20)
    axes[1, 1].set(xlabel='Prediction Error', ylabel='Frequency', title='Prediction Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('cnn_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_report_to_md(metrics, config, total_samples, descriptor_info, file_path='cnn_training_report.md'):
    """将训练报告保存到Markdown文件"""
    report = f"""# CNN模型训练报告
## 1. 摘要
- **数据集**: `results-v4.csv`
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
    
    report += """---
## 3. 模型评估结果
"""
    # 保证 'overall' 在最前面
    report_order = ['overall', 'lnQ', 'e']
    
    for param in report_order:
        if param in metrics:
            m = metrics[param]
            param_name = "总体" if param == 'overall' else f"参数{param}"
            report += f"""### {param_name}性能
| 指标 | 值 |
| :--- | :--- |
| **MSE** | {m['mse']:.6f} |
| **MAE** | {m['mae']:.6f} |
| **R² Score** | {m['r2']:.6f} |
"""
    report += """---
## 4. 训练可视化
![训练结果图](cnn_training_results.png)
"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n训练报告已保存到 '{file_path}'")

def main():
    """主函数"""
    print("="*60)
    print("CNN模型训练 - Q/e 预测")
    print("="*60)
    
    config = {
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 300,
        'early_stopping_patience': 100
    }
    
    print("训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        X, y = load_and_preprocess_data('results-v4.csv')
        results = train_model(X, y, config)
        model, scaler_X, scaler_y, train_losses, val_losses, X_test_scaled, y_test, y_test_scaled, device = results
        predictions, metrics = evaluate_model(model, scaler_X, scaler_y, X_test_scaled, y_test, y_test_scaled, device)
        plot_results(train_losses, val_losses, y_test, predictions)
        
        descriptor_info = "MACCS Keys (167 bits)"
        save_report_to_md(metrics, config, X.shape[0], descriptor_info)

        model_save_dict = {
            'model_state_dict': model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'config': config,
            'device_type': device.type,
            'model_architecture': {
                'input_length': X.shape[1],
                'output_size': y.shape[1],
                'dropout_rate': config['dropout_rate']
            }
        }
        torch.save(model_save_dict, 'cnn_model_optimized.pth')

        if os.path.exists('best_cnn_model_temp.pth'):
            os.remove('best_cnn_model_temp.pth')

        print("\n" + "="*60)
        print("训练完成！")
        print(f"模型已保存到 'cnn_model_optimized.pth'")
        print(f"训练结果图已保存到 'cnn_training_results.png'")
        print(f"使用的计算设备: {device}")
        print("="*60)

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

def load_trained_model(model_path='cnn_model_optimized.pth'):
    """加载训练好的CNN模型"""
    checkpoint = torch.load(model_path, map_location='cpu')
    arch = checkpoint['model_architecture']
    model = CNN(
        input_length=arch['input_length'],
        output_size=arch['output_size'],
        dropout_rate=arch['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']
    return model, scaler_X, scaler_y

if __name__ == "__main__":
    main()
