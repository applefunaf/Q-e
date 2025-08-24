import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem, Crippen, rdFingerprintGenerator
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
    # 设置线程数 - Apple Silicon通常有8-10个性能核心
    torch.set_num_threads(os.cpu_count())
    print(f"设置PyTorch线程数: {torch.get_num_threads()}")
    
    # 启用混合精度训练支持
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
    """多层感知机模型 - 优化用于Apple Silicon"""
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化 - 提高训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

def smiles_to_features(smiles):
    """将SMILES字符串转换为分子描述符作为特征，并处理无效SMILES"""
    try:
        smiles = str(smiles).strip()
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        
        if mol is None:
            print(f"警告: 无法解析SMILES '{smiles}'，已跳过。")
            return None
            
        # 1. ECFP指纹 (Morgan Fingerprint with radius=1, ECFP2) - 使用新的MorganGenerator API
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=1, fpSize=2048)
        ecfp_fp = fp_gen.GetFingerprint(mol)
        ecfp_features = [int(bit) for bit in ecfp_fp.ToBitString()]

        # 组合所有特征
        features = ecfp_features
        return features

    except Exception as e:
        print(f"警告: 处理SMILES '{smiles}' 时发生错误: {e}，已跳过。")
        return None

def load_and_preprocess_data(csv_path):
    """加载和预处理数据，并对Q值取对数"""
    print("正在加载数据...")
    df = pd.read_csv(csv_path)
    print(f"原始数据集形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    # 过滤掉Q <= 0的数据，因为ln(Q)会无效
    q_column_name = df.columns[1]
    original_rows = len(df)
    df_valid_q = df[df[q_column_name] > 0].copy()
    if len(df_valid_q) < original_rows:
        print(f"警告: 移除了 {original_rows - len(df_valid_q)} 行，因为 Q <= 0")
    
    # 只用SMILES列和Q/e列
    features = []
    valid_indices = []
    # 使用过滤后的df_valid_q进行迭代
    for idx, smiles in enumerate(df_valid_q.iloc[:, 0]):
        feature_vector = smiles_to_features(str(smiles))
        if feature_vector is not None:
            features.append(feature_vector)
            valid_indices.append(df_valid_q.index[idx])

    if not features:
        raise ValueError("无法从任何SMILES中提取特征")

    X = np.array(features)
    # 使用 .loc 和 valid_indices 从过滤后的DataFrame中提取目标值
    y = df_valid_q.loc[valid_indices].iloc[:, 1:3].values

    # 对Q值取对数
    print("对目标变量Q进行对数转换 (ln(Q))")
    y[:, 0] = np.log(y[:, 0])

    print(f"特征形状: {X.shape}")
    print(f"目标形状: {y.shape}")
    return X, y

def train_model(X, y, config):
    """训练模型 - Apple Silicon 优化版本"""
    # 获取设备
    device = get_device()
    
    # 优化CPU性能
    optimize_cpu_performance()
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    

    # 特征标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # 创建数据加载器
    train_dataset = ChemicalDataset(X_train_scaled, y_train_scaled)
    test_dataset = ChemicalDataset(X_test_scaled, y_test_scaled)
    
    # 优化批次大小和工作进程数
    batch_size = config['batch_size']
    num_workers = min(4, os.cpu_count() // 2)  # Apple Silicon优化
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'mps' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'mps' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # 创建模型并移动到设备
    model = MLP(
        input_size=X.shape[1],
        hidden_sizes=config['hidden_sizes'],
        output_size=y.shape[1],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW( # AdamW通常比Adam更好
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=15, 
        factor=0.5
    )
    
    # 训练记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"开始在 {device} 上训练...")
    print(f"批次大小: {batch_size}, 工作进程数: {num_workers}")
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_targets in train_loader:
            # 将数据移动到正确的设备
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                # 将数据移动到正确的设备
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)

                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        # 调整学习率
        scheduler.step(val_loss)
        
        # 早停机制
        # 只有在val_loss是有效数值时才进行比较和保存
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model_temp.pth')
        else:
            patience_counter += 1
        
        # 在第一个epoch结束后，无论如何都保存一次模型，确保文件存在
        if epoch == 0:
            torch.save(model.state_dict(), 'best_model_temp.pth')
            # 如果第一个epoch的val_loss有效，则将其设为best_val_loss
            if not np.isnan(val_loss):
                best_val_loss = val_loss

        epoch_time = time.time() - epoch_start_time
        
        # 打印进度
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{config["epochs"]}], '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # 早停
        if patience_counter >= config.get('early_stopping_patience', 100):
            print(f"早停触发，在第 {epoch+1} 轮停止训练")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model_temp.pth'))
    
    total_time = time.time() - start_time
    print(f"训练完成，总用时: {total_time:.2f}秒")
    
    return model, scaler_X, scaler_y, train_losses, val_losses, X_test_scaled, y_test, y_test_scaled, device

def evaluate_model(model, scaler_X, scaler_y, X_test_scaled, y_test, y_test_scaled, device):
    """评估模型性能 - 设备优化版本"""
    model.eval()
    
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        predictions_scaled = model(X_test_tensor).cpu().numpy()
        
        # 反标准化预测结果
        predictions = scaler_y.inverse_transform(predictions_scaled)
    
    # 总体指标是在预测空间（ln(Q), e）上计算的
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        'overall': {'mse': mse, 'mae': mae, 'r2': r2},
        'ln(Q)': {},
        'e': {}
    }
    
    print("\n模型评估结果 (预测空间: ln(Q), e):")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    # 分别评估ln(Q)和e参数
    for i, param in enumerate(['ln(Q)', 'e']):
        mse_param = mean_squared_error(y_test[:, i], predictions[:, i])
        mae_param = mean_absolute_error(y_test[:, i], predictions[:, i])
        r2_param = r2_score(y_test[:, i], predictions[:, i])
        
        metrics[param]['mse'] = mse_param
        metrics[param]['mae'] = mae_param
        metrics[param]['r2'] = r2_param
        
        print(f"\n{param} 参数:")
        print(f"  MSE: {mse_param:.6f}")
        print(f"  MAE: {mae_param:.6f}")
        print(f"  R²: {r2_param:.6f}")
    
    return predictions, metrics

def plot_results(train_losses, val_losses, y_test, predictions):
    """绘制训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 损失曲线
    axes[0, 0].plot(train_losses, label='Training Loss')
    axes[0, 0].plot(val_losses, label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # ln(Q)参数预测对比
    axes[0, 1].scatter(y_test[:, 0], predictions[:, 0], alpha=0.7)
    axes[0, 1].plot([y_test[:, 0].min(), y_test[:, 0].max()], 
                    [y_test[:, 0].min(), y_test[:, 0].max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('True ln(Q)')
    axes[0, 1].set_ylabel('Predicted ln(Q)')
    axes[0, 1].set_title('ln(Q) Parameter: Predicted vs True')
    axes[0, 1].grid(True)
    
    # e参数预测对比
    axes[1, 0].scatter(y_test[:, 1], predictions[:, 1], alpha=0.7)
    axes[1, 0].plot([y_test[:, 1].min(), y_test[:, 1].max()], 
                    [y_test[:, 1].min(), y_test[:, 1].max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('True e')
    axes[1, 0].set_ylabel('Predicted e')
    axes[1, 0].set_title('e Parameter: Predicted vs True')
    axes[1, 0].grid(True)
    
    # 预测误差分布
    q_error = y_test[:, 0] - predictions[:, 0]
    e_error = y_test[:, 1] - predictions[:, 1]
    
    axes[1, 1].hist(q_error, alpha=0.7, label='ln(Q) Error', bins=20)
    axes[1, 1].hist(e_error, alpha=0.7, label='e Error', bins=20)
    axes[1, 1].set_xlabel('Prediction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_report_to_md(metrics, config, total_samples, descriptor_info, file_path='training_report.md'):
    """将训练报告保存到Markdown文件"""
    report = f"""# 模型训练报告

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
    
    report += f"""
---

## 3. 模型评估结果

### 总体性能

| 指标 | 值 |
| :--- | :--- |
| **MSE** | {metrics['overall']['mse']:.6f} |
| **MAE** | {metrics['overall']['mae']:.6f} |
| **R² Score** | {metrics['overall']['r2']:.6f} |
"""

    # 动态生成各参数的性能报告
    for param in metrics:
        if param != 'overall':
            report += f"""
### 参数{param}性能

| 指标 | 值 |
| :--- | :--- |
| **MSE** | {metrics[param]['mse']:.6f} |
| **MAE** | {metrics[param]['mae']:.6f} |
| **R² Score** | {metrics[param]['r2']:.6f} |
"""

    report += """
---

## 4. 训练可视化

![训练结果图](training_results.png)
"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\\n训练报告已保存到 '{file_path}'")

def main():
    """主函数 - Apple Silicon 优化版本"""
    print("="*60)
    print("多层感知机训练 - Apple Silicon 优化版本")
    print("="*60)
    
    # 配置参数 - 为Apple Silicon优化
    config = {
        'hidden_sizes': [256, 128, 64],   # 增大网络容量
        'dropout_rate': 0.3,              # 增加正则化
        'learning_rate': 0.001,           # 学习率
        'batch_size': 64,                 # 增大批次以利用并行计算
        'epochs': 300,                    # 训练轮数
        'early_stopping_patience': 100     # 早停耐心值
    }
    
    print("训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # 加载和预处理数据（改为results-v4.csv，SMILES->Q/e）
        X, y = load_and_preprocess_data('results-v4.csv')

        # 训练模型
        results = train_model(X, y, config)
        model, scaler_X, scaler_y, train_losses, val_losses, X_test_scaled, y_test, y_test_scaled, device = results

        # 评估模型
        predictions, metrics = evaluate_model(model, scaler_X, scaler_y, X_test_scaled, y_test, y_test_scaled, device)

        # 绘制结果
        plot_results(train_losses, val_losses, y_test, predictions)

        # 保存报告
        descriptor_info = "ECFP2 Fingerprints (2048 bits)"
        save_report_to_md(metrics, config, X.shape[0], descriptor_info, file_path='training_report_lnQ.md')

        # 保存模型 - 包含设备信息
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
            }
        }

        torch.save(model_save_dict, 'mlp_model_optimized.pth')

        # 清理临时文件
        if os.path.exists('best_model_temp.pth'):
            os.remove('best_model_temp.pth')

        print("\n" + "="*60)
        print("训练完成！")
        print(f"模型已保存到 'mlp_model_optimized.pth'")
        print(f"训练结果图已保存到 'training_results.png'")
        print(f"使用的计算设备: {device}")
        print("="*60)

        # 保存训练报告
        total_samples = X.shape[0]
        descriptor_info = "ECFP2 Fingerprints"
        save_report_to_md(metrics, config, total_samples, descriptor_info, file_path='training_report_lnQ.md')

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

def load_trained_model(model_path='mlp_model_optimized.pth'):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 重建模型架构
    arch = checkpoint['model_architecture']
    model = MLP(
        input_size=arch['input_size'],
        hidden_sizes=arch['hidden_sizes'],
        output_size=arch['output_size'],
        dropout_rate=arch['dropout_rate']
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 获取缩放器
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']
    
    return model, scaler_X, scaler_y

if __name__ == "__main__":
    main()