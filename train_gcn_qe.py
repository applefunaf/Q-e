import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from rdkit import Chem
import warnings
import time
import os

warnings.filterwarnings('ignore')

# --- 设备和性能优化 ---
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

def optimize_cpu_performance():
    """优化CPU性能设置"""
    torch.set_num_threads(os.cpu_count())
    print(f"设置PyTorch线程数: {torch.get_num_threads()}")
    torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False

# --- 分子到图的转换 (PyG) ---
def get_atom_features(atom):
    """为单个原子生成特征向量"""
    possible_atoms = ['C', 'N', 'O', 'F', 'H', 'S', 'Cl', 'Br', 'I', 'P']
    atom_type = [0] * len(possible_atoms)
    try:
        atom_type[possible_atoms.index(atom.GetSymbol())] = 1
    except ValueError:
        pass
    return atom_type + [
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        # atom.GetValence(), # Temporarily removed to ensure stability
        int(atom.GetIsAromatic()),
        atom.GetFormalCharge()
    ]

def smiles_to_pyg_data(smiles):
    """将SMILES字符串转换为PyG的Data对象"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)

        atom_features_list = [get_atom_features(atom) for atom in mol.GetAtoms()]
        if not atom_features_list:
            return None
            
        x = torch.tensor(atom_features_list, dtype=torch.float)

        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([(i, j), (j, i)])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)

    except Exception as e:
        print(f"警告: 处理SMILES '{smiles}' 时发生错误: {e}，已跳过。")
        return None

# --- 数据加载和预处理 (PyG) ---
def load_and_preprocess_data(csv_path):
    """加载数据并将SMILES转换为PyG图数据列表"""
    print("正在加载数据并将SMILES转换为PyG图...")
    df = pd.read_csv(csv_path)
    
    data_list = []
    fail_count = 0
    # 使用列名访问，更稳健
    for _, row in df.iterrows():
        if 'SMILES' not in row:
            raise ValueError("CSV文件中找不到'SMILES'列")
        
        smiles = str(row['SMILES'])
        data = smiles_to_pyg_data(smiles)
        
        if data is not None:
            if 'Q' not in row or 'e' not in row:
                raise ValueError("CSV文件中找不到'Q'或'e'列")
            
            # 目标值转换为对数
            q_val = row['Q']
            if q_val <= 0:
                print(f"警告: SMILES '{smiles}' 的Q值非正 ({q_val})，已跳过。")
                fail_count += 1
                continue
            
            targets = np.array([np.log(q_val), row['e']], dtype=np.float32)
            data.y = torch.tensor(targets, dtype=torch.float).unsqueeze(0)
            data_list.append(data)
        else:
            fail_count += 1

    if not data_list:
        print(f"错误: 无法从任何SMILES中提取图特征。总共尝试了 {len(df)} 个，全部失败。")
        print("请检查CSV文件格式和SMILES字符串的有效性。")
        if fail_count > 0 and 'SMILES' in df.columns:
             print("前5个处理失败的SMILES示例:")
             for i in range(min(5, len(df))):
                 print(f"  - {df['SMILES'].iloc[i]}")
        raise ValueError("无法从任何SMILES中提取图特征")

    print(f"成功处理 {len(data_list)} 个分子。")
    return data_list

# --- GCN 模型定义 (PyG) ---
class GCN(nn.Module):
    """基于PyG的图卷积网络模型"""
    def __init__(self, input_feat_dim, output_size, hidden_dim=64, dropout_rate=0.3):
        super(GCN, self).__init__()
        self.gcn1 = GCNConv(input_feat_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.gcn2(x, edge_index))
        
        graph_embedding = global_mean_pool(x, batch)
        
        x = F.relu(self.fc1(graph_embedding))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return self.fc2(x)

# --- 训练和评估 (PyG) ---
def train_model(data_list, config):
    """训练PyG GCN模型"""
    device = get_device()
    optimize_cpu_performance()
    
    all_y = torch.cat([data.y for data in data_list]).numpy()
    scaler_y = StandardScaler().fit(all_y)
    for data in data_list:
        data.y = torch.tensor(scaler_y.transform(data.y.numpy()), dtype=torch.float)

    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)
    
    model = GCN(
        input_feat_dim=train_loader.dataset[0].num_node_features,
        output_size=2,
        hidden_dim=config['hidden_dim'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('patience', 30)
    
    print(f"开始在 {device} 上训练...")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        train_losses.append(epoch_train_loss / len(train_loader))
        
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch)
                loss = criterion(outputs, batch.y)
                epoch_val_loss += loss.item()
        
        current_val_loss = epoch_val_loss / len(test_loader)
        val_losses.append(current_val_loss)
        scheduler.step(current_val_loss)
        
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), 'best_gcn_model_pyg_temp.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{config["epochs"]}], Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}')

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} as validation loss did not improve for {patience} epochs.")
            break

    model.load_state_dict(torch.load('best_gcn_model_pyg_temp.pth'))
    print(f"训练完成，总用时: {time.time() - start_time:.2f}秒")
    
    y_test = scaler_y.inverse_transform(torch.cat([data.y for data in test_data]).numpy())
    return model, scaler_y, test_loader, y_test, train_losses, val_losses, device

def evaluate_model(model, scaler_y, test_loader, y_test, device):
    """评估PyG GCN模型性能"""
    model.eval()
    predictions_scaled = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            predictions_scaled.append(outputs.cpu())
            
    predictions_scaled = torch.cat(predictions_scaled, dim=0).numpy()
    y_pred_unscaled = scaler_y.inverse_transform(predictions_scaled)
    
    # y_test is already unscaled, it's our y_true_unscaled
    y_true_unscaled = y_test

    # 将对数转换回原始值
    q_true = np.exp(y_true_unscaled[:, 0])
    q_pred = np.exp(y_pred_unscaled[:, 0])
    e_true = y_true_unscaled[:, 1]
    e_pred = y_pred_unscaled[:, 1]

    # 计算指标
    metrics = {
        'Q': {
            'RMSE': np.sqrt(mean_squared_error(q_true, q_pred)),
            'MAE': mean_absolute_error(q_true, q_pred),
            'R2': r2_score(q_true, q_pred)
        },
        'e': {
            'RMSE': np.sqrt(mean_squared_error(e_true, e_pred)),
            'MAE': mean_absolute_error(e_true, e_pred),
            'R2': r2_score(e_true, e_pred)
        }
    }
    return y_pred_unscaled, metrics

# --- 绘图和报告 ---
def plot_results(y_true_unscaled, y_pred_unscaled, metrics):
    """绘制训练结果图"""
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Q 属性图
    q_true = np.exp(y_true_unscaled[:, 0])
    q_pred = np.exp(y_pred_unscaled[:, 0])
    axes[0].scatter(q_true, q_pred, alpha=0.6, label=f"R² = {metrics['Q']['R2']:.3f}")
    axes[0].plot([min(q_true), max(q_true)], [min(q_true), max(q_true)], 'r--', label="Ideal Line")
    axes[0].set_xlabel("Actual Q")
    axes[0].set_ylabel("Predicted Q")
    axes[0].set_title("Q: Actual vs. Predicted")
    axes[0].legend()
    axes[0].grid(True)

    # e 属性图
    e_true = y_true_unscaled[:, 1]
    e_pred = y_pred_unscaled[:, 1]
    axes[1].scatter(e_true, e_pred, alpha=0.6, label=f"R² = {metrics['e']['R2']:.3f}")
    axes[1].plot([min(e_true), max(e_true)], [min(e_true), max(e_true)], 'r--', label="Ideal Line")
    axes[1].set_xlabel("Actual e")
    axes[1].set_ylabel("Predicted e")
    axes[1].set_title("e: Actual vs. Predicted")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('gcn_pyg_training_results.png', dpi=300)
    plt.show()

def save_report_to_md(metrics, config, total_samples, descriptor_info, file_path='gcn_pyg_training_report.md'):
    """将训练报告保存到Markdown文件"""
    report = f"""# GCN (PyG) 模型训练报告
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
    report += "--- \n## 3. 模型评估结果\n"
    for param, m in metrics.items():
        report += f"### {param.upper()} 性能\n| 指标 | 值 |\n| :--- | :--- |\n"
        report += f"| **MSE** | {m['mse']:.6f} |\n| **MAE** | {m['mae']:.6f} |\n| **R² Score** | {m['r2']:.6f} |\n"
    report += """---
## 4. 训练可视化
![训练结果图](gcn_pyg_training_results.png)
"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n训练报告已保存到 '{file_path}'")

# --- 主函数 ---
def main():
    """主函数"""
    config = {
        'csv_path': 'results-v4-with-qm9.csv',
        'epochs': 300,
        'batch_size': 64,
        'learning_rate': 0.001,
        'hidden_dim': 128,
        'dropout_rate': 0.3,
        'patience': 30
    }
    
    print("训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        data_list = load_and_preprocess_data('results-v4.csv')
        
        results = train_model(data_list, config)
        model, scaler_y, test_loader, y_test, train_losses, val_losses, device = results
        
        predictions, metrics = evaluate_model(model, scaler_y, test_loader, y_test, device)
        
        plot_results(y_test, predictions, metrics)
        
        descriptor_info = "PyG Graph Representation"
        save_report_to_md(metrics, config, len(data_list), descriptor_info)

        model_save_dict = {
            'model_state_dict': model.state_dict(),
            'scaler_y': scaler_y,
            'config': config,
            'input_feat_dim': data_list[0].num_node_features
        }
        torch.save(model_save_dict, 'gcn_model_pyg_optimized.pth')

        if os.path.exists('best_gcn_model_pyg_temp.pth'):
            os.remove('best_gcn_model_pyg_temp.pth')

        print("\n" + "="*60)
        print("训练完成！")
        print(f"模型已保存到 'gcn_model_pyg_optimized.pth'")
        print(f"训练结果图已保存到 'gcn_pyg_training_results.png'")
        print("="*60)

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
