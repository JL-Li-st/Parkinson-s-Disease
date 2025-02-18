import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import logging

# Adding necessary paths
sys.path.extend(['../'])

# Importing the provided modules
from feeders.feeder_ntu import Feeder
from model.degcn import Model
from graph.ntu_rgb_d import Graph

# Set the seed for reproducibility
def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set up logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename='logs/test_loss.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Helper function to log and print messages
def log_message(message):
    print(message)
    logging.info(message)

# 主函数，用于直接加载模型并测试
def main():
    # 固定参数
    args = {
        'data_path': './data/',
        'label_path': './data/',
        'batch_size': 8,
        'num_workers': 4,
        'num_class': 1,  # 二分类
        'seed': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    # 初始化随机种子
    init_seed(args['seed'])
    log_message(f"Arguments: {args}")

    # 加载数据集
    dataset = Feeder(data_folder=args['data_path'], label_file_path=args['label_path'], split='test')
    log_message("Dataset loaded successfully.")

    # 将数据集划分为测试集
    test_loader = torch_data.DataLoader(
        dataset=dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args['num_workers']
    )

    # 初始化模型
    model = Model(
        num_class=args['num_class'],
        num_point=17,
        num_person=1,
        graph='graph.ntu_rgb_d.Graph',
        graph_args=dict(labeling_mode='spatial'),
        in_channels=3
    )
    device = torch.device(args['device'])
    model = model.to(device)

    # 加载已训练的模型权重
    best_model_path = 'weights/17_2k_100_best_model_Turning.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        log_message(f"Loaded best model from {best_model_path}")
    else:
        log_message(f"Best model file not found: {best_model_path}")
        return

    # 计算测试集损失
    model.eval()  # 设置模型为评估模式
    total_test_loss = 0
    criterion = nn.BCELoss()  # 使用与训练时相同的损失函数
    all_test_labels = []
    all_test_outputs = []

    # 不计算梯度进行推理
    with torch.no_grad():
        for data, label, _ in test_loader:
            data, label = data.to(device), label.float().to(device)
            label = label.view(-1, 1)

            # 前向传播
            output = model(data)

            # 计算测试损失
            test_loss = criterion(output, label)
            total_test_loss += test_loss.item()

            # 收集所有标签和输出用于后续评估
            all_test_labels.append(label.cpu().numpy())
            all_test_outputs.append(output.cpu().numpy())

    # 计算平均测试损失
    avg_test_loss = total_test_loss / len(test_loader)
    log_message(f"Test Loss: {avg_test_loss:.4f}")

    # 可选：计算评估指标
    all_test_labels = np.vstack(all_test_labels)
    all_test_outputs = np.vstack(all_test_outputs)
    accuracy = accuracy_score(all_test_labels, (all_test_outputs > 0.5).astype(int))
    precision = precision_score(all_test_labels, (all_test_outputs > 0.5).astype(int))
    recall = recall_score(all_test_labels, (all_test_outputs > 0.5).astype(int))
    f1 = f1_score(all_test_labels, (all_test_outputs > 0.5).astype(int))

    log_message(f"Accuracy: {accuracy:.4f}")
    log_message(f"Precision: {precision:.4f}")
    log_message(f"Recall: {recall:.4f}")
    log_message(f"F1 Score: {f1:.4f}")

if __name__ == '__main__':
    main()
