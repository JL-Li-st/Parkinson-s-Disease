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
    filename='logs/17_PD_C_grad.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Helper function to log and print messages
def log_message(message):
    print(message)
    logging.info(message)




def plot_gradient_heatmap(data, gradient, epoch, batch_idx, batch_size, fold):
    """
    Plot heatmap of the average gradient with respect to the input data for all samples in the batch.
    This version includes fold information to differentiate between k-folds.
    """
    # Compute the mean absolute gradient across all samples in the batch
    grad_abs = gradient.abs().mean(dim=0).squeeze()  # Mean across batch size (dim=0) and channels (dim=1)

    # If data has more than one channel (3 channels), we can choose one channel or take the mean of all channels
    if grad_abs.ndimension() == 3:  #  (3, 100, 25)
        grad_abs_sample = grad_abs.mean(dim=0)  # Take mean across the channel dimension
    else:
        grad_abs_sample = grad_abs  # Already 2D data, no need to modify

    grad_abs_sample = grad_abs_sample.cpu().numpy()  # Convert to NumPy array for plotting

    # Increase figure size to make the plot wider
    plt.figure(figsize=(16, 8))  # Increase width to make space for all x-ticks

    # Plot heatmap with auto aspect ratio
    plt.imshow(grad_abs_sample, cmap='hot', interpolation='nearest', aspect='auto')  # Adjust aspect to auto

    # Show color bar
    plt.colorbar()

    # Title for the plot
    plt.title(f'Gradient Heatmap (Epoch {epoch}, Batch {batch_idx}, Fold {fold})')
    # Set x-ticks to show all positions
    plt.xticks(np.arange(grad_abs_sample.shape[1]))  # Show all x-axis ticks



    # Save the heatmap in the 'grad' folder, with fold and epoch information in the filename
    plt.savefig(f'myplot/heatmap/17_PD_C/gradient_heatmap_fold{fold}_epoch{epoch}_batch{batch_idx}.png')
    plt.close()



# 主函数，用于训练和验证
def main():
    # 固定参数
    args = {
        'data_path': './data/',
        'label_path': './data/',
        'batch_size': 8,
        'num_workers': 4,
        'learning_rate': 0.0001,
        'epochs': 100,
        'milestones': [30, 40],
        'gamma': 0.1,
        'num_class': 1,  # 更新为 1，二分类
        'num_person': 1,
        'seed': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_splits': 2
    }

    # 初始化随机种子
    init_seed(args['seed'])
    log_message(f"Arguments: {args}")

    # 加载数据集
    dataset = Feeder(data_folder=args['data_path'], label_file_path=args['label_path'], split='train')
    log_message("Dataset loaded successfully.")

    # 将数据集划分为 70% 训练/验证和 30% 测试集
    train_val_idx, test_idx = train_test_split(
        np.arange(len(dataset)), test_size=0.3, random_state=args['seed'], shuffle=True
    )
    train_val_subset = torch_data.Subset(dataset, train_val_idx)
    test_subset = torch_data.Subset(dataset, test_idx)

    # 初始化 KFold
    kf = KFold(n_splits=args['n_splits'], shuffle=True, random_state=args['seed'])

    # 用于跟踪最佳折的指标
    best_fold_idx = -1
    best_auc = -1
    best_metrics = {}
    best_model_path = 'weights/17_2k_100_best_model_PD_C.pth'

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_subset)):
        log_message(f"Fold {fold + 1}/{args['n_splits']}")

        # 划分训练和验证数据
        train_subset = torch_data.Subset(train_val_subset, train_idx)
        val_subset = torch_data.Subset(train_val_subset, val_idx)

        # 加载训练和验证数据
        train_loader = torch_data.DataLoader(
            dataset=train_subset,
            batch_size=args['batch_size'],
            shuffle=True,
            num_workers=args['num_workers']
        )
        val_loader = torch_data.DataLoader(
            dataset=val_subset,
            batch_size=args['batch_size'],
            shuffle=False,
            num_workers=args['num_workers']
        )

        # 初始化模型
        model = Model(
            num_class=args['num_class'],
            num_point=17,
            num_person=args['num_person'],
            graph='graph.ntu_rgb_d.Graph',
            graph_args=dict(labeling_mode='spatial'),
            in_channels=3
        )
        device = torch.device(args['device'])
        model = model.to(device)

        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=args['milestones'], gamma=args['gamma'])

        # 用于跟踪每个 epoch 的指标
        train_losses = []
        val_losses = []
        val_aucs = []

        # 训练循环
        for epoch in range(args['epochs']):
            model.train()
            total_loss = 0

            for batch_idx, (data, label, _) in enumerate(train_loader):
                data, label = data.to(device), label.float().to(device)
                label = label.view(-1, 1)

                # 确保输入数据会计算梯度
                data.requires_grad_()

                # 前向传播
                output = model(data)

                # 计算损失
                loss = criterion(output, label)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # 在每个 epoch 中，只对第一个批次生成热力图
                if batch_idx == 0:
                    plot_gradient_heatmap(data, data.grad, epoch, batch_idx, data.size(0), fold)

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            log_message(f"Epoch [{epoch + 1}/{args['epochs']}], Training Loss: {avg_loss:.4f}")

            # 学习率调整
            scheduler.step()

            # 验证过程
            model.eval()
            total_val_loss = 0
            all_labels = []
            all_outputs = []
            with torch.no_grad():
                for data, label, _ in val_loader:
                    data, label = data.to(device), label.float().to(device)
                    label = label.view(-1, 1)

                    # 前向传播
                    output = model(data)

                    # 计算验证损失
                    val_loss = criterion(output, label)
                    total_val_loss += val_loss.item()

                    all_labels.append(label.cpu().numpy())
                    all_outputs.append(output.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            log_message(f'Validation Loss: {avg_val_loss:.4f}')

            # 计算 AUC
            all_labels = np.vstack(all_labels)
            all_outputs = np.vstack(all_outputs)
            fpr, tpr, _ = roc_curve(all_labels, all_outputs)
            auc_score = auc(fpr, tpr)
            val_aucs.append(auc_score)
            log_message(f'Validation AUC: {auc_score:.4f}')

        # 根据 AUC 存储最佳折
        if max(val_aucs) > best_auc:
            best_auc = max(val_aucs)
            best_fold_idx = fold
            best_metrics = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_aucs': val_aucs
            }
            torch.save(model.state_dict(), best_model_path)

    log_message(f"Best model saved at {best_model_path}")

    # 在测试集上评估
    test_loader = torch_data.DataLoader(
        dataset=test_subset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args['num_workers']
    )
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_test_labels = []
    all_test_outputs = []
    with torch.no_grad():
        for data, label, _ in test_loader:
            data, label = data.to(device), label.float().to(device)
            label = label.view(-1, 1)

            # 前向传播
            output = model(data)

            all_test_labels.append(label.cpu().numpy())
            all_test_outputs.append(output.cpu().numpy())

    all_test_labels = np.vstack(all_test_labels)
    all_test_outputs = np.vstack(all_test_outputs)

    # 计算评估指标
    test_accuracy = accuracy_score(all_test_labels, all_test_outputs.round())
    test_precision = precision_score(all_test_labels, all_test_outputs.round())
    test_recall = recall_score(all_test_labels, all_test_outputs.round())
    test_f1 = f1_score(all_test_labels, all_test_outputs.round())
    fpr, tpr, _ = roc_curve(all_test_labels, all_test_outputs)
    test_auc = auc(fpr, tpr)

    log_message(f"Test Accuracy: {test_accuracy:.4f}")
    log_message(f"Test Precision: {test_precision:.4f}")
    log_message(f"Test Recall: {test_recall:.4f}")
    log_message(f"Test F1 Score: {test_f1:.4f}")
    log_message(f"Test AUC: {test_auc:.4f}")

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {test_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('17_PD_C_grad_roc_curve.png')
    log_message("ROC curve saved as 'PD_C_grad_roc_curve.png'")

    # 绘制训练和验证损失曲线
    plt.figure()
    epochs = range(1, len(best_metrics['train_losses']) + 1)
    plt.plot(epochs, best_metrics['train_losses'], 'b-', label='Training Loss')
    plt.plot(epochs, best_metrics['val_losses'], 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig('17_PD_C_grad_loss_curve.png')
    log_message("Loss curve saved as 'PD_C_grad_loss_curve.png'")

if __name__ == '__main__':
    main()