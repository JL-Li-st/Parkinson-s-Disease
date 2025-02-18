# import os
# import sys
# import numpy as np
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as torch_data
# from torch.optim.lr_scheduler import MultiStepLR
# from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, KFold
# import logging
#
# # Adding necessary paths
# sys.path.extend(['../'])
#
# # Importing the provided modules
# from feeders.feeder_ntu import Feeder
# from model.degcn import Model
# from graph.ntu_rgb_d import Graph
#
#
# # Set the seed for reproducibility
# def init_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#
# # Set up logging
# if not os.path.exists('logs'):
#     os.makedirs('logs')
#
# logging.basicConfig(
#     filename='logs/17_PD_C_PSA.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
#
#
# def log_message(message):
#     print(message)
#     logging.info(message)
#
#
# # Perturbation-based Sensitivity Analysis
# def perturbation_sensitivity_analysis(model, test_loader, device, perturbation_magnitude=0.05):
#     model.eval()
#     sensitivity_map = np.zeros(17)  # 17个关键点的敏感性
#
#     with torch.no_grad():
#         for data, label, _ in test_loader:
#             data = data.to(device)
#             base_output = model(data).cpu().numpy()
#
#             for joint in range(17):  # 遍历所有17个关键点
#                 perturbed_data = data.clone()
#                 perturbed_data[:, :, joint, :] += perturbation_magnitude * torch.randn_like(
#                     perturbed_data[:, :, joint, :])
#                 perturbed_output = model(perturbed_data).cpu().numpy()
#
#                 sensitivity_map[joint] += np.mean(np.abs(perturbed_output - base_output))
#
#     sensitivity_map /= len(test_loader)  # 取平均值
#
#     # 对结果进行排序
#     sorted_indices = np.argsort(sensitivity_map)[::-1]
#     sorted_sensitivity = sensitivity_map[sorted_indices]
#
#     # 输出排序后的敏感性数据
#     for idx, sens in zip(sorted_indices, sorted_sensitivity):
#         log_message(f"Joint {idx}: Sensitivity {sens:.6f}")
#
#     # 绘制敏感性分析结果
#     plt.figure(figsize=(10, 5))
#     plt.bar(range(17), sensitivity_map, color='b', alpha=0.6)
#     plt.xlabel('Joint Index')
#     plt.ylabel('Sensitivity')
#     plt.title('Perturbation-based Sensitivity Analysis')
#     plt.savefig('17_PD_C_sensitivity_analysis.png')
#     log_message("Sensitivity analysis saved as '17_PD_C_sensitivity_analysis.png'")
#
#     return sensitivity_map
#
#
# # Main function for training and validation
# def main():
#     args = {
#         'data_path': './data/PD_C/17_PD_C',
#         'label_path': './data/PD_C/labels.xlsx',
#         'batch_size': 8,
#         'num_workers': 4,
#         'learning_rate': 0.0001,
#         'epochs': 100,
#         'milestones': [30, 40],
#         'gamma': 0.1,
#         'num_class': 1,
#         'num_person': 1,
#         'seed': 1,
#         'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#         'n_splits': 2
#     }
#
#     init_seed(args['seed'])
#     log_message(f"Arguments: {args}")
#
#     dataset = Feeder(data_folder=args['data_path'], label_file_path=args['label_path'], split='train')
#     log_message("Dataset loaded successfully.")
#
#     train_val_idx, test_idx = train_test_split(
#         np.arange(len(dataset)), test_size=0.3, random_state=args['seed'], shuffle=True
#     )
#     test_subset = torch_data.Subset(dataset, test_idx)
#
#     test_loader = torch_data.DataLoader(
#         dataset=test_subset,
#         batch_size=args['batch_size'],
#         shuffle=False,
#         num_workers=args['num_workers']
#     )
#
#     model = Model(
#         num_class=args['num_class'],
#         num_point=17,  # Changed to 17 keypoints
#         num_person=args['num_person'],
#         graph='graph.ntu_rgb_d.Graph',
#         graph_args=dict(labeling_mode='spatial'),
#         in_channels=3
#     )
#     device = torch.device(args['device'])
#     model = model.to(device)
#
#     model.load_state_dict(torch.load('weights/17_2k_100_best_model_PD_C.pth'))
#     log_message("Loaded best trained model.")
#
#     # 进行 Perturbation-based Sensitivity Analysis
#     sensitivity_map = perturbation_sensitivity_analysis(model, test_loader, device)
#     log_message(f"Sorted Sensitivity Map: {sensitivity_map}")
#
#
# if __name__ == '__main__':
#     main()
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
    filename='logs/17_PD_C_PSA.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def log_message(message):
    print(message)
    logging.info(message)


# Perturbation-based Sensitivity Analysis
def perturbation_sensitivity_analysis(model, test_loader, device, perturbation_magnitude=0.05):
    model.eval()
    sensitivity_map = np.zeros(17)  # 17个关键点的敏感性

    with torch.no_grad():
        for data, label, _ in test_loader:
            data = data.to(device)
            base_output = model(data).cpu().numpy()

            for joint in range(17):  # 遍历所有17个关键点
                perturbed_data = data.clone()
                perturbed_data[:, :, joint, :] += perturbation_magnitude * torch.randn_like(
                    perturbed_data[:, :, joint, :])
                perturbed_output = model(perturbed_data).cpu().numpy()

                sensitivity_map[joint] += np.mean(np.abs(perturbed_output - base_output))

    sensitivity_map /= len(test_loader)  # 取平均值

    # 对结果进行排序
    sorted_indices = np.argsort(sensitivity_map)[::-1]
    sorted_sensitivity = sensitivity_map[sorted_indices]

    # 输出排序后的敏感性数据
    for idx, sens in zip(sorted_indices, sorted_sensitivity):
        log_message(f"Joint {idx}: Sensitivity {sens:.6f}")

    # 绘制敏感性分析结果
    plt.figure(figsize=(10, 5))
    plt.bar(range(17), sensitivity_map, color='b', alpha=0.6)
    plt.xlabel('Joint Index')
    plt.ylabel('Sensitivity')
    plt.title('Perturbation-based Sensitivity Analysis')

    # 显式设置横轴刻度
    plt.xticks(ticks=range(17), labels=range(17))

    plt.savefig('17_Turning_sensitivity_analysis.png')
    log_message("Sensitivity analysis saved as '17_Turning_sensitivity_analysis.png'")

    return sensitivity_map


# Main function for training and validation
def main():
    args = {
        'data_path': './data/',
        'label_path': './data/',
        'batch_size': 8,
        'num_workers': 4,
        'learning_rate': 0.0001,
        'epochs': 100,
        'milestones': [30, 40],
        'gamma': 0.1,
        'num_class': 1,
        'num_person': 1,
        'seed': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_splits': 2
    }

    init_seed(args['seed'])
    log_message(f"Arguments: {args}")

    dataset = Feeder(data_folder=args['data_path'], label_file_path=args['label_path'], split='train')
    log_message("Dataset loaded successfully.")

    train_val_idx, test_idx = train_test_split(
        np.arange(len(dataset)), test_size=0.3, random_state=args['seed'], shuffle=True
    )
    test_subset = torch_data.Subset(dataset, test_idx)

    test_loader = torch_data.DataLoader(
        dataset=test_subset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args['num_workers']
    )

    model = Model(
        num_class=args['num_class'],
        num_point=17,  # Changed to 17 keypoints
        num_person=args['num_person'],
        graph='graph.ntu_rgb_d.Graph',
        graph_args=dict(labeling_mode='spatial'),
        in_channels=3
    )
    device = torch.device(args['device'])
    model = model.to(device)

    model.load_state_dict(torch.load('weights/17_2k_100_best_model_PD_C.pth'))
    log_message("Loaded best trained model.")

    # 进行 Perturbation-based Sensitivity Analysis
    sensitivity_map = perturbation_sensitivity_analysis(model, test_loader, device)
    log_message(f"Sorted Sensitivity Map: {sensitivity_map}")


if __name__ == '__main__':
    main()
