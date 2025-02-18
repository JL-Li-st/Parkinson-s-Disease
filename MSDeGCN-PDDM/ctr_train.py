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
from model.ctrgcn import Model
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
    filename='logs/ctrgcn.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Helper function to log and print messages
def log_message(message):
    print(message)
    logging.info(message)

# Main function for training and validation
def main():
    # Fixed arguments
    args = {
        'data_path': './data/',
        'label_path': './data/',
        'batch_size': 8,
        'num_workers': 4,
        'learning_rate': 0.0001,
        'epochs': 100,
        'milestones': [30, 40],
        'gamma': 0.1,
        'num_class': 1,  # Update to 1 for binary classification
        'num_person': 1,
        'seed': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_splits': 2
    }

    # Initialize random seed
    init_seed(args['seed'])
    log_message(f"Arguments: {args}")

    # Load dataset
    dataset = Feeder(data_folder=args['data_path'], label_file_path=args['label_path'], split='train')
    log_message("Dataset loaded successfully.")

    # Split dataset into 70% training/validation and 30% testing
    train_val_idx, test_idx = train_test_split(
        np.arange(len(dataset)), test_size=0.3, random_state=args['seed'], shuffle=True
    )
    train_val_subset = torch_data.Subset(dataset, train_val_idx)
    test_subset = torch_data.Subset(dataset, test_idx)

    # Initialize KFold
    kf = KFold(n_splits=args['n_splits'], shuffle=True, random_state=args['seed'])

    # Tracking metrics for the best fold
    best_fold_idx = -1
    best_auc = -1
    best_metrics = {}
    best_model_path = 'weights/ctr_best_model.pth'

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_subset)):
        # if fold < 3:  # 跳过第0折
        #     continue
        log_message(f"Fold {fold + 1}/{args['n_splits']}")

        # Split training and validation data
        train_subset = torch_data.Subset(train_val_subset, train_idx)
        val_subset = torch_data.Subset(train_val_subset, val_idx)

        # Load training and validation data
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

        # Initializing the model
        model = Model(
            num_class=args['num_class'],
            num_point=25,
            num_person=args['num_person'],
            graph='graph.ntu_rgb_d.Graph',
            graph_args=dict(labeling_mode='spatial'),
            in_channels=3
        )
        device = torch.device(args['device'])
        model = model.to(device)

        # Defining the loss function and optimizer,
        # weight_decay=1e-4：weight_decay（权重衰减）是一种正则化技术，用于防止模型过拟合。
        # 在每次参数更新时，会对参数进行一定程度的衰减，以限制参数的大小，提高模型的泛化能力。
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=args['milestones'], gamma=args['gamma'])

        # Tracking metrics for each epoch
        train_losses = []
        val_losses = []
        val_aucs = []

        # Training loop
        for epoch in range(args['epochs']):
            model.train()
            total_loss = 0

            for batch_idx, (data, label, _) in enumerate(train_loader):
                data, label = data.to(device), label.float().to(device)
                label = label.view(-1, 1)

                # Forward pass
                output = model(data)

                # Calculate loss
                loss = criterion(output, label)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            log_message(f"Epoch [{epoch + 1}/{args['epochs']}], Training Loss: {avg_loss:.4f}")

            # Adjust learning rate
            scheduler.step()

            # Validation process
            model.eval()
            total_val_loss = 0
            all_labels = []
            all_outputs = []
            with torch.no_grad():
                for data, label, _ in val_loader:
                    data, label = data.to(device), label.float().to(device)
                    label = label.view(-1, 1)

                    # Forward pass
                    output = model(data)

                    # Calculate validation loss
                    val_loss = criterion(output, label)
                    total_val_loss += val_loss.item()

                    all_labels.append(label.cpu().numpy())
                    all_outputs.append(output.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            log_message(f'Validation Loss: {avg_val_loss:.4f}')

            # Calculate AUC
            all_labels = np.vstack(all_labels)
            all_outputs = np.vstack(all_outputs)
            fpr, tpr, _ = roc_curve(all_labels, all_outputs)
            auc_score = auc(fpr, tpr)
            val_aucs.append(auc_score)
            log_message(f'Validation AUC: {auc_score:.4f}')

        # Store the best fold based on AUC
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

    # Evaluate on the test set
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

            # Forward pass
            output = model(data)

            all_test_labels.append(label.cpu().numpy())
            all_test_outputs.append(output.cpu().numpy())

    all_test_labels = np.vstack(all_test_labels)
    all_test_outputs = np.vstack(all_test_outputs)

    # Calculate evaluation metrics
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

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {test_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ctr_roc_curve.png')
    log_message("ROC curve saved as 'ctr_roc_curve.png'")

    # Plot Training and Validation Loss Curve
    plt.figure()
    epochs = range(1, len(best_metrics['train_losses']) + 1)
    plt.plot(epochs, best_metrics['train_losses'], 'b-', label='Training Loss')
    plt.plot(epochs, best_metrics['val_losses'], 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig('ctr_loss_curve.png')
    log_message("Loss curve saved as 'ctr_loss_curve.png'")

if __name__ == '__main__':
    main()
