import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import datasets, transforms

from models.AlexNet import AlexNet
from models.EfficientNet import EfficientNet
from models.GoogLeNet import GoogLeNet
from models.LeNet import LeNet
from models.MNASNet import MNASNet
from models.MobileNetV3Small import MobileNetV3Small
from models.ResNet18 import ResNet18
from models.ShuffleNetV2 import ShuffleNetV2
from models.SqueezeNet import SqueezeNet
from models.VGG11 import VGG11


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lenet',
                        choices=['lenet', 'alexnet', 'resnet18', 'googlenet', 'vgg11', 'efficientnet',
                                 'mnasnet', 'mobilenetv3', 'shufflenetv2', 'squeezenet'])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--weight_decay', type=int, default=0.0001)
    parser.add_argument('--lr_scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'])
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--lr_factor', type=float, default=0.1)
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--warmup_start_lr', type=float, default=1e-6)
    return parser.parse_args()


class WarmupLRScheduler:
    def __init__(self, optimizer, warmup_epochs, warmup_start_lr, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr
        self.last_epoch = -1

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # 线性预热
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * epoch / self.warmup_epochs
        else:
            # 使用基础学习率
            lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.last_epoch = epoch
        return lr


def get_transforms(training=True):
    if training:
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomApply([
                transforms.RandomRotation(15),  # 随机旋转±15度
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),  # 随机平移±10%
                    scale=(0.9, 1.1),  # 随机缩放±10%
                ),
            ], p=0.5),
            transforms.ColorJitter(
                brightness=0.2,  # 随机亮度调整±20%
                contrast=0.2,  # 随机对比度调整±20%
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # 随机擦除
        ])
    else:
        # 测试时不使用数据增强
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    return transform


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc


def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          warmup_scheduler, device, epochs, patience, model_name, lr_scheduler):
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    lrs = []
    val_loss = float('inf')
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm.tqdm(range(epochs + 1), desc='Training', unit='epoch'):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # 更新warmup学习率
        if epoch <= warmup_scheduler.warmup_epochs:
            warmup_scheduler.step(epoch)
        elif scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model with model name
            torch.save(model.state_dict(), f'./states/best_model_{model_name}_{lr_scheduler}.pth')
            print('Find new best model')
        else:
            patience_counter += 1
            print(f'Patience counter: {patience_counter}/{patience}')
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    return train_losses, train_accs, val_losses, val_accs, lrs


def plot_metrics(train_losses, train_accs, val_losses, val_accs, lrs, model_name, lr_scheduler):
    plt.style.use('ggplot')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    # Plot learning rates
    ax3.plot(lrs, label='Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(f'./plots/training_metrics_{model_name}_{lr_scheduler}.png')
    plt.close()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create states and plots directories if they don't exist
    try:
        os.makedirs('./states')
        os.makedirs('./plots')
    except FileExistsError:
        pass

    # 使用不同的transform用于训练集和验证/测试集
    train_transform = get_transforms(training=True)
    test_transform = get_transforms(training=False)

    # 训练集
    dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # 创建验证集时使用测试transform
    val_dataset = datasets.MNIST('./data', train=True, download=True, transform=test_transform)

    # 随机划分训练集和验证集
    train_indices = torch.randperm(len(dataset))[:train_size]
    val_indices = torch.randperm(len(dataset))[train_size:train_size + val_size]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=4, pin_memory=True)

    # Model selection
    if args.model == 'lenet':
        model = LeNet(dropout_rate=args.dropout)
        best_model = LeNet(dropout_rate=args.dropout)
    elif args.model == 'alexnet':
        model = AlexNet(dropout_rate=args.dropout)
        best_model = AlexNet(dropout_rate=args.dropout)
    elif args.model == 'resnet18':
        model = ResNet18(dropout_rate=args.dropout)
        best_model = ResNet18(dropout_rate=args.dropout)
    elif args.model == 'googlenet':
        model = GoogLeNet(dropout_rate=args.dropout)
        best_model = GoogLeNet(dropout_rate=args.dropout)
    elif args.model == 'vgg11':
        model = VGG11(dropout_rate=args.dropout)
        best_model = VGG11(dropout_rate=args.dropout)
    elif args.model == 'mnasnet':
        model = MNASNet(dropout_rate=args.dropout)
        best_model = MNASNet(dropout_rate=args.dropout)
    elif args.model == 'mobilenetv3':
        model = MobileNetV3Small(dropout_rate=args.dropout)
        best_model = MobileNetV3Small(dropout_rate=args.dropout)
    elif args.model == 'shufflenetv2':
        model = ShuffleNetV2(dropout_rate=args.dropout)
        best_model = ShuffleNetV2(dropout_rate=args.dropout)
    elif args.model == 'squeezenet':
        model = SqueezeNet(dropout_rate=args.dropout)
        best_model = SqueezeNet(dropout_rate=args.dropout)
    elif args.model == 'efficientnet':
        model = EfficientNet(dropout_rate=args.dropout)
        best_model = EfficientNet(dropout_rate=args.dropout)
    else:
        raise ValueError('Invalid model name')

    model = model.to(device)
    summary(model, (1, 28, 28))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.warmup_start_lr, weight_decay=args.weight_decay)

    warmup_scheduler = WarmupLRScheduler(
        optimizer,
        args.warmup_epochs,
        args.warmup_start_lr,
        args.lr
    )

    # 设置学习率调度器
    if args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor,
            patience=args.lr_patience, min_lr=args.min_lr
        )
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs,
            eta_min=args.min_lr
        )
    else:
        scheduler = None

    start_train = time.time()
    # Training
    metrics = train(model, train_loader, val_loader, criterion, optimizer,
                    scheduler, warmup_scheduler, device, args.epochs,
                    args.patience, args.model, args.lr_scheduler)
    end_train = time.time()
    print(f'\nTraining time: {end_train - start_train:.2f} seconds')

    # Plot results
    plot_metrics(*metrics, args.model, args.lr_scheduler)

    # Load best model and evaluate on test set
    test_dataset = datasets.MNIST('./data', train=False, download=True,
                                  transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=4, pin_memory=True)

    best_model.to(device)
    best_model.load_state_dict(torch.load(f'./states/best_model_{args.model}_{args.lr_scheduler}.pth',
                                          weights_only=False))

    test_start = time.time()
    test_loss, test_acc = evaluate_model(best_model, test_loader, criterion, device)
    test_end = time.time()
    print('\nTest set results:')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    print(f'Test time: {test_end - test_start:.2f} seconds')

    # if no csv file exists, create one and write the header
    try:
        with open('results.csv', 'r'):
            pass
    except FileNotFoundError:
        with open('results.csv', 'a') as f:
            f.write(
                '''Model,Scheduler,Test Loss,Test Accuracy,Best Train Acc,Best Val Acc,Training time,Test time,Epochs,Time/Epoch,Epoches to Get 99% Val Acc,Time to Get 99% Val Acc,Params,Size\n''')

    actual_epochs = len(metrics[0]) - 1
    best_val_acc = max(metrics[3])
    best_train_acc = max(metrics[1])
    # how many epochs to reach 99% val acc
    epoch_99_val_acc = next((i for i, val_acc in enumerate(metrics[3]) if val_acc >= 99), actual_epochs)

    # Save all statistics in a csv file
    with open('results.csv', 'a') as f:
        f.write(
            f'''{args.model},{args.lr_scheduler},{test_loss:.4f},{test_acc:.2f},{best_train_acc:.2f},{best_val_acc:.2f},{end_train - start_train:.2f},{test_end - test_start:.2f},{actual_epochs},{(end_train - start_train) / actual_epochs:.2f},{epoch_99_val_acc},{(end_train - start_train) * epoch_99_val_acc / actual_epochs:.2f},{sum(p.numel() for p in model.parameters())},{os.path.getsize(f"./states/best_model_{args.model}_{args.lr_scheduler}.pth")}\n''')


if __name__ == '__main__':
    main()
