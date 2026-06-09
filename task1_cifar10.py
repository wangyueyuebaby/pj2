import argparse
import json
import os
import random
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data.loaders import get_cifar_loader
from models.vgg import get_number_of_parameters


def set_random_seeds(seed_value=2020, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_activation(name):
    if name == 'relu':
        return nn.ReLU(True)
    if name == 'leaky_relu':
        return nn.LeakyReLU(0.1, inplace=True)
    if name == 'elu':
        return nn.ELU(inplace=True)
    if name == 'gelu':
        return nn.GELU()
    if name == 'silu':
        return nn.SiLU(inplace=True)
    raise ValueError('Unknown activation: {}'.format(name))


class ConvBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, activation='relu', stride=1, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(inp_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            get_activation(activation)
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, activation='relu', stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBlock(inp_ch, out_ch, activation, stride=stride, dropout=dropout)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch))
        if stride != 1 or inp_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
        else:
            self.shortcut = nn.Identity()
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return self.activation(out)


class CIFARProjectNet(nn.Module):
    """Compact residual CNN for Task 1 CIFAR-10 experiments."""

    def __init__(self, channels=(64, 128, 256), activation='relu', dropout=0.1, num_classes=10):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            ConvBlock(3, c1, activation, dropout=dropout),
            ResidualBlock(c1, c1, activation, dropout=dropout),
            ResidualBlock(c1, c2, activation, stride=2, dropout=dropout),
            ResidualBlock(c2, c2, activation, dropout=dropout),
            ResidualBlock(c2, c3, activation, stride=2, dropout=dropout),
            ResidualBlock(c3, c3, activation, dropout=dropout),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes))

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, prediction, target):
        ce = F.cross_entropy(
            prediction,
            target,
            reduction='none',
            label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_criterion(name):
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    if name == 'label_smoothing':
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    if name == 'focal':
        return FocalLoss(gamma=2.0, label_smoothing=0.05)
    raise ValueError('Unknown loss: {}'.format(name))


def get_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            loss_sum += criterion(prediction, y).item()
            correct += prediction.argmax(dim=1).eq(y).sum().item()
            total += y.size(0)
    return correct / total, loss_sum / total


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum = 0
    correct = 0
    total = 0
    start = time.time()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        prediction = model(x)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        loss_sum += loss.item() * batch_size
        correct += prediction.argmax(dim=1).eq(y).sum().item()
        total += batch_size
    return {
        'loss': loss_sum / total,
        'accuracy': correct / total,
        'seconds': time.time() - start,
        'images_per_second': total / max(time.time() - start, 1e-8)
    }


def run_experiment(config, args, device):
    print('Running {}'.format(config['name']))
    train_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=True,
        shuffle=True,
        num_workers=args.num_workers,
        n_items=args.n_items,
        augment=True)
    val_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.val_items)

    model = CIFARProjectNet(
        channels=config['channels'],
        activation=config['activation'],
        dropout=config['dropout']).to(device)
    criterion = build_criterion(config['loss'])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.model_dir, '{}_best.pt'.format(config['name']))
    checkpoint_path = os.path.join(args.model_dir, '{}_checkpoint.pt'.format(config['name']))
    partial_path = os.path.join(args.output_dir, '{}_partial.json'.format(config['name']))
    best_acc = 0
    history = []
    parameters_n = get_number_of_parameters(model)
    start_epoch = 0

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_acc = checkpoint.get('best_acc', 0)
        history = checkpoint.get('history', [])
        start_epoch = checkpoint.get('epoch', len(history))
        print('Resumed {} from epoch {}'.format(config['name'], start_epoch))

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_loss = get_accuracy(model, val_loader, device)
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
        row = {
            'epoch': epoch + 1,
            'lr': scheduler.get_last_lr()[0],
            'train_loss': train_stats['loss'],
            'train_accuracy': train_stats['accuracy'],
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epoch_seconds': train_stats['seconds'],
            'images_per_second': train_stats['images_per_second']
        }
        history.append(row)
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc': best_acc,
            'history': history,
            'config': config,
            'parameters': parameters_n
        }, checkpoint_path)
        with open(partial_path, 'w') as f:
            json.dump({
                'config': config,
                'parameters': parameters_n,
                'best_val_accuracy': best_acc,
                'best_test_error': 1 - best_acc,
                'best_model_path': best_path,
                'checkpoint_path': checkpoint_path,
                'history': history
            }, f, indent=2)
        print('epoch {}/{} train_acc={:.4f} val_acc={:.4f} val_error={:.4f} sec={:.1f}'.format(
            epoch + 1, args.epochs, row['train_accuracy'], row['val_accuracy'],
            1 - row['val_accuracy'], row['epoch_seconds']))

    return {
        'config': config,
        'parameters': parameters_n,
        'best_val_accuracy': best_acc,
        'best_test_error': 1 - best_acc,
        'best_model_path': best_path,
        'checkpoint_path': checkpoint_path,
        'history': history
    }


def plot_histories(results, figure_path):
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    for result in results:
        val_error = [1 - row['val_accuracy'] for row in result['history']]
        plt.plot(range(1, len(val_error) + 1), val_error, label=result['config']['name'])
    plt.xlabel('Epoch')
    plt.ylabel('Test error')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Task 1 CIFAR-10 training script.')
    parser.add_argument('--data-root', default='./data/')
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--n-items', default=-1, type=int,
                        help='Use a positive value for a partial training set smoke test.')
    parser.add_argument('--val-items', default=-1, type=int,
                        help='Use a positive value for a partial test set smoke test.')
    parser.add_argument('--seed', default=2020, type=int)
    parser.add_argument('--output-dir', default='./reports/task1')
    parser.add_argument('--model-dir', default='./reports/models/task1')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from per-experiment checkpoints when available.')
    parser.add_argument('--experiments', nargs='*', default=None,
                        help='Optional subset of experiment names to run.')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_random_seeds(args.seed, device.type)
    os.makedirs(args.output_dir, exist_ok=True)

    configs = [
        {
            'name': 'base_relu_ce',
            'channels': (64, 128, 256),
            'activation': 'relu',
            'loss': 'cross_entropy',
            'dropout': 0.10,
            'lr': 1e-3,
            'weight_decay': 5e-4
        },
        {
            'name': 'wider_gelu_smooth',
            'channels': (96, 192, 384),
            'activation': 'gelu',
            'loss': 'label_smoothing',
            'dropout': 0.15,
            'lr': 8e-4,
            'weight_decay': 1e-3
        },
        {
            'name': 'compact_leaky_focal',
            'channels': (48, 96, 192),
            'activation': 'leaky_relu',
            'loss': 'focal',
            'dropout': 0.10,
            'lr': 1e-3,
            'weight_decay': 5e-4
        }
    ]

    if args.experiments:
        configs = [config for config in configs if config['name'] in args.experiments]
    results = [run_experiment(config, args, device) for config in configs]
    json_path = os.path.join(args.output_dir, 'task1_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    plot_histories(results, os.path.join(args.output_dir, 'task1_test_error.png'))
    print('Saved results to {}'.format(json_path))


if __name__ == '__main__':
    main()
