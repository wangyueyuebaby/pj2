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
from torch import nn
from tqdm import tqdm

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters


def set_random_seeds(seed_value=2020, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(device_arg):
    if device_arg != 'auto':
        return torch.device(device_arg)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            correct += prediction.argmax(dim=1).eq(y).sum().item()
            total += y.size(0)
    return correct / total


def get_grad_norm(model):
    grad_square_sum = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad_square_sum += parameter.grad.detach().pow(2).sum().item()
    return grad_square_sum ** 0.5


def train(model, optimizer, criterion, train_loader, val_loader, device, epochs_n=20, best_model_path=None,
          checkpoint_path=None, partial_path=None, metadata=None, resume=False):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    losses_list = []
    grad_norms = []
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0
    start_epoch = 0

    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losses_list = checkpoint.get('step_losses', [])
        grad_norms = checkpoint.get('step_grad_norms', [])
        previous_learning = checkpoint.get('learning_curve', [])
        previous_train_acc = checkpoint.get('train_accuracy_curve', [])
        previous_val_acc = checkpoint.get('val_accuracy_curve', [])
        for idx, value in enumerate(previous_learning[:epochs_n]):
            learning_curve[idx] = value
        for idx, value in enumerate(previous_train_acc[:epochs_n]):
            train_accuracy_curve[idx] = value
        for idx, value in enumerate(previous_val_acc[:epochs_n]):
            val_accuracy_curve[idx] = value
        max_val_accuracy = checkpoint.get('best_val_accuracy', 0)
        max_val_accuracy_epoch = checkpoint.get('best_val_accuracy_epoch', 0) - 1
        start_epoch = checkpoint.get('epoch', 0)
        print('Resumed {} from epoch {}'.format(metadata.get('run_name', 'run') if metadata else 'run', start_epoch))

    batches_n = len(train_loader)
    for epoch in tqdm(range(start_epoch, epochs_n), unit='epoch'):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()

            losses_list.append(loss.item())
            grad_norms.append(get_grad_norm(model))

            optimizer.step()
            epoch_loss += loss.item()
            epoch_correct += prediction.argmax(dim=1).eq(y).sum().item()
            epoch_total += y.size(0)

        learning_curve[epoch] = epoch_loss / batches_n
        train_accuracy_curve[epoch] = epoch_correct / epoch_total
        val_accuracy_curve[epoch] = get_accuracy(model, val_loader, device)
        if val_accuracy_curve[epoch] > max_val_accuracy:
            max_val_accuracy = val_accuracy_curve[epoch]
            max_val_accuracy_epoch = epoch
            if best_model_path is not None:
                torch.save(model.state_dict(), best_model_path)

        result = {
            'step_losses': losses_list,
            'step_grad_norms': grad_norms,
            'learning_curve': learning_curve,
            'train_accuracy_curve': train_accuracy_curve,
            'val_accuracy_curve': val_accuracy_curve,
            'best_val_accuracy': max_val_accuracy,
            'best_val_accuracy_epoch': max_val_accuracy_epoch + 1
        }
        if checkpoint_path is not None:
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                **result
            }, checkpoint_path)
        if partial_path is not None:
            partial_result = dict(result)
            if metadata is not None:
                partial_result.update(metadata)
            with open(partial_path, 'w') as f:
                json.dump(partial_result, f, indent=2)

        print('epoch {}/{} loss={:.4f} train_acc={:.4f} val_acc={:.4f}'.format(
            epoch + 1, epochs_n, learning_curve[epoch],
            train_accuracy_curve[epoch], val_accuracy_curve[epoch]))

    return {
        'step_losses': losses_list,
        'step_grad_norms': grad_norms,
        'learning_curve': learning_curve,
        'train_accuracy_curve': train_accuracy_curve,
        'val_accuracy_curve': val_accuracy_curve,
        'best_val_accuracy': max_val_accuracy,
        'best_val_accuracy_epoch': max_val_accuracy_epoch + 1
    }


def compute_loss_band(loss_curves):
    min_length = min(len(curve) for curve in loss_curves)
    clipped = np.array([curve[:min_length] for curve in loss_curves])
    return clipped.min(axis=0), clipped.max(axis=0), clipped.mean(axis=0)


def run_lr_sweep(model_cls, model_name, learning_rates, args, train_loader, val_loader, device):
    results = []
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    start = time.time()

    for lr in learning_rates:
        print('\nTraining {} with lr={}'.format(model_name, lr))
        set_random_seeds(args.seed, device.type)
        model = model_cls()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        best_model_path = os.path.join(args.model_dir, '{}_lr_{}.pt'.format(model_name, lr))
        checkpoint_path = os.path.join(args.model_dir, '{}_lr_{}_checkpoint.pt'.format(model_name, lr))
        partial_path = os.path.join(args.output_dir, '{}_lr_{}_partial.json'.format(model_name, lr))
        metadata = {
            'run_name': '{}_lr_{}'.format(model_name, lr),
            'model_name': model_name,
            'learning_rate': lr,
            'parameters': get_number_of_parameters(model),
            'best_model_path': best_model_path,
            'checkpoint_path': checkpoint_path
        }
        result = train(
            model,
            optimizer,
            criterion,
            train_loader,
            val_loader,
            device,
            epochs_n=args.epochs,
            best_model_path=best_model_path,
            checkpoint_path=checkpoint_path,
            partial_path=partial_path,
            metadata=metadata,
            resume=args.resume)
        result.update(metadata)
        results.append(result)

    print('{} sweep finished in {:.1f}s'.format(model_name, time.time() - start))
    return results


def plot_loss_landscape(vgg_results, bn_results, figure_path):
    vgg_min, vgg_max, vgg_mean = compute_loss_band([r['step_losses'] for r in vgg_results])
    bn_min, bn_max, bn_mean = compute_loss_band([r['step_losses'] for r in bn_results])
    steps = np.arange(min(len(vgg_min), len(bn_min)))

    plt.figure(figsize=(8, 5))
    plt.fill_between(steps, vgg_min[:len(steps)], vgg_max[:len(steps)],
                     color='tab:orange', alpha=0.25, label='VGG-A min-max')
    plt.plot(steps, vgg_mean[:len(steps)], color='tab:orange', linewidth=1.5, label='VGG-A mean')
    plt.fill_between(steps, bn_min[:len(steps)], bn_max[:len(steps)],
                     color='tab:blue', alpha=0.25, label='VGG-A-BN min-max')
    plt.plot(steps, bn_mean[:len(steps)], color='tab:blue', linewidth=1.5, label='VGG-A-BN mean')
    plt.xlabel('Training step')
    plt.ylabel('Cross entropy loss')
    plt.title('Loss Landscape: VGG-A vs VGG-A with BatchNorm')
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path, dpi=180)
    plt.close()


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Task 2 VGG-A BatchNorm loss landscape.')
    parser.add_argument('--data-root', default='./data/')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--n-items', default=-1, type=int,
                        help='Use a positive value for a partial training set smoke test.')
    parser.add_argument('--val-items', default=-1, type=int,
                        help='Use a positive value for a partial test set smoke test.')
    parser.add_argument('--learning-rates', nargs='+', default=[1e-3, 2e-3, 1e-4, 5e-4], type=float)
    parser.add_argument('--weight-decay', default=0.0, type=float)
    parser.add_argument('--seed', default=2020, type=int)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-dir', default='./reports/loss_landscape')
    parser.add_argument('--figure-path', default='./reports/figures/loss_landscape_bn_compare.png')
    parser.add_argument('--model-dir', default='./reports/models/task2')
    parser.add_argument('--resume', action='store_true',
                        help='Resume each model/lr run from checkpoints when available.')
    parser.add_argument('--models', nargs='*', default=None, choices=['vgg_a', 'vgg_a_bn'],
                        help='Optional subset of model names to run.')
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device)
    print('Using device: {}'.format(device))

    train_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=True,
        shuffle=True,
        num_workers=args.num_workers,
        n_items=args.n_items)
    val_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.val_items)

    selected_models = args.models or ['vgg_a', 'vgg_a_bn']
    vgg_results = []
    bn_results = []
    if 'vgg_a' in selected_models:
        vgg_results = run_lr_sweep(
            VGG_A, 'vgg_a', args.learning_rates, args, train_loader, val_loader, device)
    if 'vgg_a_bn' in selected_models:
        bn_results = run_lr_sweep(
            VGG_A_BatchNorm, 'vgg_a_bn', args.learning_rates, args, train_loader, val_loader, device)

    if vgg_results and bn_results:
        plot_loss_landscape(vgg_results, bn_results, args.figure_path)
        print('Saved loss landscape figure to {}'.format(args.figure_path))
    save_json(os.path.join(args.output_dir, 'vgg_a_results.json'), vgg_results)
    save_json(os.path.join(args.output_dir, 'vgg_a_bn_results.json'), bn_results)


if __name__ == '__main__':
    main()
