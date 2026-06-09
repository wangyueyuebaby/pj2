# CS30064.01 Project 2

Author: Wang Yue

This repository contains the code for CS30064.01 Project 2:

- Task 1: custom CIFAR-10 classification network.
- Task 2: VGG-A vs VGG-A with Batch Normalization and loss landscape visualization.

## Files

```text
data/loaders.py              CIFAR-10 dataloader and augmentation
models/vgg.py                VGG-A and VGG-A-BatchNorm
task1_cifar10.py             Task 1 training script
VGG_Loss_Landscape.py        Task 2 loss landscape script
utils/nn.py                  Weight initialization utilities
reports/task1/               Task 1 json results and figures
reports/loss_landscape/      Task 2 json results
reports/figures/             Loss landscape figure
```

Large model weights are not stored in this Github repository. They should be downloaded from the model weight link in the report.

## Environment

The code automatically uses CUDA when available and falls back to CPU otherwise.

```bash
pip install torch torchvision numpy matplotlib tqdm ipython
```

## Reproduce Task 1

```bash
python task1_cifar10.py --epochs 40 --batch-size 128 --num-workers 4 --resume
```

## Reproduce Task 2

```bash
python VGG_Loss_Landscape.py --epochs 20 --batch-size 128 --learning-rates 0.001 0.002 0.0001 0.0005 --resume
```

## Results

Task 1 best model:

- `wider_gelu_smooth`
- Best test accuracy: 94.12%
- Best test error: 5.88%

Task 2:

- Best VGG-A accuracy: 78.40%
- Best VGG-A-BN accuracy: 82.54%

Batch Normalization improves optimization stability and allows VGG-A to train better under larger learning rates.

## Extended Analysis & Visualizations

The following scripts implement the advanced analysis required by the project rubric, providing deeper insights into optimization dynamics, loss landscapes, and network interpretability. They are intentionally configured with small subsets so the grader can quickly verify the code path and regenerate the figures in `./figs/`.

Optimizer comparison:

```bash
python optimizer_comparison.py --epochs 3 --n-items 2048 --val-items 1000 --batch-size 128 --num-workers 0 --output-dir ./figs
```

Network insight visualizations:

```bash
python network_insights.py --checkpoint ./reports/models/task1/base_relu_ce_best.pt --batch-size 128 --num-workers 0 --output-dir ./figs
```

Gradient predictiveness:

```bash
python gradient_predictiveness.py --n-items 512 --max-batches 4 --batch-size 128 --num-workers 0 --learning-rates 0.001 0.002 0.0001 0.0005 --output-dir ./figs
```

Generated files:

- `figs/optimizer_comparison.png`: AdamW vs SGD with momentum on `base_relu_ce`.
- `figs/first_layer_filters.png`: learned first-layer `3x3` filters.
- `figs/confusion_matrix.png`: CIFAR-10 test confusion matrix.
- `figs/gradient_predictiveness.png`: VGG-A vs VGG-A-BN gradient-change comparison.
