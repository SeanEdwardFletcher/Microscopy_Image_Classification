
# 🧠 Training Guide

## 🧠 Training Script Overview

The training logic is implemented in `smp_training_script.py`. It supports standard classification (via cross-entropy or Earth Mover's loss) and contrastive learning with dynamic or fixed margins. If `--use_cross_validation` is passed, k-fold training is used (5-fold by default). Otherwise, it uses a traditional train/validation split.

## 🔧 Training Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Model type: `resnet_reg` or `resnet_cbam`. |
| `--no_pretrained_weights` | Disable loading pretrained weights. |
| `--weights_path` | Path to `.pth` model weights file (or `None`). |
| `--num_epochs` | Number of training epochs. |
| `--patience` | Early stopping patience. |
| `--num_classes` | **(Required)** Number of output classes. |
| `--batch_size` | Training batch size. |
| `--learning_rate` | Learning rate for the optimizer. |
| `--use_scheduler` | Use StepLR learning rate scheduler. |
| `--optimizer` | Optimizer: `adam` or `sgd`. |
| `--loss_function` | `cross_entropy`, `contrastive`, or `earth_movers`. |
| `--loss_type` | Type of margin matrix: `regular` or `dynamic`. |
| `--embedding_size` | Size of the learned embedding vector. |
| `--k` | Number of neighbors for k-NN (not for CV). |
| `--contrast_margin` | Margin for contrastive loss. |
| `--train_dir` | **(Required)** Path to training dataset. |
| `--test_dir` | **(Required)** Path to test dataset. |
| `--apply_thresholding` | Apply image intensity thresholding. |
| `--std_percentage` | Used in thresholding; controls dynamic range. |
| `--no_blurring` | Disable Gaussian blurring (enabled by default). |
| `--plot_sne` | Visualize embeddings with t-SNE after training. |
| `--device` | Compute device: `cuda` or `cpu`. |
| `--use_cross_validation` | Enable 5-fold cross-validation. |
| `--no_save_models` | Disable saving model checkpoints. |
| `--no_log_output` | Disable logging to `.log` file. |

## 🚀 Example Usage

### Training
```bash
python smp_training_script.py   --model resnet_cbam   --train_dir ./data/train   --test_dir ./data/test   --num_classes 4   --device cuda
```

### Train with Contrastive Loss
```bash
python smp_training_script.py   --model resnet_cbam   --loss_function contrastive   --loss_type dynamic   --embedding_size 128   --contrast_margin 1.0   --train_dir ./data/train   --test_dir ./data/test   --num_classes 4   --device cuda
```

