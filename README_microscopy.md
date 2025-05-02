
# 🧪 Microscopy Image Classification

This repository contains code for training and evaluating modified ResNet50 models on grayscale phase contrast microscopy images. It supports both classification using a fully connected layer (len=4) and contrastive learning using compact embeddings (e.g., 128D), enabling robust evaluation via k-nearest neighbors (k-NN) and tile-level voting aggregation for slide-level prediction.

## 🔍 Key Features

- Support for **cross-entropy**, **contrastive**, and **Earth Mover’s Distance** losses.
- **ResNet and CBAM-ResNet** backbones for grayscale image input.
- Embedding-based **k-NN evaluation** and classification.
- **Tiling** and **voting strategies** to reassemble whole-slide predictions.
- Optional **thresholding** and **blurring** for image preprocessing.
- Modular and reproducible training, validation, and testing pipelines.

## 📦 Requirements

- Python 3.8+
- PyTorch >= 1.10
- torchvision
- numpy
- scikit-learn
- Pillow
- tqdm
- (Optional) seaborn, matplotlib for visualization

Install all dependencies via:

```bash
pip install -r requirements.txt
```

## 🛠️ Dataset Structure

Organize your datasets (training/validation/test) as follows:

```
dataset/
  ├── class_0/
  │   ├── image_1.png
  │   ├── image_2.png
  │   └── ...
  ├── class_1/
  │   └── ...
```

You can pass this directory via `--train_dir` and `--test_dir`.

## 🚀 Example Usage

### Train a Classifier
```bash
python train_classifier.py \
  --model resnet_cbam \
  --train_dir ./data/train \
  --test_dir ./data/test \
  --num_classes 4 \
  --device cuda
```

### Train with Contrastive Loss
```bash
python train_contrastive.py \
  --loss_function contrastive \
  --loss_type dynamic \
  --embedding_size 128 \
  --train_dir ./data/train \
  --test_dir ./data/test \
  --num_classes 4
```

### Test with k-NN Evaluation
```bash
python test_model.py \
  --eval_strategy knn \
  --model resnet_cbam \
  --weights_path ./models/best_model.pth \
  --test_dir ./data/test \
  --output_json results.json \
  --num_classes 4
```

### Reassemble and Tint Tiled Predictions
```bash
python reconstruct_image.py \
  --json results.json \
  --blend_red_blue
```

## ⚙️ Command-Line Arguments

### 🧠 Model & Training Options

| Argument | Description |
|----------|-------------|
| `--model` | Model type: `resnet_reg` or `resnet_cbam`. |
| `--no_pretrained_weights` | Disable loading pretrained weights. |
| `--weights_path` | Path to `.pth` model weights file (or `None`). |

### 🧪 Hyperparameters

| Argument | Description |
|----------|-------------|
| `--num_epochs` | Number of training epochs. |
| `--patience` | Patience for early stopping. |
| `--num_classes` | **(Required)** Number of classes in dataset. |
| `--batch_size` | Batch size. |
| `--learning_rate` | Learning rate. |
| `--use_scheduler` | Enable StepLR learning rate scheduler. |
| `--optimizer` | Optimizer type: `adam` or `sgd`. |
| `--loss_function` | Loss function: `cross_entropy`, `contrastive`, or `earth_movers`. |
| `--loss_type` | Use of penalty matrix: `regular` or `dynamic`. |
| `--embedding_size` | Size of the embedding vector. |
| `--k` | Number of neighbors for k-NN evaluation. |
| `--contrast_margin` | Margin for contrastive loss. |

### 🗂️ Dataset & Preprocessing

| Argument | Description |
|----------|-------------|
| `--train_dir` | **(Required)** Path to training dataset. |
| `--test_dir` | **(Required)** Path to test dataset. |
| `--apply_thresholding` | Enable pixel-intensity thresholding. |
| `--std_percentage` | Std-dev percentage for thresholding range. |
| `--no_blurring` | Disable Gaussian blurring. |
| `--plot_sne` | Enable t-SNE visualization of embeddings. |

### 🧮 Other Options

| Argument | Description |
|----------|-------------|
| `--device` | Choose `cuda` or `cpu`. |
| `--use_cross_validation` | Train using k-fold cross-validation. |

### 💾 Output & Logging

| Argument | Description |
|----------|-------------|
| `--no_save_models` | Prevent saving trained models. |
| `--no_log_output` | Prevent writing `.log` output file. |

## 📊 Results

Model predictions and metadata (image name, predicted class, etc.) are stored in JSON format for further analysis. This format supports:
- Tile-level reassembly
- Visualization
- Voting-based aggregation

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for full details.
