
# 🧪 Microscopy Image Classification

## 📘 Project Overview

The training process is handled by the `smp_training_script.py` file and supports both standard classification and contrastive learning. Depending on the chosen loss function and model architecture, it dynamically prepares dataloaders, initializes the appropriate model (with optional pretrained weights), and sets up the optimizer, scheduler, and evaluation pipeline. If `--use_cross_validation` is enabled, the training is performed using k-fold cross-validation; otherwise, a single training/validation split is used. Training supports three types of loss functions: `cross_entropy`, `contrastive`, and `earth_movers`, and includes augmentation strategies suitable for microscopy data.

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
pip install -r smp_requirements.txt
```

## 🛠️ Dataset Structure

To support group-based evaluation and the use of loss functions tailored for continuous or ordinal datasets, prefix each class folder and its contained images with a unique uppercase letter (e.g., `A`, `B`, `C`, etc.). This naming convention ensures compatibility with downstream voting mechanisms and enables proper interpretation of class relationships during evaluation.

Organize your datasets (training/validation/test) as follows:

```
dataset/
  ├── A_class01/
  │   ├── A_image_1.png
  │   ├── A_image_2.png
  │   └── ...
  ├── B_class02/
  │   └── B_image_1.png
```

You can pass these directories via `--train_dir` and `--test_dir`.

## 🧠 Training 

Please see README_training.md

## 🧠 Testing

Please see README_testing.md

## Explanation of specific .py files

### 🧩 `smp_create_sub_images.py` — Tiling & Patch Extraction

This script processes full images and creates smaller sub-images (tiles) to facilitate localized training and reduce background noise. Sub-images are stored using a naming convention that allows reassembly and voting later during evaluation.

Key functionality:
- Loads grayscale microscopy images from a source directory
- Splits each image into non-overlapping patches (tiles)
- Saves the resulting sub-images in a tile-structured format: `originalName_row_col.png`

**Command-line Arguments:**
- `--image_dir` (required): Path to the folder containing full-sized input images.
- `--destination_dir`: Directory where the generated tiles will be saved. If not specified, saves to the current working directory.
- `--rows`: Number of rows to divide each image into (default: 2).
- `--columns`: Number of columns to divide each image into (default: 2).
- `--rotate`: If specified, rotates each image 90 degrees clockwise before tiling (useful for standardizing orientation).

This script is typically run once prior to training or testing to generate the required tile-based dataset.


### 🧱 `smp_reassemblilng_tiles.py` — Reconstructing Full Images from Tiles

This script reassembles previously tiled sub-images into a full microscopy image, with optional visual enhancements. It can apply color tinting to patches based on prediction accuracy (e.g., blue for correct, red for incorrect) or use blended tints weighted by class probabilities. Optional borders can be added around patches for clearer visualization.

Key functionality:
- Loads a tiled image layout (e.g., 6×7 grid) from disk
- Uses model predictions from a JSON file to color each tile
- Supports two coloring modes:
  - **Hard tinting**: red = incorrect, blue = correct
  - **Probability blending**: red and blue blended based on class probabilities
- Assembles all patches into a single output image
- Saves or displays the final image for analysis or presentation

**Function Parameters:**
- `patches_dir`: Path to the folder containing the tiled sub-images (e.g., tiles from `smp_create_sub_images.py`).
- `full_image_name`: The base filename used to identify which tile set to reconstruct (e.g., `sample123` for tiles like `sample123_01.jpg`).
- `rows`: Number of rows in the original tiling grid. Default is 6.
- `cols`: Number of columns in the original tiling grid. Default is 7.
- `save_path`: If specified, saves the reassembled image to this path.
- `border_thickness`: Width (in pixels) of the optional red border to draw around each tile. Default is 0 (disabled).
- `border_color`: RGB tuple defining the color of the border. Default is red `(255, 0, 0)`.
- `color_tint`: Optional tint color to apply to all patches (used when no prediction is available).
- `predictions_json`: JSON object containing predictions (as exported from your testing script).
- `blend_red_blue`: If `True`, blends red and blue based on predicted probability. If `False`, applies hard blue/red based on correctness.

This script is helpful for visualizing model behavior on a full-slide basis and identifying spatial patterns in misclassification.

### 📈 `smp_display_training_metrics.py` — Training Curve Visualization

This script loads training metrics from JSON files and visualizes loss and accuracy curves for model diagnostics. It parses all `.json` logs in a folder, extracts epoch-wise statistics, and generates side-by-side plots for training/validation loss and accuracy.

Key functionality:
- Parses JSON logs saved during training (one per run)
- Extracts training/validation loss and accuracy by epoch
- Generates matplotlib plots to help analyze model performance over time
- Automatically detects model type from the filename for labeling

Run this script locally after training to inspect training dynamics and compare multiple runs visually.


### 📊 `smp_test_results_evaluation.py` — Grouped Evaluation Summary

This script analyzes a JSON output file from a model testing session (e.g., produced by `smp_testing_script.py`) and provides a breakdown of classification accuracy both at the **patch level** and at the **whole-image level**, grouped by the first letter of each image's filename (e.g., `'a'`, `'b'`, etc.). It is designed to help evaluate per-group model performance and compare raw patch predictions with aggregated voting results.

Key functionality:
- Loads predictions and voting results from a test output JSON file
- Aggregates **patch-level accuracy** by initial group letter (e.g., `a_`, `b_`, etc.)
- Aggregates **whole-image voting accuracy** (both strict majority and probability voting)
- Summarizes and prints the count of correct and incorrect classifications per group

**Usage Notes:**
- The script assumes that image filenames begin with a group-identifying character (`a`, `b`, `c`, etc.)
- The test results file must contain both `predictions` and `voting_evaluation` sections (as produced by the main testing script with `--voting_evaluation` enabled)

**Typical Output:**
```
📈 Patch-Level Breakdown:
Group A: Correct=42, Incorrect=8
Group B: Correct=39, Incorrect=11
...

📊 Whole-Image Breakdown (Strict Majority):
Group A: Correct=5, Incorrect=1
...

📊 Whole-Image Breakdown (Probability Voting):
Group A: Correct=6, Incorrect=0
...
```

This script is useful for debugging group-specific errors, evaluating voting effectiveness, and preparing summary statistics for analysis or reports.


## 📊 Results

Model predictions and metadata (image name, predicted class, etc.) are stored in JSON format for further analysis. This format supports:
- Tile-level reassembly
- Visualization
- Voting-based aggregation

## 🚀 Example Usage

### Training
```bash
python smp_training_script.py   --model resnet_cbam   --train_dir ./data/train   --test_dir ./data/test   --num_classes 4   --device cuda
```

### Train with Contrastive Loss
```bash
python smp_training_script.py   --model resnet_cbam   --loss_function contrastive   --loss_type dynamic   --embedding_size 128   --contrast_margin 1.0   --train_dir ./data/train   --test_dir ./data/test   --num_classes 4   --device cuda
```

### Test with k-NN Evaluation
```bash
python smp_testing_script.py   --eval_strategy knn   --model resnet_cbam   --weights_path ./models/best_model.pth   --test_dir ./data/test   --train_dir ./data/train   --output_json results_knn.json   --num_classes 4   --k 5   --voting_evaluation
```

### Test using the fc-layer len=4
```bash
python smp_testing_script.py   --eval_strategy vanilla   --model resnet_cbam   --weights_path ./models/best_model.pth   --test_dir ./data/test   --train_dir ./data/train   --output_json results_classification.json   --num_classes 4   --voting_evaluation
```

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for full details.
