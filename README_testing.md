
# 🧪 Testing Guide

## 🧪 Testing Script Overview

The `smp_testing_script.py` evaluates trained models using either vanilla classification or k-NN retrieval over embeddings. It can test a single model via `--weights_path` or evaluate all models in a folder using `--model_folder`. Optional voting schemes (`--voting_evaluation`) aggregate predictions from tiled subimages into full-slide predictions using strict majority or average probability voting.

### 🔧 Testing Arguments

| Argument | Description |
|----------|-------------|
| `--test_dir` | **(Required)** Path to the test dataset. |
| `--train_dir` | **(Required)** Path to the training set (used for k-NN fitting). |
| `--weights_path` | Path to a single model’s weights for evaluation. |
| `--model_folder` | Evaluate all models in this folder sequentially. |
| `--output_json` | Output path to store predictions. |
| `--model` | Model type: `resnet_reg`, `resnet_cbam`, or `xception`. |
| `--eval_strategy` | `vanilla` for classifier, `knn` for embeddings. |
| `--num_classes` | Number of classes in your dataset. |
| `--batch_size` | Batch size for inference. |
| `--device` | Compute device. |
| `--apply_thresholding` | Enable intensity-based image filtering. |
| `--no_blurring` | Disable blurring during testing. |
| `--std_percentage` | Control thresholding bounds. |
| `--k` | Number of neighbors for k-NN. |
| `--voting_evaluation` | Enable whole-slide prediction via tile voting. |

## 🚀 Example Usage

### Test with k-NN Evaluation
```bash
python smp_testing_script.py   --eval_strategy knn   --model resnet_cbam   --weights_path ./models/best_model.pth   --test_dir ./data/test   --train_dir ./data/train   --output_json results_knn.json   --num_classes 4   --k 5   --voting_evaluation
```

### Test using the fc-layer len=4
```bash
python smp_testing_script.py   --eval_strategy vanilla   --model resnet_cbam   --weights_path ./models/best_model.pth   --test_dir ./data/test   --train_dir ./data/train   --output_json results_classification.json   --num_classes 4   --voting_evaluation
```
