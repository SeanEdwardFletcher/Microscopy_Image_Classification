import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
import datetime
import json
import logging
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from smp_dataset_class import CustomImageDataset
from smp_models import resnet50_gray_logits, resnet50_cbam_gray_logits, resnet50_gray_embeddings, resnet50_cbam_gray_embeddings
import argparse
import os
import numpy as np
import random
import matplotlib.pyplot as plt


###                                            ###
###  functions used in all training instances  ###
###                                            ###


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 on a standard train/test split")

    # model
    parser.add_argument("--model", type=str, default="resnet_reg", choices=["resnet_reg", "resnet_cbam"],
                        help="Model to be trained; resnet_reg or resnet_cbam")
    parser.add_argument("--no_pretrained_weights", action="store_false", dest="load_weights",
                        help="turns off initializing the model with pretrained weights")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="path to weights, if 'None', uses weights from torch vision")

    # hyper parameters
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes in dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler (StepLR)")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                        help="optimizer function to use (adam/sgd)")
    parser.add_argument("--loss_function", type=str,
                        choices=["cross_entropy", "contrastive", "earth_movers"], default="cross_entropy",
                        help="Choose loss function: 'cross_entropy', 'contrastive', 'earth_movers'")
    parser.add_argument("--loss_type", type=str, choices=["regular", "dynamic"], default="regular",
                        help="Choose to use a penalty/weight matrix with contrastive/cross entropy loss: regular -> no_matrix; dynamic -> matrix")
    parser.add_argument("--embedding_size", type=int, default=128, help="size of the embedding")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors for k-NN, this k is NOT for k-fold cross validation")
    parser.add_argument("--contrast_margin", type=float, default=1, help="the margin used for contrastive loss")


    # data
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--apply_thresholding", action="store_true", default=False,
                        help="apply thresholding to the images")
    parser.add_argument("--std_percentage", type=float, default=0.5, help="used with and in the apply_thresholding functionality of the dataset class")
    parser.add_argument("--no_blurring", action="store_false", dest="gaussian_blurring", default=True,
                        help="Disable gaussian blurring to the images")
    parser.add_argument("--plot_sne", action="store_true", default=False, help="plot t-SNE graph")



    # other
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--use_cross_validation", action="store_true", default=False, help="train with cross validation")


    # disable saving
    parser.add_argument("--no_save_models", action="store_false", dest="save_models", default=True,
                        help="Disable model saving")
    parser.add_argument("--no_log_output", action="store_false", dest="log_output", default=True,
                        help="Disable logging to a .log file")

    return parser.parse_args()


def setup_logging(args, log_filename):
    """Configures the logging module with both file and console output."""

    # Configure logging to file if enabled
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",  # Overwrite previous log file
    )

    # Ensure logs also appear in the console without duplicate handlers
    logger = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(console_handler)

    logging.info("=" * 60)
    logging.info(f"📂 Logging {'enabled' if args.log_output else 'disabled'}")
    if args.log_output:
        logging.info(f"📄 Log file: {log_filename}")
    logging.info("=" * 60)


# Function to get timestamped filename
def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Function to save training metrics
def save_metrics(metrics, args, filename):
    output = {
        "args": vars(args),   # convert argparse.Namespace to dict
        "epochs": metrics     # store the epoch-wise training/validation metrics
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=4)


# Function to save hyperparameters to a text file
def save_hyperparameters(args, config):
    param_filename = config['param_file']
    with open(param_filename, "w") as f:
        f.write("🔹 Hyperparameters Used in Training:\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    tqdm.write(f"📄 Hyperparameters saved in {param_filename}")


def save_model(model, config, args, epoch, fold=None):
    """Saves the model to disk at the given epoch."""
    if args.save_models:
        arg_string = config["arg_string_for_logging"]
        fold_suffix = f"_fold{fold}" if fold is not None else ""
        model_path = os.path.join(
            config["model_dir"],
            f"{arg_string}{fold_suffix}_epoch{epoch + 1:03d}.pth"
        )
        torch.save(model.state_dict(), model_path)
        logging.info(f"💾 Model saved: {model_path}")


def create_config(args):
    """Creates a configuration dictionary containing run-specific paths and settings."""
    timestamp = get_timestamp()

    if args.loss_function == "contrastive":
        args_string = f"_{args.model}_{args.loss_function}_{args.loss_type}_lr{args.learning_rate}_margin{args.contrast_margin}_{timestamp}"
    else:
        args_string = f"_{args.model}_{args.loss_function}_{args.loss_type}_{timestamp}"

    run_dir = os.path.join("saved_runs", f"run_{args_string}")

    config = {
        "timestamp": timestamp,
        "arg_string_for_logging": args_string,
        "run_dir": run_dir,
        "log_dir": os.path.join(run_dir, "logs"),
        "model_dir": os.path.join(run_dir, "models"),
        "log_file": os.path.join(run_dir, f"training_log_{args_string}.log"),
        "param_file": os.path.join(run_dir, f"parameters_{args_string}.txt"),
        "metrics_file": os.path.join(run_dir, f"training_metrics_{args_string}.json"),
    }

    # Ensure directories exist
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config["model_dir"], exist_ok=True)

    return config


# Load datasets and initialize dataloaders
def get_dataloaders(args, transform):
    # load datasets and dataloaders that return image-label samples
    train_dataset_singles = CustomImageDataset(data_dir=args.train_dir,
                                               no_of_classes=args.num_classes,
                                               transform=transform,
                                               apply_threshold=args.apply_thresholding,
                                               loss_function="cross_entropy",
                                               apply_blurring=args.gaussian_blurring,
                                               std_percent=args.std_percentage)

    validation_dataset_singles = CustomImageDataset(data_dir=args.test_dir,
                                                    no_of_classes=args.num_classes,
                                                    transform=transform,
                                                    apply_threshold=args.apply_thresholding,
                                                    loss_function="cross_entropy",
                                                    apply_blurring=args.gaussian_blurring,
                                                    std_percent=args.std_percentage)

    train_loader_singles = DataLoader(train_dataset_singles, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader_singles = DataLoader(validation_dataset_singles, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # initialize variables for return
    train_dataset_pairs = None
    validation_dataset_pairs = None
    train_loader_pairs = None
    val_loader_pairs = None

    # load datasets and dataloaders that return pairs of images, pair labels, and individual image labels
    if args.loss_function == "contrastive":
        train_dataset_pairs = CustomImageDataset(data_dir=args.train_dir,
                                                 no_of_classes=args.num_classes,
                                                 transform=transform,
                                                 apply_threshold=args.apply_thresholding,
                                                 loss_function="contrastive",
                                                 apply_blurring=args.gaussian_blurring)

        validation_dataset_pairs = CustomImageDataset(data_dir=args.test_dir,
                                                      no_of_classes=args.num_classes,
                                                      transform=transform,
                                                      apply_threshold=args.apply_thresholding,
                                                      loss_function="contrastive",
                                                      apply_blurring=args.gaussian_blurring)
        train_loader_pairs = DataLoader(train_dataset_pairs, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader_pairs = DataLoader(validation_dataset_pairs, batch_size=args.batch_size, shuffle=False,num_workers=0)

    return train_loader_singles, val_loader_singles, train_loader_pairs, val_loader_pairs


def get_model(args):
    # Initialize model
    if args.loss_function == "cross_entropy" or args.loss_function == "earth_movers":
        if args.model == "resnet_reg":
            model = resnet50_gray_logits(num_classes=args.num_classes, embedding_dim=args.embedding_size)
        elif args.model == "resnet_cbam":
            model = resnet50_cbam_gray_logits(num_classes=args.num_classes, embedding_dim=args.embedding_size)
        else:
            print("ERROR invalid input for argument: 'models'")
            exit(1)
    else:  # args.loss_function == "contrastive
        if args.model == "resnet_reg":
            model = resnet50_gray_embeddings(num_classes=args.num_classes, embedding_dim=args.embedding_size)
        elif args.model == "resnet_cbam":
            model = resnet50_cbam_gray_embeddings(num_classes=args.num_classes, embedding_dim=args.embedding_size)
        else:
            print("ERROR invalid input for argument: 'models'")
            exit(1)
    return model


def load_pretrained_weights(model, weight_path=None, use_torchvision=False):
    if use_torchvision:
        # Load torchvision's pretrained ResNet-50 weights
        pretrained_dict = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).state_dict()
        model_dict = model.state_dict()

        # Convert RGB conv1 weights to grayscale (64, 3, 7, 7) -> (64, 1, 7, 7)
        if model.conv1.weight.shape[1] == 1:
            pretrained_conv1 = pretrained_dict["conv1.weight"]
            grayscale_conv1 = torch.mean(pretrained_conv1, dim=1, keepdim=True)
            pretrained_dict["conv1.weight"] = grayscale_conv1

        # Remove fc layer from pretrained_dict (because of shape mismatch)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "fc" not in k}

        # Filter and load matching layers
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)  # strict=False allows missing fc

        print("Loaded torchvision pretrained weights with grayscale adaptation, ignoring fc layer.")

    elif weight_path:
        # Load custom weights
        checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))

        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

        model.load_state_dict(checkpoint, strict=False)

        print(f"Loaded custom pretrained weights from {weight_path}.")

    else:
        print("No weights loaded. Model will use random initialization.")

    return model


def evaluate_knn(model, ref_loader, eval_loader, device, epoch, args):
    """
    Evaluates a contrastive model using k-NN classification.

    Parameters:
    - model: The trained embedding model.
    - ref_loader: DataLoader for reference/training embeddings.
    - eval_loader: DataLoader for evaluation/validation embeddings.
    - device: "cuda" or "cpu".
    - k: Number of neighbors to consider.

    Returns:
    - k-NN classification accuracy.
    """
    model.eval()
    ref_embeddings, ref_labels = [], []
    eval_embeddings, eval_labels = [], []

    with torch.no_grad():
        # Generate embeddings for the reference (training) set
        for images, targets in ref_loader:
            images = images.to(device)
            outputs = model(images)
            ref_embeddings.append(outputs.cpu().numpy())
            ref_labels.append(targets.numpy())

        # Generate embeddings for the evaluation (validation) set
        for images, targets in eval_loader:
            images = images.to(device)
            outputs = model(images)
            eval_embeddings.append(outputs.cpu().numpy())
            eval_labels.append(targets.numpy())

    # Convert embeddings and labels to numpy arrays
    ref_embeddings = np.vstack(ref_embeddings)
    ref_labels = np.concatenate(ref_labels)
    eval_embeddings = np.vstack(eval_embeddings)
    eval_labels = np.concatenate(eval_labels)

    # Fit k-NN on training set and predict on validation set
    knn = KNeighborsClassifier(n_neighbors=args.k, metric="euclidean")
    knn.fit(ref_embeddings, ref_labels)

    predicted_labels = knn.predict(eval_embeddings)
    accuracy = np.mean(predicted_labels == eval_labels)
    # print(f"\n📊 k-NN Accuracy (k={k}): {accuracy * 100:.2f}%")

    # Optional: Visualize embeddings using t-SNE
    if args.plot_sne and epoch % 10 == 0:
        visualize_tsne(
            embeddings=ref_embeddings,
            labels=ref_labels,
            args=args,
            epoch=epoch
        )

    return accuracy


###                                                                          ###
###  functions for training using cross entropy loss or earth movers loss    ###
###                                                                          ###


class EarthMoverDistanceLoss(nn.Module):
    def __init__(self):
        super(EarthMoverDistanceLoss, self).__init__()

    def forward(self, predicted, target):
        """
        Computes Earth Mover's Distance (EMD) loss.

        Parameters:
        - predicted: Predicted probability distribution (batch_size, num_classes)
        - target: Ground truth probability distribution (batch_size, num_classes)

        Returns:
        - emd_loss: Average EMD loss across the batch.
        """
        # Apply softmax to predicted logits to get probability distributions
        predicted = F.softmax(predicted, dim=1)

        # Convert target to one-hot if it's not already
        if len(target.shape) == 1:  # If target is class labels, convert to one-hot
            target = F.one_hot(target, num_classes=predicted.size(1)).float()

        # Cumulative distribution for predicted and target
        predicted_cdf = torch.cumsum(predicted, dim=1)
        target_cdf = torch.cumsum(target, dim=1)

        # Compute EMD as L1 distance between cumulative distributions
        emd_loss = torch.mean(torch.abs(predicted_cdf - target_cdf))

        return emd_loss


def create_dynamic_penalty_matrix(num_classes, base_penalty=1.0, max_penalty=3.0):
    """
    Creates a dynamically scaled penalty matrix based on class distance.

    Parameters:
    - num_classes (int): Number of classes.
    - base_penalty (float): Penalty for correct classifications (default: 1.0).
    - max_penalty (float): Penalty for the farthest misclassification (default: 3.0).

    Returns:
    - penalty_matrix (torch.Tensor): Dynamically computed penalty matrix.
    """
    penalty_matrix = np.zeros((num_classes, num_classes))

    # Set diagonal entries (correct classifications) to base_penalty
    np.fill_diagonal(penalty_matrix, base_penalty)

    # Calculate step increment to move from base_penalty to max_penalty
    max_distance = num_classes - 1
    step_increment = (max_penalty - base_penalty) / max_distance if max_distance > 0 else 0

    # Assign penalties dynamically based on class distance
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                class_distance = abs(i - j)
                penalty_matrix[i, j] = base_penalty + step_increment * class_distance

    # Convert penalty_matrix to a torch tensor
    return torch.tensor(penalty_matrix, dtype=torch.float32)


def weighted_cross_entropy(logits, labels, penalty_matrix):
    # Apply softmax to logits to get predicted probabilities
    probs = F.softmax(logits, dim=1)

    # Create one-hot encoded target for true labels
    target_one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()

    # Standard cross-entropy loss per sample
    ce_loss = -torch.sum(target_one_hot * torch.log(probs + 1e-9), dim=1)

    # Get predicted and true class indices
    pred_classes = torch.argmax(probs, dim=1)
    true_classes = labels

    # Apply penalty based on mistake type
    penalties = penalty_matrix[true_classes, pred_classes]

    # Multiply base CE loss by penalty
    weighted_loss = ce_loss * penalties

    # Return the mean weighted loss across the batch
    return weighted_loss.mean()


def train_classification_step(model, batch, optimizer, device, loss_fn, penalty_matrix=None):
    """Performs a single training step for classification with optional penalty matrix."""
    images, labels = batch
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)

    # Apply appropriate loss
    if penalty_matrix is not None:
        penalty_matrix = penalty_matrix.to(device)
        loss = loss_fn(outputs, labels, penalty_matrix)
    else:
        loss = loss_fn(outputs, labels)

    loss.backward()

    # # Apply gradient clipping to prevent exploding gradients ???
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # Compute accuracy for monitoring
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()

    return loss.item(), correct, labels.size(0)


def train_classification_model(train_params):
    """Trains a model using classification loss and computes accuracy."""
    model, train_loader_singles, val_loader_singles, train_loader_pairs, val_loader_pairs, device, optimizer, scheduler, config, args = (
        train_params["model"],
        train_params["train_loader_singles"],
        train_params["validation_loader_singles"],
        train_params["train_loader_pairs"],
        train_params["validation_loader_pairs"],
        train_params["device"],
        train_params["optimizer"],
        train_params["scheduler"],
        train_params["config"],
        train_params["args"],
    )

    best_val_loss, best_val_acc, epochs_since_improvement, metrics = 100, 0, 0, {"epochs": []}
    epoch_bar = tqdm(range(args.num_epochs), desc="Training Progress")

    if args.loss_function == "earth_movers":
        loss_fn = EarthMoverDistanceLoss()
    else:  # args.loss_function == "cross_entropy"
        if args.loss_type == "dynamic":
            loss_fn = weighted_cross_entropy
        else:  # args.loss_type == "regular"
            loss_fn = nn.CrossEntropyLoss()

    penalty_matrix = None
    if args.loss_type == "dynamic":
        penalty_matrix = create_dynamic_penalty_matrix(num_classes=args.num_classes)

    for epoch in epoch_bar:
        model.train()
        running_loss, correct_train, total_train, num_batches = 0.0, 0, 0, 0

        batch_bar = tqdm(train_loader_singles, desc=f"Epoch {epoch + 1}/{args.num_epochs} [Classification Training]",
                         leave=False)

        for batch in batch_bar:
            loss, correct, total = train_classification_step(
                model, batch, optimizer, device, loss_fn, penalty_matrix
            )
            running_loss += loss
            correct_train += correct
            total_train += total
            num_batches += 1

            batch_bar.set_postfix(loss=f"{running_loss / num_batches:.4f}")

        if scheduler:
            scheduler.step()

        train_loss = running_loss / num_batches
        train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0.0
        logging.info(f"✅ Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        # Validation Step (pass penalty_matrix to ensure consistency)
        val_loss, val_accuracy = validate_classification_model(
            model, val_loader_singles, device, loss_fn, penalty_matrix
        )
        logging.info(f"📊 Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        metrics["epochs"].append(
            {"epoch": epoch + 1,
             "train_loss": train_loss,
             "train_accuracy": train_accuracy,
             "val_loss": val_loss,
             "val_accuracy": val_accuracy}
        )

        # Save best model and implement early stopping

        #for debugging
        logging.debug(
            f"[EarlyStopping] Epoch {epoch + 1} | Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f} | No Improvement For: {epochs_since_improvement} epochs")

        if val_loss < best_val_loss:
            logging.info(f"📉 Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            save_model(model, config, args, epoch)
            best_val_loss = val_loss
            epochs_since_improvement = 0
        elif val_accuracy > best_val_acc:
            # validation accuracy does not currently reset "epochs since improvement"
            logging.info(f"📉 Validation accuracy improved from {best_val_acc:.4f} to {val_accuracy:.4f}")
            save_model(model, config, args, epoch)
            best_val_acc = val_accuracy
            epochs_since_improvement = 0
        else:
            logging.info(f"⏳ No improvement (val_loss: {val_loss:.4f} ≥ best: {best_val_loss:.4f})")
            epochs_since_improvement += 1

        if epochs_since_improvement >= args.patience:
            logging.info(f"⏹️ Early stopping triggered after {args.patience} epochs.")
            break

    # Save metrics
    save_metrics(metrics, args, config["metrics_file"])
    logging.info(f"📊 Metrics saved: {config['metrics_file']}")
    logging.info("🎉 Training Completed!")


def validate_classification_model(model, test_loader, device, the_foss_function, penalty_matrix=None):
    """Computes validation loss and accuracy for classification training."""
    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    loss_fn = the_foss_function

    with (torch.no_grad()):
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if penalty_matrix is not None:
                penalty_matrix = penalty_matrix.to(device)
                loss = loss_fn(outputs, labels, penalty_matrix)
            else:
                loss = loss_fn(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_loss = val_loss / len(test_loader) if len(test_loader) > 0 else float("inf")
    val_accuracy = 100 * correct_val / total_val if total_val > 0 else 0.0
    return val_loss, val_accuracy


###                                                                  ###
###  Functions for training models using contrastive loss            ###
###                                                                  ###


def visualize_tsne(embeddings, labels, args, perplexity=15, learning_rate=75, n_iter=500, epoch=None):
    """
    Plots a t-SNE visualization of the provided embeddings and labels with a legend.

    Parameters:
    - embeddings (np.ndarray or list): The high-dimensional embeddings (shape: [num_samples, embed_dim]).
    - labels (np.ndarray or list): Corresponding class labels (shape: [num_samples]).
    - save_path (str, optional): Path to save the generated plot. If None, the plot is shown instead.
    - title (str): Title for the plot.
    - perplexity (int): t-SNE perplexity parameter.
    - learning_rate (int): Learning rate for t-SNE optimization.
    - n_iter (int): Number of iterations for t-SNE optimization.
    """
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    title = f"t-SNE Training Embeddings {args.model} (Epoch {epoch})"

    print(f"🔍 Running t-SNE on {len(embeddings)} embeddings...")

    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))

    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = labels == label
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f"Class {label}", s=10, alpha=0.7)

    plt.legend(title="Classes", loc="best")
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()

    save_dir = "sne_plots"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"tSNE_{args.model}_epoch{epoch}.png")
    plt.savefig(save_path)

    print(f"📸 t-SNE plot saved to: {save_path}")
    plt.close()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, label):
        distances = F.pairwise_distance(x1, x2, p=2)
        loss_positive = (1 - label) * torch.pow(distances, 2)
        loss_negative = label * torch.pow(F.relu(self.margin - distances), 2)
        loss = torch.mean(loss_positive + loss_negative)
        return loss


class DynamicMarginContrastiveLoss(nn.Module):
    def __init__(self):
        super(DynamicMarginContrastiveLoss, self).__init__()

    def forward(self, x1, x2, labels, class1, class2, margin_matrix):
        # Compute Euclidean distance between embeddings
        distances = F.pairwise_distance(x1, x2, p=2)

        # Get the margin dynamically from the margin matrix
        margins = margin_matrix[class1.long(), class2.long()]

        # Ensure everything is on the same device
        labels = labels.to(x1.device)  # Move labels to the same device as x1
        margins = margins.to(x1.device)  # Move margins to the same device as x1
        distances = distances.to(x1.device)  # Just in case, ensure distances are on the same device

        # Positive pair loss (same class → minimize distance)
        loss_positive = (1 - labels) * torch.pow(distances, 2)

        # Negative pair loss (different classes → push apart with dynamic margin)
        loss_negative = labels * torch.pow(F.relu(margins - distances), 2)

        # Final contrastive loss
        loss = torch.mean(loss_positive + loss_negative)
        return loss


def create_margin_matrix(no_of_classes, base_margin=1.0, max_margin=2.0):
    """
    Creates a margin matrix for dynamic contrastive loss.

    Parameters:
    - no_of_classes (int): Number of classes.
    - base_margin (float): Minimum margin between class pairs.
    - max_margin (float): Maximum margin between the most distant class pairs.

    Returns:
    - margin_matrix (torch.Tensor): A tensor of shape (no_of_classes, no_of_classes).
    """
    # Initialize margin matrix with base margin
    margin_matrix = np.full((no_of_classes, no_of_classes), base_margin)

    # Set diagonal to zero (no margin between same class pairs)
    np.fill_diagonal(margin_matrix, 0.0)

    # Calculate step increment to move from base_margin to max_margin
    max_distance = no_of_classes - 1
    step_increment = (max_margin - base_margin) / max_distance if max_distance > 0 else 0

    # Assign margins based on class difference
    for i in range(no_of_classes):
        for j in range(no_of_classes):
            if i != j:
                # Compute margin based on class distance
                class_distance = abs(i - j)
                margin_matrix[i, j] = base_margin + step_increment * class_distance

    return torch.tensor(margin_matrix, dtype=torch.float32)


def train_contrast(model, train_loader, optimizer, device, args, margin_matrix):
    """
    Performs a single epoch of contrastive training.
    """
    model.train()
    running_loss, num_batches = 0.0, 0
    batch_bar = tqdm(train_loader, desc="[Contrastive Training]", leave=False)

    # Instantiate loss functions once before the loop
    if args.loss_type == "regular":
        loss_fn = ContrastiveLoss(margin=args.contrast_margin)
    else:  # args.loss_type == "dynamic"
        loss_fn = DynamicMarginContrastiveLoss()

    for (img1, img2), labels, class1, class2 in batch_bar:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        optimizer.zero_grad()
        output1, output2 = model(img1), model(img2)

        if args.loss_type == "regular":
            loss = loss_fn(output1, output2, labels)
        else:  # args.loss_type == "dynamic"

            loss = loss_fn(output1, output2, labels, class1, class2, margin_matrix)

        loss.backward()

        # # add gradient clipping? Are the gradients going to explode? maybe?
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()
        num_batches += 1
        batch_bar.set_postfix(loss=f"{running_loss / num_batches:.4f}")

    avg_train_loss = running_loss / num_batches
    return avg_train_loss


def train_contrastive_model(train_params):
    model, train_loader_singles, val_loader_singles, train_loader_pairs, val_loader_pairs, device, optimizer, scheduler, config, args = (
        train_params["model"],
        train_params["train_loader_singles"],
        train_params["validation_loader_singles"],
        train_params["train_loader_pairs"],
        train_params["validation_loader_pairs"],
        train_params["device"],
        train_params["optimizer"],
        train_params["scheduler"],
        train_params["config"],
        train_params["args"],
    )

    best_val_loss, epochs_since_improvement, metrics = 100, 0, {"epochs": []}
    epoch_bar = tqdm(range(args.num_epochs), desc="Training Progress")

    margin_matrix = None
    if args.loss_type == "dynamic":
        margin_matrix = create_margin_matrix(args.num_classes)

    for epoch in epoch_bar:
        avg_train_loss = train_contrast(model, train_loader_pairs, optimizer, device, args, margin_matrix)
        logging.info(f"✅ Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

        # Validation Step
        val_loss, val_accuracy = validate_contrastive_model(model=model,
                                                            train_loader_singles=train_loader_singles,
                                                            val_loader_singles=val_loader_singles,
                                                            val_loader_pairs=val_loader_pairs,
                                                            device=device,
                                                            margin_matrix=margin_matrix,
                                                            loss_type=args.loss_type,
                                                            epoch=epoch,
                                                            args=args)
        logging.info(f"📊 Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        metrics["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": 0.0,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        # Save the best model
        if val_loss < best_val_loss:
            logging.info(f"📉 Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            save_model(model, config, args, epoch)
            best_val_loss = val_loss
            epochs_since_improvement = 0
        else:
            logging.info(f"⏳ No improvement (val_loss: {val_loss:.4f} ≥ best: {best_val_loss:.4f})")
            epochs_since_improvement += 1

        if epochs_since_improvement >= args.patience:
            logging.info(f"⏹️ Early stopping triggered after {args.patience} epochs.")
            break

    # Save metrics
    save_metrics(metrics, args, config["metrics_file"])

    logging.info(f"📊 Metrics saved: {config['metrics_file']}")
    logging.info("🎉 Contrastive training completed.")


def validate_contrastive_model(model, train_loader_singles, val_loader_singles, val_loader_pairs, device, margin_matrix, loss_type, epoch, args):
    """
    Validates the contrastive model using k-NN for accuracy calculation.

    Parameters:
    - model: The trained contrastive model.
    - val_loader: DataLoader for validation dataset.
    - criterion: Contrastive loss criterion.
    - device: "cuda" or "cpu".
    - margin_matrix: Dynamic margin matrix (if applicable).
    - loss_type: Either "regular" or "dynamic".
    - k (int): Number of neighbors to consider for k-NN.

    Returns:
    - val_loss: Average validation loss.
    - val_accuracy: k-NN classification accuracy using evaluate_knn().
    """

    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    # Instantiate loss functions once before the loop
    if loss_type == "regular":
        loss_fn = ContrastiveLoss()
    else:  # loss_type == "dynamic"
        loss_fn = DynamicMarginContrastiveLoss()

    with torch.no_grad():
        for (img1, img2), labels, class1, class2 in val_loader_pairs:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            output1, output2 = model(img1), model(img2)

            if loss_type == "regular":
                loss = loss_fn(output1, output2, labels)
            elif loss_type == "dynamic":
                loss = loss_fn(output1, output2, labels, class1, class2, margin_matrix)

            val_loss += loss.item()

    # Compute average loss over the validation set
    val_loss /= len(val_loader_pairs)

    # --- Use evaluate_knn() to calculate accuracy ---
    val_accuracy = evaluate_knn(model, train_loader_singles, val_loader_singles, device, epoch, args)

    return val_loss, val_accuracy

###                                                                  ###
###  Functions for training models using k-fold cross validation     ###
###                                                                  ###


def train_with_cross_validation(dataset, model_fn, args, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    fold_metrics = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logging.info(f"\n📂 Starting Fold {fold + 1}/{k}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = model_fn().to(args.device)  # grabs a fresh model for each fold, prevents the weights from carrying over

        # Select loss function
        if args.loss_function == "earth_movers":
            loss_fn = EarthMoverDistanceLoss()
        else:
            if args.loss_type == "dynamic":
                loss_fn = weighted_cross_entropy
            else:
                loss_fn = nn.CrossEntropyLoss()

        penalty_matrix = create_dynamic_penalty_matrix(num_classes=args.num_classes) if args.loss_type == "dynamic" else None
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1) if args.use_scheduler else None

        best_val_loss = float("inf")
        best_val_acc = 0.0
        epochs_since_improvement = 0

        metrics = {"epochs": []}

        for epoch in tqdm(range(args.num_epochs), desc=f"Fold {fold+1} Training"):
            model.train()
            running_loss, correct_train, total_train = 0.0, 0, 0

            for batch in train_loader:
                loss, correct, total = train_classification_step(model, batch, optimizer, args.device, loss_fn, penalty_matrix)
                running_loss += loss * total
                correct_train += correct
                total_train += total

            if scheduler:
                scheduler.step()

            train_loss = running_loss / total_train if total_train > 0 else float("inf")
            train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0.0

            val_loss, val_accuracy = validate_classification_model(model, val_loader, args.device, loss_fn, penalty_matrix)

            metrics["epochs"].append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })

            logging.info(f"📈 Fold {fold+1}, Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_accuracy
                epochs_since_improvement = 0
                save_model(model, args.config, args, epoch, fold=fold + 1)
            elif val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                epochs_since_improvement = 0
                save_model(model, args.config, args, epoch, fold=fold + 1)
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= args.patience:
                logging.info(f"⏹️ Early stopping triggered after {args.patience} epochs.")
                break

        # Save per-fold metrics
        fold_key = f"fold_{fold+1}"
        fold_metrics[fold_key] = metrics
        save_metrics(metrics, args, f"{args.config['metrics_file'].replace('.json', f'_fold{fold+1}.json')}")
        logging.info(f"📊 Fold {fold+1} metrics saved.")

    logging.info("✅ Cross-validation complete.")
    return fold_metrics
