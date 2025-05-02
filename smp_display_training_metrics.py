import json
import matplotlib.pyplot as plt
import os
import re

def parse_json_metrics(json_file):
    """Parses a training metrics JSON file and extracts epoch-wise statistics and model info."""
    filename = os.path.basename(json_file)
    match = re.search(r"resnet_(.*?)_2025", filename)
    model_name = match.group(1) if match else "UnknownModel"

    with open(json_file, "r") as f:
        data = json.load(f)

    if "epochs" not in data:
        print(f"❌ Invalid JSON structure in {json_file} (missing 'epochs' key).")
        return None, model_name

    return data["epochs"]["epochs"], model_name

def plot_metrics(metrics, model_info):
    """Plots training & validation loss and accuracy curves."""
    epochs = [entry["epoch"] for entry in metrics]
    train_loss = [entry["train_loss"] for entry in metrics]
    val_loss = [entry["val_loss"] for entry in metrics]
    train_acc = [entry["train_accuracy"] for entry in metrics]
    val_acc = [entry["val_accuracy"] for entry in metrics]

    plt.figure(figsize=(12, 5))

    # 📉 Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss, label="Validation Loss", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_info} Loss")
    plt.legend()
    plt.grid(True)

    # 🎯 Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Accuracy", marker="o")
    plt.plot(epochs, val_acc, label="Validation Accuracy", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_info} Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_all_json_metrics_in_folder(folder_path):
    """Processes and plots metrics for every JSON file in the given folder."""
    if not os.path.isdir(folder_path):
        print(f"❌ The path '{folder_path}' is not a valid directory.")
        return

    json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".json")]

    if not json_files:
        print("No JSON files found in the specified folder.")
        return

    for jf in json_files:
        print(f"📄 Processing: {os.path.basename(jf)}")
        metrics, model_info = parse_json_metrics(jf)
        if metrics:
            plot_metrics(metrics, model_info)


if __name__ == '__main__':
    plot_all_json_metrics_in_folder(r"C:\Users\fletc\PycharmProjects\Thesis\seans_project\training_metrics\on_04_22")