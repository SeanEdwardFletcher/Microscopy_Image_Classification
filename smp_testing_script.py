from torchvision import transforms
from smp_models import resnet50_gray_logits, resnet50_cbam_gray_logits, resnet50_gray_embeddings, resnet50_cbam_gray_embeddings
# from ethans_code import ethans_xception_model
import torch
import json
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from smp_dataset_class import CustomImageDataset
import argparse
import os
import datetime
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
from copy import deepcopy


# Define transforms for testing (no data augmentation)

# # normalization for arsenic dataset
# test_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.534], std=[0.131]),
# ])

# normalization for initial taxol dataset tiled->6
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.439], std=[0.061]),
])


def parse_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--weights_path", required=False, default=None,
                        help="test a single model by providing the path to its weights")
    parser.add_argument("--model_folder", default=None,
                        help="Optional folder path containing model weights to test sequentially")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--model", choices=["resnet_reg", "resnet_cbam", "xception"], required=False)
    parser.add_argument("--eval_strategy", choices=["knn", "vanilla"], required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--apply_thresholding", action="store_true")
    parser.add_argument("--no_blurring", action="store_false", dest="gaussian_blurring", default=True,
                        help="Disable gaussian blurring to the images")
    parser.add_argument("--std_percentage", type=float, default=0.5, help="used in the apply_thresholding functionality of the dataset class")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors for k-NN")
    parser.add_argument("--voting_evaluation", action="store_true")

    return parser.parse_args()


def evaluate_strict_voting(predictions):
    image_groups = defaultdict(list)
    for entry in predictions:
        image_name = entry["image"]
        full_image_id = "_".join(image_name.split("_")[:-1])
        image_groups[full_image_id].append(entry)

    correct = 0
    total = 0
    results = []

    for full_image_id, subimage_preds in image_groups.items():
        vote_counts = Counter([p["predicted"] for p in subimage_preds])
        most_common = vote_counts.most_common()
        # If there's a tie, arbitrarily choose the first class, but mark it as incorrect
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            final_prediction = most_common[0][0]
            is_correct = False
        else:
            final_prediction = most_common[0][0]
            true_label = subimage_preds[0]["ground_truth"]
            is_correct = final_prediction == true_label
        true_label = subimage_preds[0]["ground_truth"]
        correct += is_correct
        total += 1

        results.append({
            "full_image": full_image_id,
            "predicted": final_prediction,
            "ground_truth": true_label,
            "correct": is_correct,
            "votes": dict(vote_counts)
        })

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results


def evaluate_probability_voting(predictions, num_classes):
    image_groups = defaultdict(list)
    for entry in predictions:
        image_name = entry["image"]
        full_image_id = "_".join(image_name.split("_")[:-1])
        image_groups[full_image_id].append(entry)

    correct = 0
    total = 0
    results = []

    for full_image_id, sub_preds in image_groups.items():
        avg_probs = np.zeros(num_classes)
        for p in sub_preds:
            avg_probs += np.array(p["probabilities"])
        avg_probs /= len(sub_preds)
        predicted = int(np.argmax(avg_probs))
        true_label = sub_preds[0]["ground_truth"]
        is_correct = predicted == true_label
        correct += is_correct
        total += 1
        results.append({
            "full_image": full_image_id,
            "predicted": predicted,
            "ground_truth": true_label,
            "correct": is_correct,
            "avg_probabilities": [round(x, 4) for x in avg_probs.tolist()]
        })

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results


def evaluate_classification_and_save(model, test_loader, device, save_path, args):
    model.eval()
    correct = 0
    total = 0
    results = []

    with torch.no_grad():
        for images, labels, filenames in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            for i in range(images.size(0)):
                pred_class = int(predicted[i].cpu().item())
                results.append({
                    "image": filenames[i],
                    "ground_truth": int(labels[i].cpu().item()),
                    "predicted": pred_class,
                    "confidence": round(float(probs[i][pred_class].cpu().item()), 4),
                    "logits": [round(x, 4) for x in outputs[i].detach().cpu().tolist()],
                    "probabilities": [round(x, 4) for x in probs[i].detach().cpu().tolist()]
                })

    accuracy = correct / total if total > 0 else 0.0

    output_data = {
        "summary": {
            "total_samples": total,
            "accuracy": accuracy,
            "model_type": args.model,
            "eval_strategy": args.eval_strategy,
            "weights_path": args.weights_path
        },
        "args": vars(args),
        "predictions": results
    }

    with open(save_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"🔢 Total samples evaluated: {total}")
    return accuracy, results


def evaluate_knn_and_save(model, train_loader, test_loader, device, save_path, args, k=3):
    model.eval()
    train_embeddings, train_labels = [], []
    test_embeddings, test_labels, filenames = [], [], []

    print("🔍 Extracting embeddings from the training set...")
    with torch.no_grad():
        for images, targets, _ in tqdm(train_loader, desc="Extracting Training Embeddings"):
            images = images.to(device)
            embeddings = model(images).cpu().numpy()
            train_embeddings.extend(embeddings)
            train_labels.extend(targets.numpy())

    train_embeddings = np.array(train_embeddings)
    train_labels = np.array(train_labels)

    print("🔍 Extracting embeddings from the test set...")
    with torch.no_grad():
        for images, targets, batch_filenames in tqdm(test_loader, desc="Extracting Test Embeddings"):
            images = images.to(device)
            embeddings = model(images).cpu().numpy()
            test_embeddings.extend(embeddings)
            test_labels.extend(targets.numpy())
            filenames.extend(batch_filenames)

    test_embeddings = np.array(test_embeddings)
    test_labels = np.array(test_labels)

    print(f"🏷️ Fitting k-NN with k={k} on the training set...")
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(train_embeddings, train_labels)

    predictions = knn.predict(test_embeddings)
    probabilities = knn.predict_proba(test_embeddings)
    correct = (predictions == test_labels).sum()
    accuracy = correct / len(test_labels)
    print(f"✅ k-NN Test Accuracy: {accuracy * 100:.2f}%")

    prediction_log = [
        {
            "image": filenames[i],
            "ground_truth": int(test_labels[i]),
            "predicted": int(predictions[i]),
            "confidence": round(float(probabilities[i][predictions[i]]), 4),
            "probabilities": [round(x, 4) for x in probabilities[i].tolist()]
        }
        for i in range(len(predictions))
    ]

    output_data = {
        "summary": {
            "total_samples": len(prediction_log),
            "accuracy": accuracy,
            "model_type": args.model,
            "eval_strategy": args.eval_strategy,
            "weights_path": args.weights_path,
            "k_value": args.k
        },
        "args": vars(args),
        "predictions": prediction_log
    }

    with open(save_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"📁 Predictions saved to: {save_path}")
    print(f"🔢 Total samples evaluated: {len(prediction_log)}")
    return accuracy, prediction_log


def test_model(args):

    print(f"🗳️ Voting evaluation enabled: {args.voting_evaluation}")
    test_dataset = CustomImageDataset(
        data_dir=args.test_dir,
        transform=test_transform,
        apply_threshold=args.apply_thresholding,
        std_percent=args.std_percentage,
        apply_blurring=args.gaussian_blurring,
        no_of_classes=args.num_classes,
        testing=True
    )

    train_dataset = CustomImageDataset(
        data_dir=args.train_dir,
        transform=test_transform,
        apply_threshold=args.apply_thresholding,
        std_percent=args.std_percentage,
        apply_blurring=args.gaussian_blurring,
        no_of_classes=args.num_classes,
        testing=True
    )

    print(f"✅ Training directory: {args.train_dir}")
    print(f"✅ Testing directory: {args.test_dir}")
    print(f"📊 Train dataset size: {len(train_dataset)}")
    print(f"📊 Test dataset size: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if not args.weights_path:
        raise ValueError("You must specify a --weights_path to load a model for testing.")

    if not args.model:
        detected_model = detect_model_type_from_filename(args.weights_path)
        if detected_model:
            args.model = detected_model
            print(f"🧠 Auto-detected model type: {args.model}")
        else:
            raise ValueError("❌ Could not infer model type from weights path.")

    if args.model == "resnet_reg":
        model = resnet50_gray_logits(num_classes=args.num_classes) if args.eval_strategy == "vanilla" else resnet50_gray_embeddings(num_classes=args.num_classes)
    elif args.model == "resnet_cbam":
        model = resnet50_cbam_gray_logits(num_classes=args.num_classes) if args.eval_strategy == "vanilla" else resnet50_cbam_gray_embeddings(num_classes=args.num_classes)
    elif args.model == "xception":
        # model = ethans_xception_model
        pass
    else:
        raise ValueError("Invalid model type.")

    try:
        model.load_state_dict(torch.load(args.weights_path, map_location=args.device))
    except RuntimeError as e:
        raise ValueError(f"Failed to load weights. Check if the architecture matches. Error: {e}")

    model.to(args.device)

    if not args.output_json:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json = os.path.join(os.getcwd(), f"test_results_{args.eval_strategy}_{timestamp}.json")
    else:
        output_json = args.output_json

    if args.eval_strategy == "knn":
        accuracy, results = evaluate_knn_and_save(model=model, test_loader=test_loader, train_loader=train_loader, device=args.device, save_path=output_json, args=args, k=args.k)
    else:
        accuracy, results = evaluate_classification_and_save(model, test_loader, args.device, output_json, args)

    if args.voting_evaluation:
        strict_acc, strict_results = evaluate_strict_voting(results)
        prob_acc, prob_results = evaluate_probability_voting(results, args.num_classes)

        with open(output_json, "r") as f:
            full_output = json.load(f)

        full_output["voting_evaluation"] = {
            "strict_majority": {
                "accuracy": strict_acc,
                "results": strict_results
            },
            "probability_voting": {
                "accuracy": prob_acc,
                "results": prob_results
            }
        }

        with open(output_json, "w") as f:
            json.dump(full_output, f, indent=4)

    print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
    print(f"📁 Results saved to: {output_json}")


def evaluate_models_in_folder(args):
    """
    Evaluates every model checkpoint in the specified folder using the provided args.
    Automatically updates weights_path, output_json, and model for each model file.
    """
    model_files = [f for f in os.listdir(args.model_folder) if f.endswith(".pt") or f.endswith(".pth")]

    if not model_files:
        print(f"⚠️ No .pt or .pth model files found in: {args.model_folder}")
        return

    for fname in model_files:
        print("\n" + "=" * 80)
        print(f"🔍 Evaluating model: {fname}")

        local_args = deepcopy(args)
        local_args.weights_path = os.path.join(local_args.model_folder, fname)

        # Auto-generate a unique output JSON filename
        base_name = os.path.splitext(fname)[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        local_args.output_json = os.path.join(os.getcwd(), f"results_{base_name}_{local_args.eval_strategy}_{timestamp}.json")

        local_args.model = detect_model_type_from_filename(fname)
        if local_args.model is None:
            print(f"⚠️ Could not detect model type from: {fname}, skipping.")
            continue

        test_model(local_args)


def detect_model_type_from_filename(filename):
    """
    Infers the model type from the model filename.

    Returns:
        - "resnet_cbam" if "cbam" is found in the filename
        - "resnet_reg" otherwise
    """
    fname = filename.lower()
    if "cbam" in fname:
        return "resnet_cbam"
    else:
        return "resnet_reg"


if __name__ == '__main__':
    args = parse_test_args()

    if not args.weights_path and not args.model_folder:
        print("❌ You must provide either --weights_path or --model_folder")
        exit(1)

    if args.model_folder:
        evaluate_models_in_folder(args)
    else:
        test_model(args)


r"""
    python smp_testing_script.py \
  --test_dir "/home/xz/seans_project_spring2025/data/split_tiled_2by3_taxol_ds_110/test" \
  --train_dir "/home/xz/seans_project_spring2025/data/split_tiled_2by3_taxol_ds_110/train" \
  --weights_path "/home/xz/seans_project_spring2025/models/threshold_models/_resnet_reg_cross_entropy_regular_2025-04-14_14-33-24_epoch016.pth" \
  --model "resnet_reg" \
  --apply_thresholding \
  --std_percentage .7 \
  --eval_strategy knn \
  --num_classes 4 \
  --k 5 \
  --voting_evaluation

"""


r"""
for ethan:
    python smp_testing_script.py \
  --test_dir "path_to_test_set_directory" \
  --train_dir "path_to_training_set_directory" \
  --weights_path "path_to_trained_model's_weights" \
  --model "xception" \
  --eval_strategy knn \
  --num_classes 4 \
  --k 5 \
  --voting_evaluation
"""
