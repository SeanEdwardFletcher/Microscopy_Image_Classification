import os
import random
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from smp_dataset_class_functions import inverse_bandpass_thresholding


class CustomImageDataset(Dataset):
    def __init__(self, data_dir, no_of_classes, std_percent=0.5, transform=None, apply_threshold=False, apply_blurring=True, loss_function="cross_entropy", testing=False):
        """
        Custom dataset that ensures all images are grayscale.

        Parameters:
        - data_dir (str): Path to the dataset containing class folders.
        - transform (torchvision.transforms): Transformations to apply.
        - apply_threshold (bool): Whether to apply thresholding to images.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.apply_threshold = apply_threshold
        self.apply_blurring = apply_blurring
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.loss_function = loss_function
        self.no_of_classes = no_of_classes
        self._load_data()
        self.std_percent = std_percent
        self.testing = testing

    def _load_data(self):
        """Loads image paths and class labels from the dataset directory."""
        class_folders = sorted(os.listdir(self.data_dir))  # Get class names

        # Assign a unique index to each class
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_folders)}

        for class_name in class_folders:
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                if os.path.isfile(img_path) and img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

    def _apply_gaussian_blur(self, the_image):
        image_np = np.array(the_image)  # Convert PIL Image to NumPy array
        blur_image =  cv2.GaussianBlur(image_np, (3, 3), 0)
        return Image.fromarray(blur_image)  # Convert back to PIL Image

    def _apply_thresholding(self, image):
        """Applies inverse bandpass thresholding to the image."""
        image_np = np.array(image)  # Convert PIL Image to NumPy array
        thresholded_image = inverse_bandpass_thresholding(image_np, st_d=self.std_percent)
        return Image.fromarray(thresholded_image)  # Convert back to PIL Image

    def __len__(self):
        """Returns total number of images in dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label by index.
        Dynamically returns a positive or negative pair if loss_function == "contrastive".
        """
        img1_path = self.image_paths[idx]
        label1 = self.labels[idx]
        filename1 = os.path.basename(img1_path)

        img1 = Image.open(img1_path).convert("L")

        # Apply thresholding or blurring if enabled
        if self.apply_threshold:
            img1 = self._apply_thresholding(img1)
        elif self.apply_blurring and not self.apply_threshold:
            img1 = self._apply_gaussian_blur(img1)

        # Apply transformations
        if self.transform:
            img1 = self.transform(img1)

        if self.loss_function == "contrastive":
            # Create positive or negative pair dynamically
            if random.random() < 0.5:  # Positive pair
                pos_indices = [i for i, l in enumerate(self.labels) if l == label1 and i != idx]
                pair_idx = random.choice(pos_indices) if pos_indices else idx
            else:  # Negative pair
                neg_indices = [i for i, l in enumerate(self.labels) if l != label1]
                pair_idx = random.choice(neg_indices) if neg_indices else idx

            img2_path = self.image_paths[pair_idx]
            label2 = self.labels[pair_idx]

            img2 = Image.open(img2_path).convert("L")
            if self.apply_threshold:
                img2 = self._apply_thresholding(img2)
            elif self.apply_blurring and not self.apply_threshold:
                img2 = self._apply_gaussian_blur(img2)

            if self.transform:
                img2 = self.transform(img2)

            # Return a positive or negative pair with corresponding labels
            pair_label = 1 if label1 == label2 else 0
            return (img1, img2), pair_label, label1, label2

        else:
            if self.testing:
                return img1, label1, filename1
            else:
                return img1, label1
