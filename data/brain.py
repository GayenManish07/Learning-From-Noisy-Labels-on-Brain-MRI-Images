import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from .utils import noisify  # Assuming noisify is still needed

class CustomDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 noise_type=None, noise_rate=0.2, random_state=0, image_size=(256, 256)):
        """
        Args:
            root (str): Root directory of the dataset.
            train (bool): If True, loads the training set; otherwise, loads the test set.
            transform (callable): A function/transform to apply to the images (after resizing and converting to tensor).
            target_transform (callable): A function/transform to apply to the labels.
            noise_type (str): Type of noise to inject into labels (e.g., 'symmetric', 'pairflip').
            noise_rate (float): Proportion of labels to corrupt with noise.
            random_state (int): Random seed for noise injection.
            image_size (tuple): Desired size (height, width) for resizing images.
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.noise_type = noise_type
        self.dataset = 'custom'  # Name of the dataset for noisify function
        self.image_size = image_size  # Desired image size (height, width)

        # Set the data directory based on train/test
        self.data_dir = os.path.join(root, 'train' if train else 'test')

        # Load images and labels
        self.image_paths = []
        self.labels = []

        # Iterate through class folders
        class_folders = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(class_folders)}  # Map class names to indices
        self.num_classes = len(class_folders)  # Number of classes in the dataset

        for cls in class_folders:
            cls_dir = os.path.join(self.data_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_idx[cls])

        # Convert labels to numpy array for noisify function
        self.labels = np.array(self.labels)

        # Inject noise into labels if required
        if noise_type != 'clean':
            self.noisy_labels, self.actual_noise_rate = noisify(
                dataset=self.dataset,
                train_labels=self.labels,
                noise_type=noise_type,
                noise_rate=noise_rate,
                random_state=random_state
            )
            self.noisy_labels = self.noisy_labels.squeeze()
            self.noise_or_not = (self.noisy_labels == self.labels)
        else:
            self.noisy_labels = self.labels
            self.noise_or_not = np.ones_like(self.labels, dtype=bool)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB (or 'L' for grayscale)

        # Apply default transformations (resize and convert to tensor)
        default_transform = transforms.Compose([
            transforms.Resize(self.image_size),  # Resize to desired size
            transforms.ToTensor(),               # Convert to tensor
        ])
        image = default_transform(image)

        # Get the label (noisy or clean)
        label = self.noisy_labels[index] if self.noise_type != 'clean' else self.labels[index]

        # Clamp label to valid range
        if label < 0 or label >= self.num_classes:
            print(f"Warning: Clamping invalid label {label} at index {index} to valid range.")
            label = max(0, min(label, self.num_classes - 1))  # Clamp to [0, num_classes - 1]

        # Apply target transformation if specified
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, index
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format('train' if self.train else 'test')
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Transforms (if any): {}\n'.format(self.transform.__repr__())
        fmt_str += '    Target Transforms (if any): {}\n'.format(self.target_transform.__repr__())
        return fmt_str