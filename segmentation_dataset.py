from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np
from scipy.stats import mode

class SegmentationDataset(Dataset):
    def __init__(self, dataset_dir, n_samples=None, transform=None):
        with open(os.path.join(dataset_dir, "ImageSets", "Segmentation", "train.txt")) as file:
            imageIDs = [line.rstrip() for line in file]

        if n_samples is None:
            n_samples = len(imageIDs)

        image_paths = [os.path.join(dataset_dir, "JPEGImages", filename) + ".jpg" for filename in
                       imageIDs[:n_samples]]
        annotations_paths = [os.path.join(dataset_dir, "SegmentationClass", filename) + ".png" for
                             filename in imageIDs[:n_samples]]
        self.images = [transform(Image.open(image_path)) for image_path in image_paths]
        self.annotations = [self.preprocess_VOC_mask(annotation_path) for annotation_path in annotations_paths]
        #self.annotations = [torch.tensor(np.asarray(Image.open(annotation_path).resize((24, 24), Image.NEAREST))) for annotation_path in annotations_paths]
    def __len__(self):
        # this should return the size of the dataset
        return len(self.images)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        image = self.images[idx]
        annotation = self.annotations[idx]
        return image, annotation

    def preprocess_VOC_mask(self, annotation_path):
        mask = np.array(Image.open(annotation_path).resize((24, 24), Image.NEAREST))
        idxs = np.argwhere(mask == 255)

        # Iterate over the indices and find most frequent value in the 8 surrounding values
        for idx in idxs:
            row, col = idx
            # Define the square around the current index
            square = mask[max(0, row - 1):min(row + 2, mask.shape[0]), max(0, col - 1):min(col + 2, mask.shape[1])]
            # Flatten the square into a 1D array and remove the center value
            flattened = square.flatten()
            flattened = np.delete(flattened, flattened.size // 2)
            # Find the most frequent value in the flattened array
            most_frequent = mode(flattened)[0][0]
            # Replace the value at the current index with the most frequent value
            mask[row, col] = most_frequent

        for i, u in enumerate(np.unique(mask)):
            mask[mask == u] = i
        return torch.tensor(mask)