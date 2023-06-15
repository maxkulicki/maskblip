from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt

class SegmentationDataset(Dataset):
    def __init__(self, dataset_dir, n_samples=None, transform=None, img_size=(24, 24)):
        self.img_size = img_size
        self.transform = transform
        with open(os.path.join(dataset_dir, "ImageSets", "Segmentation", "train.txt")) as file:
            imageIDs = [line.rstrip() for line in file]

        if n_samples is None:
            n_samples = len(imageIDs)

        self.image_paths = [os.path.join(dataset_dir, "JPEGImages", filename) + ".jpg" for filename in
                       imageIDs[:n_samples]]
        self.annotation_paths = [os.path.join(dataset_dir, "SegmentationClass", filename) + ".png" for
                             filename in imageIDs[:n_samples]]
        #self.images = [transform(Image.open(image_path)) for image_path in image_paths]
        #self.annotations = [self.preprocess_VOC_mask(annotation_path) for annotation_path in annotations_paths]
        #self.masked_images = [self.get_masked_image(image, annotation) for image, annotation in zip(self.images, annotations_paths)]
        #self.annotations = [torch.tensor(np.asarray(Image.open(annotation_path).resize((24, 24), Image.NEAREST))) for annotation_path in annotations_paths]
    def __len__(self):
        # this should return the size of the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        image = self.transform(Image.open(self.image_paths[idx]))
        annotation = self.preprocess_VOC_mask(self.annotation_paths[idx])
        path = self.image_paths[idx]
        #annotation = self.annotations[idx]
        #masked_image = self.masked_images[idx]
        return image, annotation, path
    def create_color_mapping(self, unique_values):
        num_values = len(unique_values)
        color_mapping = np.zeros((num_values, 3))
        # Generate evenly spaced hue values
        hues = np.linspace(0, 1, num_values)
        for i, value in enumerate(unique_values):
            hue = hues[i]
            color_mapping[i] = plt.cm.hsv(hue)[:3]  # Convert hue to RGB
        return color_mapping

    def get_masked_image(self, image, annotation):
        annotation = np.array(Image.open(annotation).resize(image.shape[1:], Image.NEAREST))
        color_mapping = self.create_color_mapping(np.unique(annotation))
        annotation = color_mapping[np.digitize(annotation, np.unique(annotation))-1]
        masked_image = image*0.35 + np.transpose(annotation, [2,0,1])/0.65
        return masked_image

    def preprocess_VOC_mask(self, annotation_path):
        #mask = np.array(Image.open(annotation_path).resize(self.img_size, Image.NEAREST))
        mask = np.array(Image.open(annotation_path))
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
            if most_frequent != 255:
                mask[row, col] = most_frequent
            else:
                mask[row, col] = 0

        # for i, u in enumerate(np.unique(mask)):
        #     mask[mask == u] = i
        return torch.tensor(mask)