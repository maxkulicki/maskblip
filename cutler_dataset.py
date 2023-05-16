from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms

class CutlerDataset(Dataset):
    def __init__(self, dataset_dir, n_samples=None, transform=None, img_size=(24, 24)):
        self.img_size = img_size

        img_dir = os.path.join(dataset_dir, "imgs")
        ann_dir = os.path.join(dataset_dir, "annotations")
        image_paths = [os.path.join(img_dir, path) for path in os.listdir(img_dir)]
        annotations_paths =  [os.path.join(ann_dir, path)  for path in os.listdir(ann_dir)]

        self.images = [transform(Image.open(image_path)) for image_path in image_paths]
        self.annotations = [self.preprocess_masks(annotation_path) for annotation_path in annotations_paths]
        #self.annotations = [torch.tensor(np.asarray(Image.open(annotation_path).resize((24, 24), Image.NEAREST))) for annotation_path in annotations_paths]
    def __len__(self):
        # this should return the size of the dataset
        return len(self.images)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        image = self.images[idx]
        annotation = self.annotations[idx]
        return image, annotation

    def preprocess_masks(self, annotation_path):
        masks = np.load(annotation_path)
        masks = [Image.fromarray(mask).resize(self.img_size, Image.NEAREST) for mask in masks.values()]
        convert_tensor = transforms.ToTensor()
        masks = [convert_tensor(mask) for mask in masks]
        masks = torch.stack(masks, dim=0).squeeze()
        masks = (masks > 0).int()
        return masks
