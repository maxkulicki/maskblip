from torch.utils.data import Dataset
import torch
import torchvision
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from torchvision import transforms
import os
from PIL import Image
import matplotlib._color_data as mcd
import cv2
import json
import numpy as np
import os
import pickle as pkl
from torchvision.datasets import CocoDetection, Cityscapes, VOCSegmentation
from scipy.io import loadmat

import matplotlib.pyplot as plt
from tqdm import tqdm

_NUMERALS = '0123456789abcdefABCDEF'
_HEXDEC = {v: int(v, 16) for v in (x + y for x in _NUMERALS for y in _NUMERALS)}
LOWERCASE, UPPERCASE = 'x', 'X'

def rgb(triplet):
    return _HEXDEC[triplet[0:2]], _HEXDEC[triplet[2:4]], _HEXDEC[triplet[4:6]]


def loadAde20K(file):
    fileseg = file.replace('.jpg', '_seg.png');
    with Image.open(fileseg) as io:
        seg = np.array(io);

    # Obtain the segmentation mask, bult from the RGB channels of the _seg file
    R = seg[:, :, 0];
    G = seg[:, :, 1];
    B = seg[:, :, 2];
    ObjectClassMasks = (R / 10).astype(np.int32) * 256 + (G.astype(np.int32));

    # Obtain the instance mask from the blue channel of the _seg file
    Minstances_hat = np.unique(B, return_inverse=True)[1]
    Minstances_hat = np.reshape(Minstances_hat, B.shape)
    ObjectInstanceMasks = Minstances_hat

    level = 0
    PartsClassMasks = [];
    PartsInstanceMasks = [];
    while True:
        level = level + 1;
        file_parts = file.replace('.jpg', '_parts_{}.png'.format(level));
        if os.path.isfile(file_parts):
            with Image.open(file_parts) as io:
                partsseg = np.array(io);
            R = partsseg[:, :, 0];
            G = partsseg[:, :, 1];
            B = partsseg[:, :, 2];
            PartsClassMasks.append((np.int32(R) / 10) * 256 + np.int32(G));
            PartsInstanceMasks = PartsClassMasks
            # TODO:  correct partinstancemasks


        else:
            break

    objects = {}
    parts = {}

    attr_file_name = file.replace('.jpg', '.json')
    if os.path.isfile(attr_file_name):
        with open(attr_file_name, 'r') as f:
            input_info = json.load(f)

        contents = input_info['annotation']['object']
        instance = np.array([int(x['id']) for x in contents])
        names = [x['raw_name'] for x in contents]
        corrected_raw_name = [x['name'] for x in contents]
        partlevel = np.array([int(x['parts']['part_level']) for x in contents])
        ispart = np.array([p > 0 for p in partlevel])
        iscrop = np.array([int(x['crop']) for x in contents])
        listattributes = [x['attributes'] for x in contents]
        polygon = [x['polygon'] for x in contents]
        for p in polygon:
            p['x'] = np.array(p['x'])
            p['y'] = np.array(p['y'])

        objects['instancendx'] = instance[ispart == 0]
        objects['class'] = [names[x] for x in list(np.where(ispart == 0)[0])]
        objects['corrected_raw_name'] = [corrected_raw_name[x] for x in list(np.where(ispart == 0)[0])]
        objects['iscrop'] = iscrop[ispart == 0]
        objects['listattributes'] = [listattributes[x] for x in list(np.where(ispart == 0)[0])]
        objects['polygon'] = [polygon[x] for x in list(np.where(ispart == 0)[0])]

        parts['instancendx'] = instance[ispart == 1]
        parts['class'] = [names[x] for x in list(np.where(ispart == 1)[0])]
        parts['corrected_raw_name'] = [corrected_raw_name[x] for x in list(np.where(ispart == 1)[0])]
        parts['iscrop'] = iscrop[ispart == 1]
        parts['listattributes'] = [listattributes[x] for x in list(np.where(ispart == 1)[0])]
        parts['polygon'] = [polygon[x] for x in list(np.where(ispart == 1)[0])]

    return {'img_name': file, 'segm_name': fileseg,
            'class_mask': ObjectClassMasks, 'instance_mask': ObjectInstanceMasks,
            'partclass_mask': PartsClassMasks, 'part_instance_mask': PartsInstanceMasks,
            'objects': objects, 'parts': parts}


def plot_polygon(img_name, info, show_obj=True, show_parts=False):
    colors = mcd.CSS4_COLORS
    color_keys = list(colors.keys())
    all_objects = []
    all_poly = []
    if show_obj:
        all_objects += info['objects']['class']
        all_poly += info['objects']['polygon']
    if show_parts:
        all_objects += info['parts']['class']
        all_poly += info['objects']['polygon']

    img = cv2.imread(img_name)
    thickness = 5
    for it, (obj, poly) in enumerate(zip(all_objects, all_poly)):
        curr_color = colors[color_keys[it % len(color_keys)]]
        pts = np.concatenate([poly['x'][:, None], poly['y'][:, None]], 1)[None, :]
        color = rgb(curr_color[1:])
        img = cv2.polylines(img, pts, True, color, thickness)
    return img


class ADE20KDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        DATASET_PATH = root_dir

        index_file = 'ADE20K_2021_17_01/index_ade20k.pkl'
        with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
            index_ade20k = pkl.load(f)
        nfiles = len(index_ade20k['filename'])

        self.images = []
        self.masks = []
        self.paths = []  # Added to store the image paths

        for i in tqdm(range(nfiles)):
            index_ade20k['filename'][i] = index_ade20k['filename'][i].replace('/data2/', '../datasets/')
            full_file_name = '{}/{}'.format(index_ade20k['folder'][i], index_ade20k['filename'][i])
            info = loadAde20K('{}/{}'.format(DATASET_PATH, full_file_name))
            img = cv2.imread(info['img_name'])[:, :, ::-1]
            self.images.append(torch.from_numpy(img.copy()))
            seg = cv2.imread(info['segm_name'])[:, :, ::-1]
            self.masks.append(torch.from_numpy(seg.copy()))
            self.paths.append(info['img_name'])  # Store the image path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        path = self.paths[idx]  # Get the path for this index

        return image, mask, path

def coco_annsToMask(img, anns):
    img_size = img.size()
    img_size = (img_size[1], img_size[2])

    # Create an empty array to hold the label image
    label_img = np.zeros(img_size, dtype=np.int32)


    for ann in anns:
        # Get the binary mask for this object
        rle = mask_utils.frPyObjects(ann['segmentation'], img_size[0], img_size[1])
        m = mask_utils.decode(rle)

        # Check if m has a third dimension
        if m.ndim == 3:
            # If yes, sum along the third dimension to combine all objects into one mask
            m = np.sum(m, axis=2)

        m = m * ann['category_id']
        label_img = np.maximum(label_img, m)

    return torch.from_numpy(label_img)

class VOCWithPaths(VOCSegmentation):
    def __init__(self, root, transform=None, target_transform=None):
        super(VOCWithPaths, self).__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        img, target = super(VOCWithPaths, self).__getitem__(index)
        path = self.images[index]
        return img, target, path

class CityscapesWithPaths(Cityscapes):
    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None, target_transform=None):
        super(CityscapesWithPaths, self).__init__(root, split, mode, target_type, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        img, target = super(CityscapesWithPaths, self).__getitem__(index)
        path = self.images[index]
        return img, target, path

class CocoDetectionWithPaths(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super(CocoDetectionWithPaths, self).__init__(root, annFile, transform)

    def __getitem__(self, index):
        img, target = super(CocoDetectionWithPaths, self).__getitem__(index)
        target = coco_annsToMask(img, target)
        path = self.coco.loadImgs(self.ids[index])[0]['file_name']
        return img, target, path

# Usage
# dataset = CocoDetectionWithPaths('../datasets/coco/val2017',
#                                  '../datasets/coco/annotations/instances_val2017.json',
#                                  transform=transforms.ToTensor())

class PascalContextDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, annotation_folder):
        super().__init__()
        self.img_folder = img_folder
        self.annotation_folder = annotation_folder
        self.annotation_files = sorted(os.listdir(annotation_folder))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        annotation_file = self.annotation_files[idx]
        annotation_path = os.path.join(self.annotation_folder, annotation_file)

        # Assuming the image file has the same name as the annotation file but with a .jpg extension
        img_file = os.path.splitext(annotation_file)[0] + '.jpg'
        img_path = os.path.join(self.img_folder, img_file)

        # Open image file
        img = Image.open(img_path)
        img = self.transform(img)

        # Load .mat annotation file
        mat = loadmat(annotation_path)
        annotation = torch.from_numpy(mat['LabelMap'].astype(np.int32))

        return img, annotation, img_file


def load_dataset(dataset_name):
    preprocessing_fn = None
    if dataset_name == 'voc':
        dataset = VOCWithPaths('../datasets/', transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
        #dataset = torchvision.datasets.VOCSegmentation('../datasets', transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
    elif dataset_name == 'cityscapes':
        dataset = CityscapesWithPaths('../datasets/cityscapes', split='train', mode='fine', target_type='semantic', transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
    elif dataset_name == 'coco':
        dataset = CocoDetectionWithPaths('../datasets/coco/val2017',
                                 '../datasets/coco/annotations/instances_val2017.json',
                                 transform=transforms.ToTensor())
        preprocessing_fn = coco_annsToMask
    elif dataset_name == 'ade20k':
        dataset = ADE20KDataset('../datasets')
    elif dataset_name == 'pascal-context':
        dataset = PascalContextDataset('../datasets/VOCdevkit/VOC2012/JPEGImages',
                                       '../datasets/VOCdevkit/VOC2012/PascalContext/trainval')
    else:
        raise NotImplementedError('Dataset {} not implemented.'.format(dataset_name))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    return dataset, data_loader, preprocessing_fn




if __name__ == '__main__':

    dataset, data_loader, preprocessing_fn = load_dataset('pascal-context')
    #data_loader = torch.utils.data.DataLoader(datase, batch_size=1, shuffle=True, num_workers=1)
    for img, anns, path in data_loader:
        print(img.shape)
        print(anns.shape)
        print(path)
        break

