import os
import sys

from matplotlib import pyplot as plt

from maskblip import maskblip_segmentation
from maskblip_diff import MaskBLIP
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from lavis.models import load_model_and_preprocess
from nltk.stem import WordNetLemmatizer
from nlp import get_noun_chunks, load_spacy, find_matching_labels
import pickle as pkl
import cv2
from utils import utils_ade20k

def synset_match(nouns, synset):
    for noun in nouns:
        for syn in synset:
            if noun.strip() in syn.split(" "):
                return True
    return False
def get_recall_precision(captions, labels):
    matches = 0
    for label in labels:
        if any(label in caption for caption in captions):
                matches += 1

    if len(labels) == 0:
        recall = 1
        precision = 1
    else:
        recall = matches / len(labels)
        precision = matches / len(captions)
    return recall, precision

def get_ade20k_label_mapping(mapping_path):
    id_to_label = {}
    label_to_id = {}
    with open(mapping_path, encoding="utf8") as f:
        for line in f:
            id = line.split()[0]
            label = line.split()[4].replace(",", "")
            id_to_label[id] = label
            label_to_id[label] = id

    return id_to_label, label_to_id


if __name__ == "__main__":
    # Load index with global information about ADE20K
    DATASET_PATH = 'datasets/mkulicki_25ec15b0'
    index_file = 'ADE20K_2021_17_01/index_ade20k.pkl'

    with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
        index_ade20k = pkl.load(f)
    n_samples = 10

    for idx in range(n_samples):
        file_name = index_ade20k['filename'][idx]
        full_file_name = '{}/{}'.format(index_ade20k['folder'][idx], index_ade20k['filename'][idx])

        info = utils_ade20k.loadAde20K('{}/{}'.format(DATASET_PATH, full_file_name))

        img = cv2.imread(info['img_name'])[:, :, ::-1]
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(img)
        label_idx = index_ade20k['objectPresence'][:, idx].nonzero()[0]
        part_idx = index_ade20k['objectIsPart'][:, idx].nonzero()[0]
        labels = [index_ade20k['objectnames'][i].split(',')[0] for i in label_idx if i not in part_idx]
        seg = cv2.imread(info['segm_name'])[:, :, ::-1]
        ax[1].imshow(seg)
        clusters = cv2.resize(seg, (24,24), interpolation=cv2.INTER_NEAREST)
        ax[2].imshow(clusters)
        plt.show()
        print(labels)


    ade20k_dir = "datasets/ADEChallengeData2016/"
    mapping_path = ade20k_dir + "objectInfo150.txt"
    id_to_label, label_to_id = get_ade20k_label_mapping(mapping_path)
    n_samples = 100

    device = 'cpu'#("cuda" if torch.cuda.is_available() else "cpu")

    model, vis_processors, _ = load_model_and_preprocess("blip_caption", "base_coco")
    model = MaskBLIP(model, device, use_ssn=True)

    batch_size = 1 # set the batch size

    # get a list of image paths
    image_paths = [os.path.join(ade20k_dir, "images", "training", filename) for filename in
                   os.listdir(os.path.join(ade20k_dir, "images", "training"))]
    annotations_paths = [os.path.join(ade20k_dir, "annotations", "training", filename.replace("jpg", "png")) for
                         filename in os.listdir(os.path.join(ade20k_dir, "images", "training"))]
    # create a data loader to load images in batches
    data_loader = torch.utils.data.DataLoader(
        dataset=image_paths,
        batch_size=batch_size,
        shuffle=True
    )

    # iterate over the data loader to process images in batches
    for batch in data_loader:
        annotations = []
        labels = []
        images = []
        for image_path, annotation_path in zip(batch, annotations_paths):
            annotation = np.asarray(Image.open(annotation_path))
            annotations.append(annotation)
            labels.extend([id_to_label[str(label_id)] for label_id in np.unique(annotation) if label_id != 0])

            image = Image.open(image_path)
            image = vis_processors["eval"](image).unsqueeze(0).to(device)
            images.append(image)

        images = torch.cat(images, dim=0)
        output_segmentation, output_labels = model.forward(images)
        print(output_labels)
        print(labels)




    #
    # for image_path in os.listdir(ade20k_dir + "images/training/")[:n_samples]:
    #     annotations_path = ade20k_dir + "annotations/training/" + image_path.replace("jpg", "png")
    #     annotations = np.asarray(Image.open(annotations_path))
    #     labels = [id_to_label[str(label_id)] for label_id in np.unique(annotations) if label_id != 0]
    #
    #     image_path = ade20k_dir + "images/training/" + image_path
    #     image = Image.open(image_path)
    #     image = vis_processors["eval"](image).unsqueeze(0).to(device)
    #     output_segmentation, output_labels = model.forward(image)
    #     print(output_labels)
    #     print(labels)


