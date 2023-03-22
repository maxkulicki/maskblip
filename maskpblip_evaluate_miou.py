import os
import sys

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from lavis.models import load_model_and_preprocess
from nltk.stem import WordNetLemmatizer
from maskblip import maskblip_segmentation
from nlp import get_noun_chunks, load_spacy, find_matching_labels
# adapt from https://github.com/CSAILVision/placeschallenge/blob/master/sceneparsing/evaluationCode/utils_eval.py
def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred)
    imLab = np.asarray(imLab)

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)
def miou(ground_truth, segmentation):
    mIoU = 0
    for class_id in np.unique(ground_truth):
        inter, union = intersectionAndUnion(segmentation, ground_truth, class_id)
        IoU = 1.0 * inter / (np.spacing(1) + union)
        mIoU += IoU
    mIoU /= len(np.unique(ground_truth))
    return mIoU

def pixel_accuracy(ground_truth, segmentation, include_background=False):
    if include_background:
        pixel_labeled = np.sum(ground_truth >= 0)
    else:
        pixel_labeled = np.sum(ground_truth > 0)

    pixel_correct = np.sum((segmentation == ground_truth) * (ground_truth > 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_accuracy
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
    ade20k_dir = "datasets/ADEChallengeData2016/"
    mapping_path = ade20k_dir + "objectInfo150.txt"
    id_to_label, label_to_id = get_ade20k_label_mapping(mapping_path)
    n_samples = 10

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True,
                                                         device=device)
    #spacy_model = load_spacy()
    similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    avg_pixel_acc = 0
    for image_path in os.listdir(ade20k_dir + "images/training/")[:n_samples]:
        annotations_path = ade20k_dir + "annotations/training/" + image_path.replace("jpg", "png")
        annotations =np.asarray(Image.open(annotations_path))
        labels = [id_to_label[str(label_id)] for label_id in np.unique(annotations) if label_id != 0]

        image_path = ade20k_dir + "images/training/" + image_path
        image = Image.open(image_path)
        output_segmentation, output_labels = maskblip_segmentation(image, model, device, vis_processors, refine=True)
        output_labels = find_matching_labels(output_labels, labels, model=similarity_model, background=True)
        output_ids = [0 if label in ['unknown', 'background'] else label_to_id[label] for label in output_labels]
        for idx, output in enumerate(np.unique(output_segmentation)):
            output_segmentation[output_segmentation == output] = output_ids[idx]

        #mean_iou = miou(annotations, output_segmentation)
        pixel_acc = pixel_accuracy(annotations, output_segmentation, include_background=True)
        avg_pixel_acc += pixel_acc
        #print("Mean IoU:", mean_iou)
        print("Pixel accuracy:", pixel_acc)
        print("Ground truth labels:", labels)
        print("Predicted labels:", output_labels)

    print("Average pixel accuracy:", avg_pixel_acc / n_samples)