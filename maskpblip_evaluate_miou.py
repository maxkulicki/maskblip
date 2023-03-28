import os
import sys

from matplotlib import pyplot as plt

from maskblip import maskblip_segmentation

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from lavis.models import load_model_and_preprocess
from nltk.stem import WordNetLemmatizer
from nlp import get_noun_chunks, load_spacy, find_matching_labels
# adapt from https://github.com/CSAILVision/placeschallenge/blob/master/sceneparsing/evaluationCode/utils_eval.py

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

def evaluate_captioning_model(n_segments, threshold, n_samples):
    avg_pixel_acc = 0
    avg_recall = 0
    avg_precision = 0
    avg_n_clusters = 0
    for image_path in os.listdir(ade20k_dir + "images/training/")[:n_samples]:
        annotations_path = ade20k_dir + "annotations/training/" + image_path.replace("jpg", "png")
        annotations = np.asarray(Image.open(annotations_path))
        labels = [id_to_label[str(label_id)] for label_id in np.unique(annotations) if label_id != 0]

        image_path = ade20k_dir + "images/training/" + image_path
        image = Image.open(image_path)
        output_segmentation, output_labels = maskblip_segmentation(image, model, device, vis_processors, refine=True, n_segments=n_segments, threshold=threshold)
        n_clusters = len(np.unique(output_segmentation))
        recall, precision = get_recall_precision(output_labels, labels)


        # matching captions with ground truth labels
        output_labels = find_matching_labels(output_labels, labels, model=similarity_model, background=True)
        output_ids = [0 if label in ['unknown', 'background'] else label_to_id[label] for label in output_labels]
        for idx, output in enumerate(np.unique(output_segmentation)):
            output_segmentation[output_segmentation == output] = output_ids[idx]


        pixel_acc = pixel_accuracy(annotations, output_segmentation, include_background=True)

        avg_pixel_acc += pixel_acc
        avg_recall += recall
        avg_precision += precision
        avg_n_clusters += n_clusters

    avg_pixel_acc /= n_samples
    avg_recall /= n_samples
    avg_precision /= n_samples
    avg_n_clusters /= n_samples

    return avg_pixel_acc, avg_recall, avg_precision, avg_n_clusters

def plot_metric(metrics, metric_name, thresholds, segment_numbers):
    metrics = np.array(metrics).reshape(len(thresholds), len(segment_numbers))
    fig, ax = plt.subplots()
    ax.matshow(metrics, cmap='seismic')
    ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    ax.set_title(metric_name)
    ax.set_xticks(np.arange(len(segment_numbers)))
    ax.set_yticks(np.arange(len(thresholds)))
    ax.set_xticklabels(segment_numbers)
    ax.set_yticklabels(thresholds)
    ax.set_xlabel('Number of segments')
    ax.set_ylabel('Threshold')
    for (i, j), z in np.ndenumerate(metrics):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ade20k_dir = "datasets/ADEChallengeData2016/"
    mapping_path = ade20k_dir + "objectInfo150.txt"
    id_to_label, label_to_id = get_ade20k_label_mapping(mapping_path)
    n_samples = 1003

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True,
                                                         device=device)
    #spacy_model = load_spacy()
    similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    thresholds = [0.999, 0.9999, 0.99995]
    segment_numbers = [4, 9, 16, 25]
    clustering_methods = [(x, y) for x in thresholds for y in segment_numbers]

    pixel_accuracies = []
    recalls = []
    precisions = []
    cluster_numbers = []

    for threshold, n_segments in clustering_methods:
        print(f"Evaluating local captioning with threshold {threshold} and {n_segments} segments")
        pix_accuracy, recall, precision, cluster_number = evaluate_captioning_model(n_segments, threshold, n_samples)
        pixel_accuracies.append(pix_accuracy)
        recalls.append(recall)
        precisions.append(precision)
        cluster_numbers.append(cluster_number)
        print(f"Pixel accuracy: {pix_accuracy}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")
        print(f"Number of clusters: {cluster_number}")

    plot_metric(pixel_accuracies, "Average pixel accuracy", thresholds, segment_numbers)
    plot_metric(recalls, "Average Recall", thresholds, segment_numbers)
    plot_metric(precisions, "Average precision", thresholds, segment_numbers)
    plot_metric(cluster_numbers, "Average nr of clusters", thresholds, segment_numbers)