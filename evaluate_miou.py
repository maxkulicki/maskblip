import os
import torch
from segmentation_dataset import SegmentationDataset
from cutler_dataset import CutlerDataset
import wandb
#from multiscale_maskblip import MultiscaleMaskBLIP, clean_clusters
from maskblip import MaskBLIP, clean_clusters
import matplotlib.pyplot as plt
from xdecoder_semseg import load_xdecoder_model, segment_image, segment_with_sanity_check
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, Normalize
from dataset_loading import load_dataset
from nlp import get_noun_chunks, get_nouns
from PIL import Image



def compute_best_mean_IoU(ground_truth, prediction):

    best_ious = []
    for i in torch.unique(ground_truth):
        if i == 0:
            # Don't count background
            continue
        # Get masks for the current ground truth cluster
        gt_mask = (ground_truth == i)

        max_iou = 0
        for j in torch.unique(prediction):
            # Get masks for the current prediction cluster
            pred_mask = (prediction == j)

            # Compute Intersection over Union (IoU) for this pair
            intersection = torch.logical_and(gt_mask, pred_mask)
            union = torch.logical_or(gt_mask, pred_mask)

            intersection_sum = torch.sum(intersection).float()
            union_sum = torch.sum(union).float()

            # Compute IoU and update max_iou if this is the highest we've seen
            if union_sum == 0:
                # Special case where there's no ground truth and no prediction
                iou = 1.0
            else:
                iou = intersection_sum / union_sum

            max_iou = max(max_iou, iou)

        best_ious.append(max_iou)

    # Compute mean IoU
    mean_IoU = torch.mean(torch.tensor(best_ious))

    return mean_IoU

def evaluate_mIoU(dataset="pascal_context", device="cuda", **kwargs):
    #        'kmeans_range': {'values': [3, 4, 5, 6]},
    #     'pos_emb_dim': {'values': [256, 512, 768, 1024]},
    #     'smoothness_weight': {'min': 1.0, 'max': 10.0},
    #     'smoothness_theta': {'min': 0.5, 'max': 2.0},
    #     'nr_of_scales': {'values': [2, 3, 4, 5]},
    #     'scale_step': {'values': [32, 64, 128]}
    #captioning
    # nucleus vs beam search
    # repetition penalty
    #num beams
    # top_p
    #local/global/both
    #background/no background

    model = MaskBLIP(device, **kwargs)

    xdecoder_model = load_xdecoder_model("cuda")

    transform = Compose([
        # ToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    dataset, dataloader, _ = load_dataset("pascal-context")

    mIoU_list = []
    for batch in tqdm(dataloader):
        images, annotations, paths = batch
        images.requires_grad = True
        images = images.to(device)
        mask = annotations.to(device)

        output, captions = model(transform(images))
        captions = get_nouns(captions[0], model.spacy_model, add_background=True)
        print(captions)

        resized_output = F.interpolate(output.unsqueeze(0).float(), size=mask.shape[-2:], mode="nearest").to(device)

        # images = images.squeeze()
        images = Image.open(dataset.img_folder + "/" + paths[0])
        # xdecoder_output = segment_image(xdecoder_model, images, captions, plot=True).to(device)
        xdecoder_output, captions = segment_with_sanity_check(xdecoder_model, images, captions, plot=True)

        mIoU = compute_best_mean_IoU(mask, xdecoder_output.to(device))
        print("xdecoder mIoU: {}".format(mIoU.item()))

    print("Average mIoU: {}".format(sum(mIoU_list) / len(dataloader)))
    num_bins = 20
    # We can set the number of bins with the `bins` argument
    plt.hist(mIoU_list, bins=num_bins, edgecolor='black')
    plt.show()
    plt.savefig("mIoU_hist.png")
    return sum(mIoU_list) / len(dataloader)

if __name__ == "__main__":
    evaluate_mIoU(device='cpu')