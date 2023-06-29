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
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from dataset_loading import load_dataset
from nlp import get_noun_chunks, get_nouns
from PIL import Image

def compute_best_mean_IoU(gt_masks, predictions, sizes):
    total_mean_IoU = 0
    # print(gt_masks, "\n", predictions, "\n", sizes)
    for i, ground_truth in enumerate(gt_masks):
        prediction = predictions[i].cuda()
        #resize gt to size
        ground_truth = F.interpolate(ground_truth.unsqueeze(0).float(), size=(sizes[1][i], sizes[0][i]), mode='nearest').squeeze(0)

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
        total_mean_IoU += mean_IoU

    return total_mean_IoU / len(gt_masks)

def evaluate_mIoU(dataset="pascal_context", device="cuda", batch_size=1, **kwargs):

    dataset, dataloader, _ = load_dataset(dataset, batch_size=batch_size, device=device)

    model = MaskBLIP(device, **kwargs)
    xdecoder_model = load_xdecoder_model(device)

    transform = Compose([Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])

    mIoU_list = []
    for batch in tqdm(dataloader):
        images, annotations, paths, image_sizes = batch
        images.requires_grad = True
        images = images.to(device)
        gt_masks = annotations.to(device)

        print("-- getting captions --")
        output, captions = model(transform(images))
        print("-- got captions --")

        if 'background' in kwargs:
            captions = [get_nouns(cap, model.spacy_model, add_background=kwargs['background']) for cap in captions]
        else:
            captions = [get_nouns(cap, model.spacy_model) for cap in captions]

        # resized_output = F.interpolate(output.unsqueeze(0).float(), size=gt_masks.shape[-2:], mode="nearest").to(device)

        print("-- loading --")
        images = [Image.open(dataset.img_folder + "/" + paths[i]) for i in range(len(paths))]
        print("-- loaded --")

        # We are looping over images+captions because we need to set text encoder to predicted nouns at each prediction,
        # it cannot be done in batches
        all_xdecoder_outputs, all_captions_output = [], []
        for i, image in enumerate(images):
            xdecoder_output, caption_output = segment_with_sanity_check(xdecoder_model, image, captions[i], plot=False)
            all_xdecoder_outputs.append(xdecoder_output)
            # all_captions_output.append(caption_output)

        mIoU = compute_best_mean_IoU(gt_masks, all_xdecoder_outputs, image_sizes)
        mIoU_list.append(mIoU)

        # print("xdecoder mIoU: {}".format(mIoU.item()))

    # print("Average mIoU: {}".format(sum(mIoU_list) / len(dataloader)))
    # num_bins = 20
    # We can set the number of bins with the `bins` argument
    # plt.hist(mIoU_list, bins=num_bins, edgecolor='black')
    # plt.show()
    # plt.savefig("mIoU_hist.png")
    return sum(mIoU_list) / len(dataloader)

if __name__ == "__main__":
    evaluate_mIoU(device='cpu')