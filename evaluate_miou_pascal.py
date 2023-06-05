import os
from maskblip_diff import MaskBLIP
import torch
from lavis.models import load_model_and_preprocess
from segmentation_dataset import SegmentationDataset
from cutler_dataset import CutlerDataset
import wandb
#from multiscale_maskblip import MultiscaleMaskBLIP, clean_clusters
from multiscale_maskblip_kmeans import MultiscaleMaskBLIPK, clean_clusters
import matplotlib.pyplot as plt
from xdecoder_semseg import load_xdecoder_model, segment_image
from scipy import ndimage
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
import cv2
from torchvision.transforms import Compose, ToTensor, Normalize




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


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    n_samples = 500
    batch_size = 1
    plot = False
    wandb_track = False
    supervised = True

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = MultiscaleMaskBLIPK(device)
    captioning = False
    model.captioning = captioning
    use_xdecoder = False

    if use_xdecoder:
        xdecoder_model = load_xdecoder_model("cuda")


    if wandb_track:
        run = wandb.init(
            # Set the project where this run will be logged
            project="maskblip",
            group="multiscale",
            # Track hyperparameters and run metadata
            config={
                "batch_size": batch_size,
                "n_samples": n_samples,
            })
    else:
        run = wandb.init(mode = "disabled")

    transform = Compose([
    ToTensor(),
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])
    if supervised:
        dataset_dir = os.path.join("datasets", "VOC2012")
        dataset = SegmentationDataset(dataset_dir, n_samples, transform=transform, img_size=model.output_size)
    else:
        dataset_dir = os.path.join("cutler", "maskcut")
        dataset = CutlerDataset(dataset_dir, n_samples, transform=transform, img_size=model.output_size)

    proportions = [.9, .1]
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    train_set, val_set = torch.utils.data.random_split(dataset, lengths)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=False
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False
    )
    mIoU_list = []
    clean_mIoU_list = []
    for batch in tqdm(train_loader):
        images, annotations = batch
        images.requires_grad = True
        images = images.to(device)
        mask = annotations.to(device)
        if model.captioning:
            output, captions = model(images)
            print(captions)
        else:
            output = model(images)

        resized_output = F.interpolate(output.unsqueeze(0).float(), size=mask.shape[-2:], mode="nearest").to(device)
        mIoU = compute_best_mean_IoU(mask, resized_output)
        mIoU_list.append(mIoU.item())
        print("mIoU: {}".format(mIoU.item()))

        output = output.detach().numpy()
        cleaned_output = clean_clusters(output)
        if captioning:
            clean_captions = model.generate_captions(images, torch.tensor(cleaned_output).unsqueeze(0))
            print(clean_captions)

        try:
            resized_cleaned_output = F.interpolate(torch.tensor(cleaned_output).unsqueeze(0).unsqueeze(0).float(),
                                                   size=mask.shape[-2:], mode="nearest").to(device)
            clean_mIoU = compute_best_mean_IoU(mask, resized_cleaned_output)
            clean_mIoU_list.append(clean_mIoU.item())
        except Exception as e:
            print(f"reesized clean error")
            print(f"Error details: {e}")
            continue


        if plot:
            fig, ax = plt.subplots(1, 4, figsize=(23, 5))
            ax[0].imshow(images.detach().squeeze().permute(1,2,0).numpy())
            ax[0].title.set_text("Original image")
            ax[1].imshow(resized_output.squeeze().numpy())
            ax[1].title.set_text(f"mIoU: {mIoU.item():.3f}")
            ax[2].imshow(resized_cleaned_output.squeeze().numpy())
            ax[2].title.set_text(f"cleaned mIoU: {clean_mIoU.item():.3f}")
            ax[3].imshow(mask.squeeze().numpy())
            ax[3].title.set_text("Ground truth")
            plt.show()
        if use_xdecoder:
            images.to("cuda")
            xdecoder_output = segment_image(xdecoder_model, images, captions[0], plot=True)

    print("Average mIoU: {}".format(sum(mIoU_list) / len(train_loader)))
    print("Average cleaned mIoU: {}".format(sum(clean_mIoU_list) / len(train_loader)))
    num_bins = 20
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # We can set the number of bins with the `bins` argument
    axs[0].hist(mIoU_list, bins=num_bins, edgecolor='black')
    axs[1].hist(clean_mIoU_list, bins=num_bins, edgecolor='black')
    plt.show()
    plt.savefig("mIoU_hist.png")
