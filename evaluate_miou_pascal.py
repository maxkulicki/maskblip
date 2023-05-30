import os
from maskblip_diff import MaskBLIP
import torch
from lavis.models import load_model_and_preprocess
from segmentation_dataset import SegmentationDataset
from cutler_dataset import CutlerDataset
import wandb
from multiscale_maskblip import MultiscaleMaskBLIP
import matplotlib.pyplot as plt
from xdecoder_semseg import load_xdecoder_model, segment_image
from scipy import ndimage
import numpy as np
from torch.nn import functional as F
from scipy.stats import mode
from skimage.util import view_as_windows
from tqdm import tqdm

def majority_filter(image, size):
    # Create a sliding window view of the image
    shape = (size, size)
    windowed_image = view_as_windows(image, shape)

    # Compute the mode in each window
    modes, _ = mode(windowed_image.reshape(-1, size * size), axis=1)
    modes = modes.reshape(image.shape[0] - size + 1, image.shape[1] - size + 1)

    # Pad modes to match the original image size
    pad_width = size // 2
    modes = np.pad(modes, pad_width, mode='edge')

    return modes

def apply_recursive_majority_filter(image, footprint_size=3):
    # Apply majority filter recursively until convergence
    for i in range(20):
        new_image = majority_filter(image, footprint_size)
        mask = np.abs(image - new_image) > 1e-5  # Tolerance for floating point errors
        if not np.any(mask):
            break
        image = new_image
    print("Number of iterations: {}".format(i + 1))
    return image

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

    device = 'cpu'#("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiscaleMaskBLIP(device)
    model.captioning = False
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
    if supervised:
        dataset_dir = os.path.join("datasets", "VOC2012")
        dataset = SegmentationDataset(dataset_dir, n_samples, transform=model.vis_processors["eval"], img_size=model.output_size)
    else:
        dataset_dir = os.path.join("cutler", "maskcut")
        dataset = CutlerDataset(dataset_dir, n_samples, transform=model.vis_processors["eval"], img_size=model.output_size)

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
    total_mIoU = 0
    total_clean_mIoU = 0
    for batch in tqdm(train_loader):
        images, annotations, _ = batch
        images.requires_grad = True
        images = images.to(device)
        mask = annotations.to(device)
        if model.captioning:
            output, captions = model(images)
        else:
            output = model(images)
        mIoU = compute_best_mean_IoU(mask, output)
        total_mIoU += mIoU
        print("mIoU: {}".format(mIoU.item()))

        output = output.detach().numpy()
        cleaned_output = apply_recursive_majority_filter(output)
        clean_mIoU = compute_best_mean_IoU(mask, torch.tensor(cleaned_output))
        total_clean_mIoU += clean_mIoU

        if plot:
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            ax[0].imshow(images.detach().squeeze().permute(1,2,0).numpy())
            ax[0].title.set_text("Original image")
            ax[1].imshow(output)
            ax[1].title.set_text(f"mIoU: {mIoU.item():.3f}")
            ax[2].imshow(cleaned_output)
            ax[2].title.set_text(f"cleaned mIoU: {clean_mIoU.item():.3f}")
            ax[3].imshow(mask.squeeze().numpy())
            ax[3].title.set_text("Ground truth")
            plt.show()
        if use_xdecoder:
            images.to("cuda")
            xdecoder_output = segment_image(xdecoder_model, images, captions[0], plot=True)

    print("Average mIoU: {}".format(total_mIoU / len(train_loader)))
    print("Average cleaned mIoU: {}".format(total_clean_mIoU / len(train_loader)))
