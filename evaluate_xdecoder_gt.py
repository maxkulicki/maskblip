import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, Normalize, PILToTensor, Resize
from xdecoder_semseg import load_xdecoder_model, segment_image, plot_segmentation
from nlp import get_noun_chunks, get_nouns, load_spacy
from PIL import Image
import json
from scipy.stats import mode


def preprocess_VOC_mask(annotation_path):
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
def segment_with_sanity_check(xdecoder_model, images, noun_phrases, max_threshold=0.95, min_threshold=0.01, min_captions=3, plot=False, device='cuda:0'):
    output = torch.tensor(
        segment_image(xdecoder_model, images, noun_phrases, plot=plot)).unsqueeze(0).to(device)

    while len(noun_phrases) >= min_captions:
        class_counts = torch.bincount(output.contiguous().view(-1))

        total_pixels = float(output.numel())

        # Find the classes with occurrence more than max_threshold or less than min_threshold
        dominant_classes = ((class_counts / total_pixels) > max_threshold).nonzero(as_tuple=True)[0].tolist()
        minor_classes = ((class_counts / total_pixels) < min_threshold).nonzero(as_tuple=True)[0].tolist()

        # Check if there are any classes to remove
        if dominant_classes:
            # Remove the dominant classes from the list of captions and run again
            noun_phrases = [np for i, np in enumerate(noun_phrases) if i not in dominant_classes]
        elif minor_classes:
            # If no dominant classes, remove the minor classes
            noun_phrases = [np for i, np in enumerate(noun_phrases) if i not in minor_classes]
        else:
            # If no classes to remove, stop and return the output
            return output

        output = torch.tensor(segment_image(xdecoder_model, images, noun_phrases, plot=False)).unsqueeze(0).to(device)

    # If we reached here, it means there are less than min_captions left,
    # so just return the last resized_output we got
    return output
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
    batch_size = 1
    plot = True
    use_nouns_only = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spacy_model = load_spacy()
    xdecoder_model = load_xdecoder_model(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    t = []
    t.extend([Resize((512), interpolation=Image.BICUBIC), PILToTensor()])
    # t.extend([PILToTensor()])

    transform = Compose(t)

    image_names = []

    with open("VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt", 'r') as file:
        for line in file:
            image_name = line.strip()  # Remove leading/trailing whitespaces
            image_names.append(image_name + ".png")
    print(len(image_names))

    with open('pascal_classes.txt', 'r') as f:
        classes = f.read().splitlines()

    mIoU_list = []
    bad_mIoU_captions = {}

    for i, path in enumerate(tqdm(image_names)):
        # xdecoder_model = load_xdecoder_model(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print(path)
        image_ori = Image.open(f"VOCdevkit/VOC2012/JPEGImages/{path}".replace(".png", ".jpg"))
        image = image_ori
        image = transform(image_ori)
        # image =  Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(image.float())
        mask_path = f"VOCdevkit/VOC2012/SegmentationClass/{path}"
        mask = preprocess_VOC_mask(mask_path).to(device)
        # gt classes
        noun_phrases = classes
        # noun_phrases = [classes[i] for i in mask.unique()]
        # print("Ground truth classes: {}".format(noun_phrases))
        output, _ = segment_image(xdecoder_model, image, noun_phrases, input_tensor=False, plot=False)
        output = output.to(device)
        resized_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=output.shape, mode='nearest')

        mIoU = compute_best_mean_IoU(resized_mask, output)

        mIoU_list.append(mIoU.item())
        print("mIoU: {}".format(mIoU.item()))
        print(torch.unique(mask))
        print([classes[i] for i in torch.unique(mask)])
        if mIoU < 0:
            output = output.squeeze().cpu().numpy()
            mask = mask.squeeze().cpu().numpy()
            classes_detected = [noun_phrases[i] for i in np.unique(output)]
            fig = plot_segmentation(image_ori, output, classes_detected, classes, gt=mask, mIoU=mIoU.item())
            fig.savefig("bad_results/{}.png".format(i))
        # if i > 10:
        #     break

    print("Average mIoU: {}".format(sum(mIoU_list) / len(mIoU_list)))
    num_bins = 20
