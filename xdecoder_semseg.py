# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
import sys
import logging
pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

from PIL import Image
import numpy as np
np.random.seed(1)

import torch
from torchvision import transforms
from XDecoder.utils.arguments import load_opt_from_config_files
from detectron2.data import MetadataCatalog
from XDecoder.xdecoder.BaseModel import BaseModel
from XDecoder.xdecoder import build_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

def load_xdecoder_model(device):
    pretrained_pth = 'XDecoder/weights/xdecoder_focalt_best_openseg.pt'
    opt = load_opt_from_config_files(["XDecoder/configs/xdecoder/svlp_focalt_lang.yaml"])
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().to(device)
    return model

def segment_with_sanity_check(xdecoder_model, images, noun_phrases, max_threshold=0.95, min_threshold=0.01, min_captions=2, plot=False, device='cuda'):
    output = segment_image(xdecoder_model, images, noun_phrases, plot=plot)
    output = output.unsqueeze(0).to(device)

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
            # print("No dominant classes found, removing minor classes")
            # If no dominant classes, remove the minor classes
            noun_phrases = [np for i, np in enumerate(noun_phrases) if i not in minor_classes]
        else:
            # If no classes to remove, stop and return the output
            return output, noun_phrases

        output = segment_image(xdecoder_model, images, noun_phrases, plot=False).unsqueeze(0).to(device)

    # If we reached here, it means there are less than min_captions left,
    # so just return the last resized_output we got
    return output, noun_phrases

def segment_image(model, image_ori, classes, input_tensor=False, plot=False):
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(classes + ["background"], is_eval=True)
        metadata = MetadataCatalog.get('demo')
        model.model.metadata = metadata
        model.model.sem_seg_head.num_classes = len(classes)

        t = [transforms.Resize(512, interpolation=Image.BICUBIC)]
        transform = transforms.Compose(t)

        if not input_tensor:
            width = image_ori.size[-2]
            height = image_ori.size[-1]
            image = transform(image_ori)
            image = np.asarray(image)
            #image_ori = np.asarray(image_ori)
            image = torch.from_numpy(image.copy()).permute(2, 0, 1).cuda()

        else:
            image = image_ori
            width = image.size()[-1]
            height = image.size()[-2]
            image = transform(image)
            image_ori = image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

        batch_inputs = [{'image': image.squeeze(), 'height': height, 'width': width}]
        outputs = model.forward(batch_inputs)
        sem_seg = outputs[-1]['sem_seg'].max(0)[1]
        classes_detected = sem_seg.unique()
        classes_detected = [classes[i] for i in classes_detected]
    if plot:
        plot_segmentation(image_ori, sem_seg.cpu().numpy(), classes_detected, classes)

    return sem_seg


def plot_segmentation(image, sem_seg, classes_detected, classes, gt=None, mIoU=None):
    suptitle = "Input labels: " + ','.join(classes)
    if gt is not None:
        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        fig.suptitle(suptitle, fontsize=20)
        axs[0].imshow(image)
        axs[0].set_title("Original Image", fontsize=20)
        axs[1].imshow(gt)
        axs[1].set_title("Ground Truth", fontsize=20)
        cluster_plot = axs[2].imshow(sem_seg)
        axs[2].set_title("Segmented Image", fontsize=20)
        values = np.unique(sem_seg)
        colors = [ cluster_plot.cmap(cluster_plot.norm(value)) for value in values]
        # create a patch (proxy artist) for every color
        patches = [ mpatches.Patch(color=colors[i], label=classes_detected[i] ) for i in range(len(values)) ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(suptitle, fontsize=20)
        axs[0].imshow(image)
        axs[0].set_title("Original Image", fontsize=20)
        cluster_plot = axs[1].imshow(sem_seg)
        axs[1].set_title("Segmented Image", fontsize=20)
        values = np.unique(sem_seg)
        colors = [ cluster_plot.cmap(cluster_plot.norm(value)) for value in values]
        # create a patch (proxy artist) for every color
        patches = [ mpatches.Patch(color=colors[i], label=classes_detected[i] ) for i in range(len(values)) ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches)

    if mIoU is not None:
        plt.suptitle("mIoU: {:.2f}".format(mIoU), fontsize=20)
    plt.show()
    return fig