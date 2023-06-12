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
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode
from nltk.corpus import wordnet
import itertools

pascal_classes = [
    ["background", "backdrop", "setting"],
    ["aeroplane", "airplane", "plane", "jet", "aircraft"],
    ["bicycle", "bike", "cycle"],
    ["bird", "avian", "parrot"],
    ["boat", "ship", "vessel", "yacht"],
    ["bottle", "jar", "container"],
    ["bus", "coach", "minibus"],
    ["car", "automobile", "vehicle", "motorcar"],
    ["cat", "kitten", "feline"],
    ["chair", "seat", "stool"],
    ["cow", "bovine", "cattle"],
    ["table", "desk", "counter"],
    ["dog", "puppy", "hound", "canine"],
    ["horse", "stallion", "mare"],
    ["motorbike", "motorcycle", "bike"],
    ["person", "people", "man", "woman", "men", "women"],
    ["plant", "flower", "tree", "vegetation"],
    ["sheep", "lamb", "ram"],
    ["sofa", "couch", "settee"],
    ["train", "locomotive", "railway"],
    ["monitor", "screen", "tv", "television", "computer", "display"]
]

# def get_synonyms(word):
#     synonyms = []
#     for syn in wordnet.synsets(word):
#         for lemma in syn.lemmas():
#             synonyms.append(lemma.name())
#     return synonyms

def check_word_in_caption(caption, class_names):
    for name in class_names:
        if name in caption:
            return True
    return False

def find_matching_caption(captions, clusters, gt_mask):
    clusters = clusters.to(device)
    gt_mask = gt_mask.to(device)
    most_overlap = 0
    clusters = Resize(size=gt_mask.size()[1:], antialias=True, interpolation=InterpolationMode.NEAREST_EXACT)(clusters)
    for i, cluster in enumerate(clusters.unique()):
        overlap = ((clusters == cluster).int() * gt_mask).sum()
        if overlap > most_overlap:
            most_overlap = overlap
            best_caption = captions[i]
    return best_caption

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    n_samples = 3
    batch_size = 1
    plot = True
    supervised = True
    device = 'cpu'#("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = MultiscaleMaskBLIPK(device, scales=[384, 512])
    model.captioning = True



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

    eval_mode = ['any_cluster']#["gt_mask", "any_cluster", "best_cluster"]
    attention_modes = ["global", "local", "concat"]
    configs = list(itertools.product(eval_mode, attention_modes))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False
    )
    result_dict = {}
    for config in configs:
        print("Evaluating config: ", config)
        eval_mode, attention_mode = config
        total_recall = 0
        for batch in tqdm(dataloader):
            images, annotations = batch
            images.requires_grad = True
            images = images.to(device)
            mask = annotations.to(device)
            if eval_mode == "gt_mask":
                output, captions = model(images, gt_mask=mask, clean=False)
            else:
                output, captions = model(images, clean=True)
            print(captions)
            print(len(captions[0]))
            classes = mask.unique()
            n_labels=0
            n_matches=0
            if len(captions[0]) < 2:
                continue

            gt_labels = [pascal_classes[i] for i in classes]

            if eval_mode == "gt_mask":
                for i, caption in enumerate(captions[0]):
                    gt_label = pascal_classes[classes[i]]
                    if "background" not in gt_label:
                        n_labels += 1
                        if check_word_in_caption(caption, gt_label):
                            n_matches += 1
            else:
                print(gt_labels)
                for i, label in enumerate(gt_labels):
                    if "background" not in label:
                        n_labels += 1
                        # best cluster only
                        if eval_mode == "best_cluster":
                            matching_caption = find_matching_caption(captions[0], label, output, mask)
                            if check_word_in_caption(matching_caption, label):
                                n_matches += 1
                                break

                        # any cluster
                        elif eval_mode == "any_cluster":
                            for caption in captions[0]:
                                if check_word_in_caption(caption, label):
                                    n_matches += 1
                                    break

            recall = n_matches / n_labels
            total_recall += recall
            print(recall)

            output = output.cpu().detach().numpy()

            if plot:
                unique_clusters = np.unique(output)
                cmap = plt.cm.get_cmap('tab20', len(unique_clusters))  # 'tab20' is a good colormap for categorical data
                # Create a plot with a colorbar that has labels
                fig, axs = plt.subplots(1, 2, figsize=(25, 7))  # 1 row, 2 columns
                # The first subplot will display your raw image
                cax = axs[0].imshow(output.squeeze())
                axs[0].set_title('Segmentation')
                # This creates a colorbar for the segmentation plot
                cbar = fig.colorbar(cax, ax=axs[0], ticks=unique_clusters, spacing='proportional')
                # This sets the labels of the colorbar to correspond to your captions
                cbar.ax.set_yticklabels(captions[0])  # change fontsize and rotation as necessary

                classes = [pascal_classes[i] for i in unique_clusters]
                axs[1].imshow(images.squeeze().permute(1, 2, 0).detach().numpy())
                axs[1].set_title(gt_labels)
                # Show the plot
                plt.tight_layout()
                plt.show()

        avg_recall = total_recall / len(dataloader)
        result_dict[config] = avg_recall
        print(model.img_size)
        print("Attention mode: ", attention_mode)
        print("Use gt masks: ", eval_mode)
        print("Average recall: ", avg_recall)

    print(result_dict)
    with open("results.txt", "w") as f:
        f.write(str(result_dict))
