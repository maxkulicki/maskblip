import random

import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt
import json
import os
import torch.nn as nn
from lavis.models.vit import Attention
import torch
from lavis.models import load_model_and_preprocess
from maskblip_diff import MaskBLIP
from segmentation_dataset import SegmentationDataset
import wandb

def training_step(model, loader, optimizer, loss_function):
    avg_loss = 0
    for iter, batch in enumerate(loader):
        optimizer.zero_grad()
        images, annotations = batch
        images.requires_grad = True
        images = images.to(device)
        mask = annotations.to(device)
        clusters, captions = model(images)
        if iter in range(3):
            fig, ax = plt.subplots(1, 2)
            img = images[0].permute(1, 2, 0).detach().cpu().numpy()
            m = clusters.detach().cpu().numpy()
            ax[0].imshow(img)
            ax[1].imshow(m)
            ax[1].set_title(str(captions).replace(",", ",\n"))
            wandb.log({f"result{iter}": fig})
        loss = loss_function(clusters, captions, mask, model)
        loss.backward(retain_graph=True)
        optimizer.step()
        wandb.log({"train_loss": loss.item()})
        print(loss.item())
        avg_loss += loss.item()

    wandb.log({"avg_train_loss": loss.item()})

    return avg_loss / len(loader)

def calculate_overlap(segmentation_map1, segmentation_map2):
    overlap_dict = {}
    for seg1_id in torch.unique(segmentation_map1):
        mask1 = segmentation_map1 == seg1_id
        overlap_list = []
        for seg2_id in torch.unique(segmentation_map2):
            if seg2_id == 0:
                continue
            mask2 = segmentation_map2 == seg2_id
            overlap_area = torch.sum(mask1 & mask2)
            if overlap_area > 0:
                overlap_list.append((seg2_id, overlap_area))
        total_overlap_area = sum([x[1] for x in overlap_list])
        if total_overlap_area == 0:
            continue
        overlap_distribution = torch.tensor([(x[0], x[1]/total_overlap_area) for x in overlap_list])
        segment_ids = torch.tensor(overlap_distribution[:, 0], dtype=torch.int64)
        percentages = torch.tensor(overlap_distribution[:, 1], dtype=torch.float32)
        overlap_dict[seg1_id.item()] = {'segment_ids': segment_ids, 'percentages': percentages}
    return overlap_dict


def get_text_features(model, text):
    text = model.tokenizer(text, return_tensors="pt", padding=True).to(model.device)
    text_output = model.text_encoder(
        text.input_ids,
        attention_mask=text.attention_mask,
        return_dict=True,
        mode="text",
    )
    text_embed = text_output.last_hidden_state[:,0,:]
    return text_embed

def captioning_loss(clusters, captions, mask, model):
    labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']
    overlap = calculate_overlap(clusters, mask)
    distances = torch.tensor(0.0).to(device)
    for i, c in enumerate(overlap.keys()):
        caption = captions[c]
        # caption = random.choice(['airplane', 'bicycle', 'helicopter', 'fish', 'spaghetti'])
        gt_labels = [labels[x.item()] for x in overlap[c]['segment_ids']]
        print(caption)
        print(gt_labels)
        caption_emb = get_text_features(model.BLIPcap, caption)
        gt_emb = torch.stack([get_text_features(model.BLIPcap, x) for x in gt_labels])
        weights = overlap[c]['percentages'].to(device)
        #dist = torch.cdist(caption_emb, gt_emb).squeeze()
        dist = -(gt_emb @ caption_emb.T).squeeze()
        if len(dist.shape)<1:
            dist = dist.unsqueeze(0)
        distance = (dist @ weights).squeeze() / weights.sum()
        distances += distance
    return distances/len(overlap.keys())



if __name__ == '__main__':
    batch_size = 1
    n_samples = 1
    n_epochs = 5
    lr = 1e-1
    wandb_track = True

    device = ("cuda" if torch.cuda.is_available() else "cpu")


    model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption", "base_coco")

    model2, _, _ = load_model_and_preprocess(name="blip_feature_extractor", model_type="base",
                                                                      is_eval=True, device=device)

    model.text_encoder = model2.text_encoder
    model.tokenizer = model2.tokenizer
    del model2

    model.to(device)
    model = MaskBLIP(model, device, captioning_adapter=True)

    dataset_dir = os.path.join("datasets","VOC2012")
    dataset = SegmentationDataset(dataset_dir, n_samples, transform=vis_processors["eval"])

    # proportions = [.9, .1]
    # lengths = [int(p * len(dataset)) for p in proportions]
    # lengths[-1] = len(dataset) - sum(lengths[:-1])
    # train_set, val_set = torch.utils.data.random_split(dataset, lengths)

    train_set = dataset

    optimizer = torch.optim.Adam(model.captioning_adapter.parameters(), lr=1e-3)

    if wandb_track:
        run = wandb.init(
            # Set the project where this run will be logged
            project="maskblip",
            group="captioning adapter",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "epochs": n_epochs,
                "batch_size": batch_size,
                "n_samples": n_samples,
            })
    else:
        run = wandb.init(mode = "disabled")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=False
    )
    # val_loader = torch.utils.data.DataLoader(
    #     dataset=val_set,
    #     batch_size=batch_size,
    #     shuffle=True
    # )

    for epoch in range(n_epochs):
        train_loss = training_step(model, train_loader, optimizer, captioning_loss)

