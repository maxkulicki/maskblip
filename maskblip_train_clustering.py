import os

from matplotlib import pyplot as plt
from maskblip_diff import MaskBLIP
from PIL import Image
import numpy as np
import torch
from lavis.models import load_model_and_preprocess
from lavis.models.vit import Attention
import torch.nn as nn
from segmentation_dataset import SegmentationDataset
import wandb

class embedding_adapter(nn.Module):
    def __init__(self, device, embed_dim=768):
        super(embedding_adapter, self).__init__()
        self.adapter = Attention(embed_dim)
        self.adapter.to(device)
        self.batch_norm = nn.BatchNorm1d(576)
        self.batch_norm.to(device)
    def forward(self, x):
        x = self.adapter(x)
        x = self.batch_norm(x)
        return x


class SLICParameterSelector(nn.Module):
    def __init__(self):
        super(SLICParameterSelector, self).__init__()
        self.conv1 = nn.Conv2d(768, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.reshape(-1, 64 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        # map to range (2,7)
        n_clusters = ((torch.sigmoid(x[:, 0])-0.5) * 6).int() + 5
        # map to int range (3,10)
        n_iters = 3 + (torch.sigmoid(x[:, 1]) * 7).int()
        # map to range (0,0.1)
        compactness = torch.sigmoid(x[:, 2]) * 0.01 #torch.sigmoid(x[:, 2]) * 0 + 0.00001

        return (n_clusters, n_iters, compactness)

def consistency_index(soft_partition, hard_partition):
    """
    Computes the consistency index between a soft partition and a hard partition.

    Parameters:
        soft_partition (torch.Tensor): The soft partition tensor of shape (n_points, n_clusters).
        hard_partition (torch.Tensor): The hard partition tensor of shape (n_points,).

    Returns:
        float: The consistency index between the two partitions.
    """
    avg_result = 0
    for n, soft in enumerate(soft_partition):
        n_clusters = soft.shape[1]

        # Compute the matrix of pairwise overlaps between clusters
        overlap_matrix = torch.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                overlap_matrix[i, j] = torch.sum(torch.min(soft[:, i], (hard_partition[n,:] == j).float()))

        # Compute the numerator and denominator of the consistency index
        num = torch.max(torch.sum(overlap_matrix, dim=1))
        den = torch.sum((hard_partition[n,:] >= 0).float())
        result = num / den
        avg_result += result
    # Compute and return the consistency index
    return avg_result / len(soft_partition)
def loss_function(output, target):
    return -consistency_index(output, target)
    #return -torch.log(consistency_index(output, target))

def training_step(model, loader, optimizer, loss_function):
    for iter, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        if use_adapter:
            adapter_layer.train()
        else:
            parameter_selector.train()

        images, annotations = batch
        images.requires_grad = True
        images = images.to(device)
        mask = annotations.to(device)

        image_emb = model.BLIPcap.forward_encoder({"image": images})[:, :-1, :]
        del images
        if use_adapter:
            image_emb = adapter_layer(image_emb)
            parameters = None
        else:
            parameters = parameter_selector(image_emb.reshape(-1, 24,24,768).permute(0, 3, 1, 2))
            n_clusters, n_iters, compactness = parameters
        output = model.clustering_model(image_emb, parameters=parameters, training=True)
        del image_emb
        loss = loss_function(output, mask.flatten(1))
        loss.backward(retain_graph=True)
        optimizer.step()
        if use_adapter:
            wandb.log({"train_loss": loss.item()})
        else:
            wandb.log({"train_loss": loss.item(), "n_clusters": torch.mean(n_clusters.float()).item(),
                       "n_iters": torch.mean(n_iters.float()).item(), "compactness": torch.mean(compactness.float()).item()})

    return loss.item()

def validation_step(model, val_loader, epoch, loss_function, best_val_loss, best_model, current_model):
    avg_val_loss = 0
    for iter, batch in enumerate(val_loader):
        if use_adapter:
            adapter_layer.eval()
        else:
            parameter_selector.eval()

        images, annotations = batch
        images = images.to(device)
        mask = annotations.to(device)

        image_emb = model.BLIPcap.forward_encoder({"image": images})[:, :-1, :]
        del images
        if use_adapter:
            image_emb = adapter_layer(image_emb)
            parameters = None
        else:
            parameters = parameter_selector(image_emb.reshape(-1, 24, 24, 768).permute(0, 3, 1, 2))
        output = model.clustering_model(image_emb, parameters=parameters, training=True)
        loss = loss_function(output, mask.flatten(1))
        del image_emb
        avg_val_loss += loss.item()

        if loss.item() < best_val_loss:
            best_model = current_model
            best_val_loss = loss.item()
    avg_val_loss /= len(val_loader)
    print("Epoch: {}, Val Loss: {}".format(epoch, avg_val_loss))

    return avg_val_loss, best_model, best_val_loss

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    n_samples = 1000
    batch_size = 20
    n_epochs = 10
    lr = 0.002
    weight_decay = 0
    plot = True
    n_plots = 4
    wandb_track = True
    use_adapter = True

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model, vis_processors, _ = load_model_and_preprocess("blip_caption", "base_coco")
    model.to(device)

    parameter_selector = SLICParameterSelector().to(device)
    model = MaskBLIP(model, device)
    if use_adapter:
        adapter_layer = embedding_adapter(device, 768)
        adapter_layer.to(device)
        optimizer = torch.optim.Adam(adapter_layer.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(parameter_selector.parameters(), lr=lr)

    if wandb_track:
        run = wandb.init(
            # Set the project where this run will be logged
            project="maskblip",
            group="2nd attempt adapter training",
            # Track hyperparameters and run metadata
            config={
                "weight_decay": weight_decay,
                "learning_rate": lr,
                "epochs": n_epochs,
                "batch_size": batch_size,
                "n_samples": n_samples,
            })
    else:
        run = wandb.init(mode = "disabled")

    dataset_dir = os.path.join("datasets","VOC2012")
    dataset = SegmentationDataset(dataset_dir, n_samples, transform=vis_processors["eval"])

    proportions = [.9, .1]
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    train_set, val_set = torch.utils.data.random_split(dataset, lengths)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=True
    )

    if plot:
        plots = [[] for x in range(n_plots)]
        losses = [[] for x in range(n_plots)]
        embs = []
        masks = []
        imgs = []
        plot_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=1,
            shuffle=True
        )
        for iter, batch in enumerate(plot_loader):
            images, annotations = batch
            images = images.to(device)
            mask = annotations.to(device)
            image_emb = model.BLIPcap.forward_encoder({"image": images})[:, :-1, :]
            output = model.clustering_model(image_emb, parameters=None, training=False)
            embs.append(image_emb)
            masks.append(mask)
            imgs.append(images)
            plots.append(output)

            if iter == n_plots-1:
                break

    best_val_loss = np.infty
    best_model = parameter_selector

    with torch.no_grad():
        val_loss, best_model, best_val_loss = validation_step(model, val_loader, -1, loss_function, best_val_loss,
                                                              best_model, parameter_selector)
        wandb.log({"val_loss": val_loss})

        if plot:
            for i in range(n_plots):
                if use_adapter:
                    emb = adapter_layer(embs[i])
                    output = model.clustering_model(emb, parameters=None, training=True)
                else:
                    emb = embs[i]
                    parameters = parameter_selector(embs[i].reshape(-1, 24, 24, 768).permute(0, 3, 1, 2))
                    output = model.clustering_model(emb, parameters=parameters, training=True)
                loss = loss_function(output, mask.flatten(1))
                hard_labels = output[0].argmax(1).unflatten(0, (24, 24))
                hard_labels = hard_labels.squeeze(0).cpu().detach().numpy()

                gt_mask = masks[i].clone().squeeze().cpu().detach().numpy()
                gt_mask = (gt_mask - np.min(gt_mask)) / (np.max(gt_mask) - np.min(gt_mask))
                gt_mask = wandb.Image(gt_mask, caption="Ground_truth ")
                wandb.log({f"Result{i}": gt_mask})

                result = (hard_labels - np.min(hard_labels)) / (np.max(hard_labels) - np.min(hard_labels))
                result = wandb.Image(result, caption="Epoch: " + str(-1) + " Loss: " + str(loss.item()))
                wandb.log({f"Result{i}": result})



    # iterate over the data loader to process images in batches
    for epoch in range(n_epochs):
        train_loss = training_step(model, train_loader, optimizer, loss_function)

        with torch.no_grad():
            val_loss, best_model, best_val_loss = validation_step(model, val_loader, epoch, loss_function, best_val_loss, best_model, parameter_selector)
            wandb.log({"val_loss": val_loss})

            if plot:
                for i in range(n_plots):
                    parameters = parameter_selector(embs[i].reshape(-1, 24, 24, 768).permute(0, 3, 1, 2))
                    emb = adapter_layer(embs[i])
                    output = model.clustering_model(emb, parameters=parameters, training=True)
                    loss = loss_function(output, mask.flatten(1))
                    hard_labels = output[0].argmax(1).unflatten(0, (24, 24))
                    hard_labels = hard_labels.squeeze(0).cpu().detach().numpy()
                    result = (hard_labels-np.min(hard_labels))/(np.max(hard_labels)-np.min(hard_labels))
                    result = wandb.Image(result, caption="Epoch: " + str(epoch) + " Loss: " + str(loss.item()))
                    wandb.log({f"Result{i}": result})

    torch.save(best_model.state_dict(), os.path.join("model_weights", "slic_param_selector", "param_selector.pt"))
