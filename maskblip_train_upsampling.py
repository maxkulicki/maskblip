import os
from maskblip_diff import MaskBLIP
import numpy as np
import torch
from lavis.models import load_model_and_preprocess
from segmentation_dataset import SegmentationDataset
from cutler_dataset import CutlerDataset
import wandb

def consistency_loss(soft_partition, hard_partition, background_importance=0.1):
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
        n_clusters = soft.shape[0]
        n_gt_clusters = len(hard_partition[n].unique())

        # Compute the matrix of pairwise overlaps between clusters
        overlap_matrix = torch.zeros((n_clusters, n_gt_clusters))
        for i in range(n_clusters):
            for idx, j in enumerate(hard_partition[n].unique()):
                if j == 0:
                    overlap_matrix[i, idx] = torch.sum(torch.min(soft[:, i], (hard_partition[n,:] == j).float()) * background_importance)
                else:
                    overlap_matrix[i, idx] = torch.sum(torch.min(soft[:, i], (hard_partition[n,:] == j).float()))


        # Compute the numerator and denominator of the consistency index
        num = torch.max(torch.sum(overlap_matrix, dim=1))
        den = torch.sum((hard_partition[n,:] >= 0).float())
        result = num / den
        avg_result += result
    # Compute and return the consistency index
    return - avg_result / len(soft_partition)

def custom_loss(pred, target):
    target = target.squeeze()
    miou = 0
    for mask in target:
        best_iou = torch.max(torch.sum(pred * mask, (1,2))/torch.sum(pred, (1,2)))
        dice = 1 - 2* best_iou
        miou += best_iou

    return -miou/len(target)

def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors
    Returns:
        Soft Dice loss
    '''
    # Skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * torch.sum(y_pred * y_true, dim=axes)
    denominator = torch.sum(torch.square(y_pred) + torch.square(y_true), dim=axes)

    loss = 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))
    return loss


def training_step(model, loader, optimizer, loss_function):
    for iter, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)

        images, annotations = batch
        images.requires_grad = True
        images = images.to(device)
        mask = annotations.to(device)

        output = model(images)
        loss = loss_function(output[0], mask)
        loss.backward(retain_graph=True)
        optimizer.step()
        print("Loss: {}".format(loss.item()))
        wandb.log({"loss": loss.item()})

    return loss.item()

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    n_samples = 20
    batch_size = 1
    n_epochs = 10
    lr = 0.01
    weight_decay = 0
    plot = False
    n_plots = 4
    wandb_track = True

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model, vis_processors, _ = load_model_and_preprocess("blip_caption", "base_coco")
    model.to(device)

    model = MaskBLIP(model, device, upsampler=True)

    optimizer = torch.optim.Adam(model.upsampler.parameters(), lr=lr)

    if wandb_track:
        run = wandb.init(
            # Set the project where this run will be logged
            project="maskblip",
            group="upsampling",
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

    dataset_dir = os.path.join("cutler", "maskcut")
    dataset = CutlerDataset(dataset_dir, n_samples, transform=vis_processors["eval"], img_size=(96, 96))

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

    # with torch.no_grad():
    #     val_loss, best_model, best_val_loss = validation_step(model, val_loader, -1, loss_function, best_val_loss,
    #                                                           best_model, parameter_selector)
    #     wandb.log({"val_loss": val_loss})

    # iterate over the data loader to process images in batches
    for epoch in range(n_epochs):
        train_loss = training_step(model, train_loader, optimizer, custom_loss)
