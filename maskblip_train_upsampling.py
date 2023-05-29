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

def create_binary_masks(tensor):
    if len(tensor.shape)>2:
        binary_masks=[]
        for t in tensor:
            unique_values = t.unique()
            num_masks = unique_values.size(0)
            binary_mask = torch.zeros((num_masks,) + t.shape, dtype=torch.uint8, device=tensor.device)
            for i, value in enumerate(unique_values):
                binary_mask[i] = (t == value).to(torch.uint8)
            binary_masks.append(binary_mask)
        binary_masks = torch.cat(binary_masks, dim=0)
    else:
        unique_values = tensor.unique()
        num_masks = unique_values.size(0)

        binary_mask = torch.zeros((num_masks,) + tensor.shape[1:], dtype=torch.uint8, device=tensor.device)
        for i, value in enumerate(unique_values):
            binary_mask[i] = (tensor == value).to(torch.uint8)
        binary_masks = binary_mask
    return binary_masks


def best_iou_loss(pred, target, cluster_assignment=None):
    if len(target.shape)>3:
        target = target.squeeze()

    miou = 0
    if cluster_assignment is not None:
        for i, mask in enumerate(target):
            intersection = torch.sum(pred[cluster_assignment[i]] * mask)
            union = torch.sum(pred[cluster_assignment[i]]) + torch.sum(mask) - intersection
            iou = intersection / union
            if iou > 1:
                print("wtf")
                print(iou)
            miou += iou
    else:
        for mask in target:
            intersection = torch.sum(pred * mask, (1, 2))
            union = torch.sum(pred, (1, 2)) + torch.sum(mask) - intersection
            best_iou = torch.max(intersection / union)
            if best_iou > 1:
                print("wtf")
                print(best_iou)
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


def training_step(model, loader, optimizer, loss_function, cluster_assignments=None):
    for iter, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)

        images, annotations, _ = batch
        images.requires_grad = True
        images = images.to(device)
        mask = annotations.to(device)

        output = model(images)
        if train_supervised:
            if batch_size > 1:
                masks = []
                for m in mask:
                    m = create_binary_masks(m.unsqueeze(0))
                    masks.append(m)
                #masks = torch.cat(masks, dim=0)
            else:
                masks = create_binary_masks(mask)
        loss = 0
        if cluster_assignments is not None:
            loss += loss_function(output[0], masks, cluster_assignments[iter])
        else:
            for i, m in enumerate(masks):
                loss += loss_function(output[0][i], m)
        loss = loss/len(masks)
        #loss = loss_function(output[0], masks)
        loss.backward(retain_graph=True)
        optimizer.step()
        print("Loss: {}".format(loss.item()))
        wandb.log({"loss": loss.item()})

    return loss.item()

def get_cluster_assignments(model, data_loader, shuffle=False):
    cluster_assignments = []
    for iter, batch in enumerate(data_loader):
        images, annotations, _ = batch
        if train_supervised:
            annotations = create_binary_masks(annotations)
        images = images.to(device)
        annotations = annotations.to(device)
        output = model(images)
        assignment=[]
        for mask in annotations:
            closest_cluster = torch.argmax(torch.sum(mask * output[0], axis=(1,2)))
            assignment.append(closest_cluster.item())
        cluster_assignments.append(assignment)
    return cluster_assignments


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    n_samples = 1000
    batch_size = 1
    n_epochs = 10
    lr = 0.001
    weight_decay = 0
    plot = True
    n_plots = 8
    wandb_track = True
    train_supervised = True

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
    if train_supervised:
        dataset_dir = os.path.join("datasets", "VOC2012")
        dataset = SegmentationDataset(dataset_dir, n_samples, transform=vis_processors["eval"], img_size=(96, 96))
    else:
        dataset_dir = os.path.join("cutler", "maskcut")
        dataset = CutlerDataset(dataset_dir, n_samples, transform=vis_processors["eval"], img_size=(96, 96))

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
        shuffle=True
    )

    if plot:
        plot_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=1,
            shuffle=False
        )

    best_val_loss = np.infty
    cluster_assignments = get_cluster_assignments(model, train_loader)
    print(cluster_assignments)
    # with torch.no_grad():
    #     val_loss, best_model, best_val_loss = validation_step(model, val_loader, -1, loss_function, best_val_loss,
    #                                                           best_model, parameter_selector)
    #     wandb.log({"val_loss": val_loss})
    gt_mask_dir = os.path.join("cutler", "maskcut", "results")
    if plot:
        for iter, batch in enumerate(plot_loader):
            images, annotations, masked_image = batch
            plot = wandb.Image(masked_image.squeeze().float(), caption="Ground Truth")
            wandb.log({f"Result{iter}": plot})
            images = images.to(device)
            mask = annotations.to(device)
            if train_supervised:
                mask = create_binary_masks(mask)
            result = model(images)
            loss = best_iou_loss(result[0], mask)
            plot = torch.argmax(result[0], 0).float()
            plot = plot / plot.max()
            plot = wandb.Image(plot, caption="Epoch: -1 " + " Loss: " + str(loss))
            wandb.log({f"Result{iter}": plot})
            if iter == n_plots - 1:
                break

    for epoch in range(n_epochs):
        train_loss = training_step(model, train_loader, optimizer, best_iou_loss, cluster_assignments=cluster_assignments)
        if plot:
            for iter, batch in enumerate(plot_loader):
                images, annotations, _ = batch
                images = images.to(device)
                mask = annotations.to(device)
                result = model(images)
                loss = best_iou_loss(result[0], mask)
                plot = torch.argmax(result[0], 0).float()
                plot = plot/plot.max()
                plot = wandb.Image(plot, caption="Epoch: " + str(epoch) + " Loss: " + str(loss))
                wandb.log({f"Result{iter}": plot})
                if iter == n_plots-1:
                    break