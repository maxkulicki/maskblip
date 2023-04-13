import os

from matplotlib import pyplot as plt
from maskblip_diff import MaskBLIP
from PIL import Image
import numpy as np
import torch
from lavis.models import load_model_and_preprocess
import torch.nn as nn
from segmentation_dataset import SegmentationDataset
import wandb

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

        # map to set {4,9,16,25,36}
        n_clusters = torch.square((torch.sigmoid(x[:, 0]) * 3 ).int() + 2)
        # map to int range (3,10)
        n_iters = 3 + (torch.sigmoid(x[:, 1]) * 7).int()
        # map to range (0,0.1)
        compactness = torch.sigmoid(x[:, 2]) * 10 #torch.sigmoid(x[:, 2]) * 0 + 0.00001

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

if __name__ == "__main__":


    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model, vis_processors, _ = load_model_and_preprocess("blip_caption", "base_coco")
    model.to(device)

    parameter_selector = SLICParameterSelector().to(device)
    model = MaskBLIP(model, device)

    cat_test = False
    if cat_test:
        optimizer = torch.optim.SGD(parameter_selector.parameters(), lr=2)

        image = Image.open("images/cat.jpg")
        image = vis_processors["eval"](image).unsqueeze(0).to(device)
        image.requires_grad = True

        mask = Image.open("images/cat_seg.png").convert("L")
        mask = np.array(mask)
        for i, u in enumerate(np.unique(mask)):
            mask[mask == u] = i
        mask = torch.tensor(mask).to(device)

        image_emb = model.BLIPcap.forward_encoder({"image": image})[:, :-1, :]

        outputs = []
        losses = []
        for epoch in range(5):
            optimizer.zero_grad()
            parameters = parameter_selector(image_emb.reshape(24,24,768).permute(2,0,1).unsqueeze(0))

            output = model.clustering_model(image_emb, parameters=parameters, training=True)
            loss = loss_function(output, mask.flatten())
            loss.backward(retain_graph=True)
            optimizer.step()

            n_clusters, n_iters, compactness = parameters
            print(f"Epoch {epoch}: Loss {loss.item()}, compactness {compactness.item()}, n_clusters {n_clusters.item()}, n_iters {n_iters.item()}")

            hard_labels = output.argmax(2).unflatten(1, (24, 24))
            hard_labels = hard_labels.squeeze(0).cpu().detach().numpy()
            outputs.append(hard_labels)
            losses.append(loss.item())

        fig, ax = plt.subplots(1, len(outputs)+1, figsize=(5*len(outputs), 5))
        ax[0].imshow(mask.cpu().detach().numpy())
        ax[0].set_title("Ground Truth")
        for epoch, hard_labels in enumerate(outputs):
            ax[epoch+1].set_title("Epoch: " + str(epoch) + " Loss: " + str(round(losses[epoch], 3)))
            ax[epoch+1].imshow(hard_labels)
        plt.show()

    else:

        n_samples = 100
        batch_size = 1
        n_epochs = 4
        lr = 0.0001
        plot = True
        wandb_track = True
        optimizer = torch.optim.Adam(parameter_selector.parameters(), lr=lr)

        if wandb_track:
            run = wandb.init(
                # Set the project where this run will be logged
                project="maskblip",
                # Track hyperparameters and run metadata
                config={
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

        best_model = parameter_selector

        n_plots = 5
        plots = [[] for x in range(n_plots)]
        losses = [[] for x in range(n_plots)]
        embs = []
        masks = []
        imgs = []
        for iter, batch in enumerate(train_loader):
            images, annotations = batch
            images = images.to(device)
            mask = annotations.to(device)
            image_emb = model.BLIPcap.forward_encoder({"image": images})[:, :-1, :]
            embs.append(image_emb)
            masks.append(mask)
            imgs.append(images)
            if iter == n_plots-1:
                break


        # iterate over the data loader to process images in batches
        for epoch in range(n_epochs):
            for iter, batch in enumerate(train_loader):
                parameter_selector.train()

                images, annotations = batch
                images.requires_grad = True
                images = images.to(device)
                mask = annotations.to(device)

                image_emb = model.BLIPcap.forward_encoder({"image": images})[:, :-1, :]

                optimizer.zero_grad(set_to_none=True)
                parameters = parameter_selector(image_emb.reshape(-1, 24,24,768).permute(0, 3, 1, 2))
                output = model.clustering_model(image_emb, parameters=parameters, training=True)
                loss = loss_function(output, mask.flatten(1))
                loss.backward(retain_graph=True)
                optimizer.step()

                n_clusters, n_iters, compactness = parameters
                # hard_labels = output.argmax(2).unflatten(1, (24,24))
                # hard_labels = hard_labels.squeeze(0).cpu().detach().numpy()

                wandb.log({"loss": loss.item(), "avg_n_clusters": torch.mean(n_clusters.float()).item(),
                           "avg_n_iters": torch.mean(n_iters.float()).item(), "avg_compactness": torch.mean(compactness).item()})



                parameters = parameter_selector(image_emb.reshape(-1, 24, 24, 768).permute(0, 3, 1, 2))
                output = model.clustering_model(image_emb, parameters=parameters, training=True)


            avg_val_loss = 0
            best_val_loss = np.infty
            for iter, batch in enumerate(val_loader):
                parameter_selector.eval()

                images, annotations = batch
                images = images.to(device)
                mask = annotations.to(device)

                image_emb = model.BLIPcap.forward_encoder({"image": images})[:, :-1, :]

                parameters = parameter_selector(image_emb.reshape(-1, 24, 24, 768).permute(0, 3, 1, 2))
                output = model.clustering_model(image_emb, parameters=parameters, training=True)
                loss = loss_function(output, mask.flatten(1))
                avg_val_loss += loss.item()

                if loss.item() < best_val_loss:
                    best_model = parameter_selector
                    best_val_loss = loss.item()
            avg_val_loss /= len(val_loader)
            wandb.log({"val_loss": avg_val_loss})

            for i in range(n_plots):
                parameters = parameter_selector(embs[i].reshape(-1, 24, 24, 768).permute(0, 3, 1, 2))
                output = model.clustering_model(embs[i], parameters=parameters, training=True)
                loss = loss_function(output, mask.flatten(1))
                hard_labels = output[0].argmax(1).unflatten(0, (24, 24))
                hard_labels = hard_labels.squeeze(0).cpu().detach().numpy()
                plots[i].append(hard_labels)
                losses[i].append(loss.item())

        if plot:
            for i in range(n_plots):
                fig, ax = plt.subplots(1, n_epochs + 2, figsize=(5 * (n_epochs + 2), 5))
                ax[0].imshow(np.moveaxis(imgs[i].squeeze().cpu().detach().numpy(), 0, -1))
                ax[0].set_title("Image")
                ax[1].imshow(masks[i].squeeze().cpu().detach().numpy())
                ax[1].set_title("Ground Truth")
                for epoch, hard_labels in enumerate(plots[i]):
                    ax[epoch+2].set_title("Epoch: " + str(epoch) + " Loss: " + str(round(losses[i][epoch], 3)))
                    ax[epoch+2].imshow(hard_labels)
                plt.savefig(os.path.join("plots", f"result{i}.png"))

        torch.save(best_model.state_dict(), os.path.join("model_weights", "slic_param_selector", "param_selector.pt"))
