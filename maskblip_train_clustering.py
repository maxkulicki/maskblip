import os
import sys

from matplotlib import pyplot as plt

from maskblip import maskblip_segmentation
from maskblip_diff import MaskBLIP
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from lavis.models import load_model_and_preprocess
from sklearn.metrics import adjusted_rand_score
import torch.nn as nn

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
        x = x.view(-1, 64 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def consistency_index(soft_partition, hard_partition):
    """
    Computes the consistency index between a soft partition and a hard partition.

    Parameters:
        soft_partition (torch.Tensor): The soft partition tensor of shape (n_points, n_clusters).
        hard_partition (torch.Tensor): The hard partition tensor of shape (n_points,).

    Returns:
        float: The consistency index between the two partitions.
    """
    n_clusters = soft_partition.shape[1]

    # Compute the matrix of pairwise overlaps between clusters
    overlap_matrix = torch.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            overlap_matrix[i, j] = torch.sum(torch.min(soft_partition[:, i], (hard_partition == j).float()))

    # Compute the numerator and denominator of the consistency index
    num = torch.max(torch.sum(overlap_matrix, dim=1))
    den = torch.sum((hard_partition >= 0).float())

    # Compute and return the consistency index
    return num / den

def synset_match(nouns, synset):
    for noun in nouns:
        for syn in synset:
            if noun.strip() in syn.split(" "):
                return True
    return False
def get_recall_precision(captions, labels):
    matches = 0
    for label in labels:
        if any(label in caption for caption in captions):
                matches += 1

    if len(labels) == 0:
        recall = 1
        precision = 1
    else:
        recall = matches / len(labels)
        precision = matches / len(captions)
    return recall, precision

def get_ade20k_label_mapping(mapping_path):
    id_to_label = {}
    label_to_id = {}
    with open(mapping_path, encoding="utf8") as f:
        for line in f:
            id = line.split()[0]
            label = line.split()[4].replace(",", "")
            id_to_label[id] = label
            label_to_id[label] = id

    return id_to_label, label_to_id



def adjusted_rand_score_torch(output, target):
    """
    Computes the adjusted Rand score between two clusterings.
    Args:
        output: PyTorch tensor of shape (N,) containing the predicted cluster labels.
        target: PyTorch tensor of shape (N,) containing the ground truth cluster labels.
    Returns:
        A PyTorch tensor containing the adjusted Rand score.
    """
    assert output.shape == target.shape, "Shapes of output and target must match."
    n_samples = output.shape[0]

    # Compute the contingency table
    output_eq = (output[:, None] == output[None, :]).float()
    target_eq = (target[:, None] == target[None, :]).float()
    contingency_table = (output_eq * target_eq)

    total_sum = contingency_table.sum()

    # Compute the adjusted Rand score
    A = contingency_table.sum(dim=1).sum(dim=0)
    B = contingency_table.square().sum(dim=1).sum(dim=0) - A
    C = contingency_table.square().sum(dim=0).sum(dim=0) - A
    D = total_sum**2 - A - B - C
    numerator = A * D - B * C
    denominator = (A + B) * (A + C) * (C + D) * (B + D)
    return numerator / torch.sqrt(denominator)

def loss_function(output, target):
    return -consistency_index(output, target)

if __name__ == "__main__":

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model, vis_processors, _ = load_model_and_preprocess("blip_caption", "base_coco")
    model.to(device)

    image = Image.open("images/cat.jpg")
    image = vis_processors["eval"](image).unsqueeze(0).to(device)
    image.requires_grad = True

    mask = Image.open("images/cat_seg.png").convert("L")
    mask = np.array(mask)
    for i, u in enumerate(np.unique(mask)):
        mask[mask == u] = i
    mask = torch.tensor(mask).to(device)

    image_emb = model.forward_encoder({"image": image})[:, :-1, :]
    image_emb = image_emb.reshape(24,24,768).permute(2,0,1).unsqueeze(0)

    parameter_selector = SLICParameterSelector().to(device)

    optimizer = torch.optim.SGD(parameter_selector.parameters(), lr=1)

    model = MaskBLIP(model, device)

    # Train the model
    for epoch in range(10):
        optimizer.zero_grad()
        params = parameter_selector(image_emb)
        print(params.data)
        #map to set {4,9,16,25,36}
        n_clusters = 4 + torch.square((torch.sigmoid(params[:,0])*4-2).int())
        #map to int range (3,10)
        n_iters = 3 + (torch.sigmoid(params[:,1])*7).int()
        #map to range (0,0.1)
        compactness = torch.sigmoid(params[:,2]) * 0.1

        parameters = (n_clusters, n_iters, compactness)

        # model.clustering_model.n_clusters = n_clusters.item()
        # model.clustering_model.compactness = compactness.item()
        # model.clustering_model.n_iter = n_iters.item()

       #output, _ = model(image)
        # Compute the output of the clustering algorithm

        output = model.clustering_model(image_emb.flatten(2).permute(0,2,1), parameters=parameters, training=True)

        # Compute the loss
        loss = loss_function(output, mask.flatten())


        # Compute the gradient
        loss.backward(retain_graph=True)

        # Update the parameters
        optimizer.step()
        # for name, param in parameter_selector.named_parameters():
        #     print(name, param.grad)

        # Print the loss and the updated parameters
        print(f"Epoch {epoch}: Loss {loss.item()}, compactness {compactness.item()}, n_clusters {n_clusters.item()}, n_iters {n_iters.item()}")

        hard_labels = output.argmax(1).unflatten(0, (24, 24))
        hard_labels = hard_labels.squeeze(0).cpu().detach().numpy()
        plt.title("Epoch: " + str(epoch) + " Loss: " + str(loss.item()))
        plt.imshow(hard_labels)
        plt.show()