from lavis.models import load_model_and_preprocess
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from finch import FINCH
from positional_encodings.torch_encodings import PositionalEncoding2D
import torch
import torch.nn as nn
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from crfseg import CRF
import cv2


def resize(clusters, new_shape):
    # OpenCV uses width x height instead of height x width, so reverse the dimensions
    new_shape_cv = (new_shape[1], new_shape[0])
    # Use cv2's resize function. Note that the interpolation method should be cv2.INTER_NEAREST for nearest-neighbor interpolation
    resized_clusters = cv2.resize(clusters, new_shape_cv, interpolation=cv2.INTER_NEAREST)
    return resized_clusters

def compute_cost(clustering1, clustering2):
    return np.sum(clustering1 != clustering2)

def align_clusterings(clusterings):
    # Find the reference clustering (the one with the most unique clusters)
    ref_clustering_idx = np.argmax([len(np.unique(clustering)) for clustering in clusterings])
    ref_clustering = clusterings[ref_clustering_idx]

    # Align each clustering to the reference clustering
    aligned_clusterings = []
    for i, clustering in enumerate(clusterings):
        if i == ref_clustering_idx:
            aligned_clusterings.append(clustering)  # No need to align the reference clustering
            continue

        # Compute the cost matrix
        unique_clusters_ref = np.unique(ref_clustering)
        unique_clusters = np.unique(clustering)
        cost_matrix = np.zeros((len(unique_clusters_ref), len(unique_clusters)))
        for i, label1 in enumerate(unique_clusters_ref):
            for j, label2 in enumerate(unique_clusters):
                cost_matrix[i, j] = compute_cost(clustering == label2, ref_clustering == label1)

        # Apply the Hungarian algorithm to find the best alignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create the aligned clustering
        aligned_clustering = clustering.copy()
        for old_label, new_label in zip(unique_clusters[col_ind], unique_clusters_ref[row_ind]):
            aligned_clustering[clustering == old_label] = new_label

        aligned_clusterings.append(aligned_clustering)

    return aligned_clusterings


# def compute_cost(cluster1, cluster2):
#     centroid1 = np.mean(cluster1, axis=0)
#     centroid2 = np.mean(cluster2, axis=0)
#     return np.linalg.norm(centroid1 - centroid2)
#
# def align_clusterings(clusterings):
#     # Find the reference clustering (the one with the most clusters)
#     ref_clustering_idx = np.argmax([len(clustering) for clustering in clusterings])
#     ref_clustering = clusterings[ref_clustering_idx]
#
#     # Align each clustering to the reference clustering
#     aligned_clusterings = []
#     for i, clustering in enumerate(clusterings):
#         if i == ref_clustering_idx:
#             aligned_clusterings.append(clustering)  # No need to align the reference clustering
#             continue
#
#         # Compute the cost matrix
#         cost_matrix = np.zeros((len(ref_clustering), len(clustering)))
#         for i, ref_cluster in enumerate(ref_clustering):
#             for j, cluster in enumerate(clustering):
#                 cost_matrix[i, j] = compute_cost(ref_cluster, cluster)
#
#         # Apply the Hungarian algorithm to find the best alignment
#         row_ind, col_ind = linear_sum_assignment(cost_matrix)
#
#         # Create the aligned clustering
#         aligned_clustering = [None]*len(clustering)
#         for old_idx, new_idx in enumerate(col_ind):
#             aligned_clustering[new_idx] = clustering[old_idx]
#
#         aligned_clusterings.append(aligned_clustering)
#
#     return aligned_clusterings

def interpolate_pos_encoding(pos_embed, emb_size):
    # Assuming pos_embed is of shape (1, npatch + 1, dim)
    npatch = pos_embed.shape[1] - 1
    N = npatch
    dim = pos_embed.shape[-1]

    # New dimensions
    w = emb_size
    h = emb_size

    if npatch == w * h:
        return pos_embed

    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]

    w0 = w
    h0 = h
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        size=(int(w0), int(h0)),
        mode='bicubic',
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

def create_probability_map(clusterings, epsilon=1e-6):
    num_clusters = max([np.max(clustering) for clustering in clusterings]) + 1
    prob_map = np.zeros(list(clusterings[0].shape) + [num_clusters])

    for clustering in clusterings:
        for label in range(num_clusters):
            prob_map[:,:,label] += (clustering == label)

    prob_map /= len(clusterings)
    prob_map += epsilon

    return prob_map / np.sum(prob_map, axis=-1, keepdims=True)


if __name__ == "__main__":
    img_path = "images/batman.jpg"
    raw_image = Image.open(img_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption", "base_coco")
    print("model loaded")
    clusterings = []
    img_sizes = [384, 448, 512, 576, 640]
    max_emb_size = max(img_sizes) // 16
    for img_size in img_sizes:
        emb_size = img_size // 16

        p_enc_2d = PositionalEncoding2D(700)

        model.visual_encoder.pos_embed = nn.Parameter(interpolate_pos_encoding(model.visual_encoder.pos_embed, emb_size))
        preprocess = Compose([
            Resize(size=(img_size, img_size)),  # replace NEW_WIDTH, NEW_HEIGHT with desired dimensions
            ToTensor(),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        image = preprocess(raw_image).unsqueeze(0).to(device)
        embs = model.forward_encoder({"image": image})[:, :-1, :]
        embs = embs.reshape(1, emb_size, emb_size, -1)
        p_enc = p_enc_2d(embs)
        embs = torch.cat([embs, p_enc], dim=-1)
        c, num_clust, req_c = FINCH(embs.view(-1, embs.shape[-1]).cpu().detach().numpy())

        clusters = c[:,-1].reshape(emb_size, emb_size)

        clusterings.append(resize(clusters, (max_emb_size, max_emb_size)))
    aligned_clusterings = align_clusterings(clusterings)
    print("clusterings aligned")

    for img_size, clusters in zip(img_sizes, aligned_clusterings):
        plt.imshow(clusters)
        plt.title(f"Image size: {img_size}, aligned")
        plt.show()

    prob_map = create_probability_map(aligned_clusterings)
    print("probability map created")
    crf_model = CRF(n_spatial_dims=2)
    output = crf_model(torch.tensor(np.transpose(prob_map, (2,0,1))).unsqueeze(0))
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(raw_image)
    ax[0].title.set_text("Original image")
    ax[1].imshow(torch.argmax(output.squeeze(),0).numpy())
    ax[1].title.set_text("CRF applied")
    plt.show()










