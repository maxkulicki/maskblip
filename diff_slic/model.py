import torch
import torch.nn as nn
from diff_slic.lib.ssn.ssn import ssn
from skimage.segmentation import slic
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image

def conv_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )

class SSNClusteringModel(nn.Module):
    def __init__(self, device, n_clusters=9, n_iter=10, compactness=3):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.spatial_importance = compactness
        self.device = device
    def forward(self, image_emb):
        batch_size = image_emb.shape[0]
        x_pos = torch.linspace(0, 23, 24).expand(24, 24).flatten().to(
            self.device) * self.spatial_importance
        y_pos = torch.linspace(0, 23, 24).expand(24, 24).T.flatten().to(
            self.device) * self.spatial_importance
        image_emb_with_spatial = torch.cat(
            (x_pos.repeat(batch_size, 1).unsqueeze(2), y_pos.repeat(batch_size, 1).unsqueeze(2), image_emb), dim=2)
        grid_image_emb = image_emb_with_spatial.unflatten(1, (24, 24))
        grid_image_emb[:, 0, :, :] = grid_image_emb[:, 0, :, :]
        grid_image_emb[:, 1, :, :] = grid_image_emb[:, 1, :, :]
        return ssn(grid_image_emb, self.n_clusters, self.n_iter)

class SlicClusteringModel(nn.Module):
    def __init__(self, n_clusters=9, n_iter=10, threshold=0.9995, compactness=3, refine=False):
        super().__init__()
        self.n_segments = n_clusters
        self.threshold = threshold
        self.refine = refine
        self.n_iter = n_iter
        self.compactness = compactness
    def forward(self, image_emb, raw_image=None):
        grid_image_emb = image_emb.unflatten(1, (24, 24)).squeeze().cpu().detach().numpy()
        clusters = slic(grid_image_emb, n_segments=self.n_segments, compactness=self.compactness, sigma=0, channel_axis=2,
                             enforce_connectivity=False, max_num_iter=self.n_iter)
        #clusters = self.merge_clusters(clusters, grid_image_emb)
        if self.refine:
            clusters = self.refine_clusters(clusters, np.asarray(raw_image), n_segments=150, compactness=20, sigma=5)

        return torch.tensor(clusters)

    def refine_clusters(blip_clusters, image, n_segments=150, compactness=10, sigma=1):
        image_slic = np.asarray(slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma))
        new_clusters = np.zeros_like(image_slic)
        shape = (image_slic.shape[1], image_slic.shape[0])
        clustered_img = np.asarray(
            Image.fromarray(blip_clusters.squeeze().astype(np.uint8)).resize(shape, Image.NEAREST))
        for superpixel in np.unique(image_slic):
            superpixel_mask = image_slic == superpixel
            superpixel_label = np.argmax(np.bincount(clustered_img[superpixel_mask]))
            new_clusters[superpixel_mask] = superpixel_label

        return new_clusters

    def merge_clusters(self, clusters, embs):
        cluster_sizes = []
        cluster_embs = []
        clusters = np.expand_dims(clusters, 2)
        for i in range(np.max(clusters)):
            mask = np.broadcast_to(clusters == i, embs.shape)
            size = np.count_nonzero(mask)
            avg_embedding = np.mean(np.ma.masked_array(embs, mask), axis=(1, 2))
            cluster_embs.append(avg_embedding.squeeze())
            cluster_sizes.append(size)

            avg_embedding = torch.tensor(avg_embedding).float()

        similarities = cosine_similarity(cluster_embs)
        for i in range(len(similarities)):
            similarities[i, i] = 0
        most_similar = np.unravel_index(similarities.argmax(), similarities.shape)

        while similarities[most_similar] > self.threshold:
            clusters[clusters == most_similar[1]] = most_similar[0]
            for i in range(most_similar[1], np.max(clusters)):
                clusters[clusters == i + 1] = i

            cluster_embs[most_similar[0]] = cluster_embs[most_similar[0]] * cluster_sizes[most_similar[0]] \
                                            + cluster_embs[most_similar[1]] * cluster_sizes[most_similar[1]] \
                                            / (cluster_sizes[most_similar[0]] + cluster_sizes[
                most_similar[1]])  # weighted average
            cluster_sizes[most_similar[0]] += cluster_sizes[most_similar[1]]
            cluster_embs.pop(most_similar[1])
            cluster_sizes.pop(most_similar[1])

            similarities = cosine_similarity(cluster_embs)
            for i in range(len(similarities)):
                similarities[i, i] = 0
            most_similar = np.unravel_index(similarities.argmax(), similarities.shape)

        return torch.tensor(clusters)
