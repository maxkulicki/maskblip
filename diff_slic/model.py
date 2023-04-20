import torch
import torch.nn as nn
from diff_slic.lib.ssn.ssn import ssn
from skimage.segmentation import slic
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import json

def conv_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )

class ClusteringModel(nn.Module):
    def __init__(self, device, n_clusters, n_iter, compactness, merging_threshold=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.compactness = compactness
        self.device = device
        self.merging_threshold = merging_threshold


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
        clusters = torch.unsqueeze(clusters, 2)
        for i in range(torch.max(clusters)):
            mask = np.broadcast_to(clusters == i, embs.shape)
            size = np.count_nonzero(mask)
            avg_embedding = np.mean(np.ma.masked_array(embs, mask), axis=(1, 2))
            cluster_embs.append(avg_embedding.squeeze())
            cluster_sizes.append(size)

        similarities = cosine_similarity(cluster_embs)
        for i in range(len(similarities)):
            similarities[i, i] = 0
        most_similar = np.unravel_index(similarities.argmax(), similarities.shape)

        merges = 0

        while similarities[most_similar] > self.merging_threshold:
            clusters[clusters == most_similar[1]] = most_similar[0]
            for i in range(most_similar[1], torch.max(clusters)):
                clusters[clusters == i + 1] = i

            cluster_embs[most_similar[0]] = cluster_embs[most_similar[0]] * cluster_sizes[most_similar[0]] \
                                            + cluster_embs[most_similar[1]] * cluster_sizes[most_similar[1]] \
                                            / (cluster_sizes[most_similar[0]] + cluster_sizes[
                most_similar[1]])  # weighted average
            cluster_sizes[most_similar[0]] += cluster_sizes[most_similar[1]]
            cluster_embs.pop(most_similar[1])
            cluster_sizes.pop(most_similar[1])
            merges += 1

            similarities = cosine_similarity(cluster_embs)
            for i in range(len(similarities)):
                similarities[i, i] = 0
            most_similar = np.unravel_index(similarities.argmax(), similarities.shape)

        print(f"Merged {merges} clusters")
        return torch.tensor(clusters)


class SSNClusteringModel(ClusteringModel):
    def __init__(self, device, n_clusters, n_iter, compactness, merging_threshold=None):
        super().__init__(device, n_clusters, n_iter, compactness)
        self.merging_threshold = merging_threshold
        self.init_data = json.load(open('centroid_data.dict', 'r'))
    def forward(self, image_emb, training=False, parameters=None):
        batch_size = image_emb.shape[0]
        if parameters is not None:
            n_clusters, n_iter, compactness = parameters
        else:
            n_clusters = torch.full((batch_size,), self.n_clusters).to(self.device)
            n_iter = torch.full((batch_size,), self.n_iter).to(self.device)
            compactness = torch.full((batch_size,), self.compactness).to(self.device)

        x_pos = torch.linspace(0, 23, 24).expand(24, 24).flatten().to(
            self.device) * compactness.view(-1,1)
        y_pos = torch.linspace(0, 23, 24).expand(24, 24).T.flatten().to(
            self.device) * compactness.view(-1,1)
        image_emb_with_spatial = torch.cat(
            (x_pos.unsqueeze(2), y_pos.unsqueeze(2), image_emb), dim=2)
        grid_image_emb = image_emb_with_spatial.unflatten(1, (24, 24))
        grid_image_emb[:, 0, :, :] = grid_image_emb[:, 0, :, :]
        grid_image_emb[:, 1, :, :] = grid_image_emb[:, 1, :, :]
        clusters = ssn(grid_image_emb, n_clusters, n_iter, self.init_data, training=training)

        # clusters = torch.squeeze(clusters)
        # if self.merging_threshold is not None:
        #     clusters = self.merge_clusters(clusters, grid_image_emb.squeeze().cpu().detach().numpy())
        # if batch_size == 1:
        #     clusters = clusters.unsqueeze(0)
        return clusters

class SlicClusteringModel(ClusteringModel):
    def __init__(self, device, n_clusters=4, n_iter=6, compactness=0.01, merging_threshold=None):
        super().__init__(device, n_clusters, n_iter, compactness)
        self.refine = False
        self.merging_threshold = merging_threshold
    def forward(self, image_emb, raw_image=None):
        grid_image_emb = image_emb.unflatten(1, (24, 24)).squeeze().cpu().detach().numpy()
        clusters = torch.tensor(slic(grid_image_emb, n_segments=self.n_clusters, compactness=self.compactness, sigma=0, channel_axis=2,
                             enforce_connectivity=False, max_num_iter=self.n_iter))

        if self.merging_threshold is not None:
            clusters = self.merge_clusters(clusters, grid_image_emb)

        if self.refine:
            clusters = self.refine_clusters(clusters, np.asarray(raw_image), n_segments=150, compactness=20, sigma=5)

        return clusters

