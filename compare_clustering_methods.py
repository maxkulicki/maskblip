import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from skimage.segmentation import slic
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import OPTICS, AffinityPropagation, KMeans, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, Birch
from itertools import product
def merge_clusters(clusters, embs, threshold=0.995):
    cluster_sizes = []
    cluster_embs = []
    clusters = np.expand_dims(clusters, 2)
    for i in range(np.max(clusters)):
        mask = np.broadcast_to(clusters == i, embs.shape)
        size = np.count_nonzero(mask)
        avg_embedding = np.mean(np.ma.masked_array(embs, mask), axis=(1, 2))
        cluster_embs.append(avg_embedding)
        cluster_sizes.append(size)

        avg_embedding = torch.tensor(avg_embedding).float().to(device)


    similarities = cosine_similarity(cluster_embs)
    for i in range(len(similarities)):
        similarities[i, i] = 0
    most_similar = np.unravel_index(similarities.argmax(), similarities.shape)

    while similarities[most_similar] > threshold:
        print("Similarity: ", similarities[most_similar])
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

    print("Merging done")
    return clusters


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

raw_image = Image.open("images/img3.jpg").convert("RGB")

# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# generate caption
image_emb = model.forward_encoder({"image": image})[:, :-1, :]
grid_image_emb = image_emb.unflatten(1, (24,24)).squeeze().detach().cpu().numpy()

#compactness = 15


#clustering_methods = ["slic", "kmeans", "agglomerative", "birch", "optics"]
distance_thresholds = [120, 140, 160]
spatial_importances = [0, 3, 5]
clustering_methods = list(product(distance_thresholds, spatial_importances))
results = {}
for method in clustering_methods:
    # if method =="slic":
    #     clusters = slic(grid_image_emb, n_clusters=10, compactness=0.001, sigma=1, channel_axis=2, enforce_connectivity=False)
    #     clusters = merge_clusters(clusters, grid_image_emb, threshold=0.999)
    # elif method == "dbscan":
    #     clusters = DBSCAN(eps=10, min_samples=40).fit_predict(image_emb).reshape(24, 24)
    # elif method == "affinity":
    #     clusters = AffinityPropagation().fit_predict(image_emb).reshape(24, 24)
    # elif method == "kmeans":
    #     clusters = KMeans(n_clusters=5, random_state=0).fit_predict(image_emb).reshape(24, 24)
    # elif method == "meanshift":
    #     clusters = MeanShift().fit_predict(image_emb).reshape(24, 24)
    # elif method == "spectral":
    #     clusters = SpectralClustering(n_clusters=5, random_state=0).fit_predict(image_emb).reshape(24, 24)
    # elif method == "agglomerative":
    #     clusters = AgglomerativeClustering(n_clusters=5).fit_predict(image_emb).reshape(24, 24)
    # elif method == "birch":
    #     clusters = Birch(n_clusters=3).fit_predict(image_emb).reshape(24, 24)
    #     #clusters = merge_clusters(clusters, grid_image_emb, threshold=0.9995)
    # elif method == "optics":
    #     clusters = OPTICS().fit_predict(image_emb).reshape(24, 24)
    dist, spatial_importance = method

    x_pos = torch.linspace(-1, 1, 24).expand(24, 24).flatten() * spatial_importance
    y_pos = torch.linspace(-1, 1, 24).expand(24, 24).T.flatten() * spatial_importance
    image_emb_with_spatial = torch.cat((image_emb, x_pos.unsqueeze(1).unsqueeze(0), y_pos.unsqueeze(1).unsqueeze(0)), dim=2)

    image_emb_with_spatial = image_emb_with_spatial.squeeze().detach().numpy()
    clusters = AgglomerativeClustering(n_clusters=None, distance_threshold=dist).fit_predict(image_emb_with_spatial).reshape(24, 24)
    #clusters = merge_clusters(clusters, grid_image_emb, threshold=0.9998)
    #get flattened indices of each cluster
    cluster_indices = []
    for i in range(clusters.min(), clusters.max() + 1):
        cluster_indices.append(np.where(clusters.flatten() == i))

    results[method] = clusters

fig, axs = plt.subplots(2,5)
implot = axs[0, 0].imshow(raw_image)
axs[0, 0].set_title("Original Image")
for i, method in enumerate(clustering_methods):
    axs[(i+1)//5, (i+1)%5].imshow(results[method])
    axs[(i+1)//5, (i+1)%5].set_title(method)

plt.show()



