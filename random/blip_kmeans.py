from sklearn.cluster import KMeans
from random.maskblip_diff import MaskBLIP
from lavis.models import load_model_and_preprocess
import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def blip_kmeans(model, img_dir, vis_processors, n_images=100, n_clusters=20, max_iter=100, tol=1e-6, n_plots=5):
    image_paths = [os.path.join(img_dir, path) for path in os.listdir(img_dir)]
    all_embeddings = []
    for image_path in image_paths[:n_images]:
        image = Image.open(image_path)
        image = vis_processors['eval'](image)
        image = image.unsqueeze(0)
        image = image.to(device)
        embeddings = model.BLIPcap.forward_encoder({"image": image})[:, :-1, :]
        all_embeddings.append(embeddings.squeeze().detach().cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings)
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol, init="k-means++").fit(all_embeddings)
    print(kmeans.n_iter_)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    clustered_images = labels.reshape(n_images, 24, 24)
    for i, img in enumerate(clustered_images):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[0].title.set_text(str(np.unique(img)))
        ax[1].imshow(Image.open(image_paths[i]))
        plt.show()
        if i == n_plots:
            break
    return labels, centers

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption", "base_coco")
    model = MaskBLIP(model, device, upsampler=True)
    image_dir = "../datasets/VOC2012/JPEGImages"
    blip_kmeans(model, image_dir, vis_processors)