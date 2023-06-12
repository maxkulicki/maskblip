from lavis.models import load_model_and_preprocess
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from positional_encodings.torch_encodings import PositionalEncoding2D
import torch
import torch.nn as nn
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from crfseg import CRF
import cv2
from skimage.util import view_as_windows
from scipy.stats import mode
import torch_kmeans
import wandb


def print_cuda_memory():
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")


class MaskBLIP(torch.nn.Module):
    def __init__(self, device, scales=[384, 512], cluster_range=(2, 8), smoothness_weight=6, smoothness_theta=0.8, pos_emb_dim=256):
        super().__init__()
        model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption", "base_coco")
        #model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)

        model2, _, _ = load_model_and_preprocess(name="blip_feature_extractor", model_type="base",
                                                 is_eval=True, device=device)
        model.tokenizer = model2.tokenizer
        model.to(device)
        del (model2)

        self.device = device
        self.BLIPcap = model
        self.captioning = True
        self.prompt = self.init_prompt()
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors
        self.crf = CRF(n_spatial_dims=2, smoothness_weight=smoothness_weight, smoothness_theta=smoothness_theta)
        self.scales = scales
        self.img_size = max(self.scales)
        self.output_size = (max(self.scales) // 16, max(self.scales) // 16)
        self.cluster_range = cluster_range
        self.pos_emb_dim = pos_emb_dim
    def init_prompt(self):
        prompt = [self.BLIPcap.prompt]
        # prompt_text = "A one-word summary of this image: "
        # prompt = [prompt_text]
        prompt = self.BLIPcap.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt.input_ids[:, 0] = self.BLIPcap.tokenizer.bos_token_id
        prompt.input_ids = prompt.input_ids[:, :-1]
        return prompt

    def forward(self, raw_image, gt_mask=None, clean=False, attention_mode="global"):
        clusterings = []
        max_emb_size = max(self.scales) // 16
        if gt_mask is None:
            for img_size in self.scales:
                emb_size = img_size // 16

                p_enc_2d = PositionalEncoding2D(self.pos_emb_dim)

                self.BLIPcap.visual_encoder.pos_embed = nn.Parameter(
                    interpolate_pos_encoding(self.BLIPcap.visual_encoder.pos_embed, emb_size))
                self.BLIPcap.visual_encoder.patch_embed.img_size = (img_size, img_size)

                image = Resize(size=(img_size, img_size), antialias=True)(raw_image).to(self.device)
                embs = self.BLIPcap.forward_encoder({"image": image})[:, :-1, :]
                embs = embs.reshape(1, emb_size, emb_size, -1)
                p_enc = p_enc_2d(embs)
                embs = torch.cat([embs, p_enc], dim=-1)
                for n_clust in self.cluster_range:
                    kmeans = torch_kmeans.KMeans(n_clusters=n_clust, verbose=False)
                    result = kmeans(embs.flatten(1,2)).labels
                    result_np = result.reshape(emb_size, emb_size).cpu().numpy()
                    result_np = resize(result_np, (max_emb_size, max_emb_size))
                    clusterings.append(result_np)
                    del result, result_np
                    torch.cuda.empty_cache()

                del embs, image, p_enc, kmeans
                torch.cuda.empty_cache()
                #print_cuda_memory()

            aligned_clusterings = align_clusterings(clusterings)
            prob_map = create_probability_map(aligned_clusterings)
            final_clusters = self.crf(torch.tensor(np.transpose(prob_map, (2,0,1))).unsqueeze(0))
            final_clusters = torch.argmax(final_clusters, dim=1)
        else:
            final_clusters = torch.tensor(gt_mask).unsqueeze(0).to(self.device)
            final_clusters = Resize(size=self.output_size, antialias=True, interpolation=InterpolationMode.NEAREST_EXACT)(final_clusters)

        if clean:
            final_clusters = torch.tensor(clean_clusters(final_clusters.detach().cpu().numpy())).unsqueeze(0)

        if self.captioning:
            self.BLIPcap.visual_encoder.pos_embed = nn.Parameter(
                interpolate_pos_encoding(self.BLIPcap.visual_encoder.pos_embed, self.output_size[0]))
            captions_list = self.generate_captions(raw_image, final_clusters, attention_mode)
            return final_clusters, captions_list
        else:
            return final_clusters

    def generate_captions(self, image, clusters, attention_mode):
        image = Resize(size=(self.img_size, self.img_size), antialias=True)(image).to(self.device)

        image_emb = self.BLIPcap.forward_encoder({"image": image})[:, :-1, :]
        captions_list = []
        for idx, c in enumerate(clusters):
            captions=[]
            c = c.unsqueeze(0)
            # get flattened indices of each cluster
            cluster_indices = []

            for i in torch.unique(c):
                cluster_indices.append(torch.where(c.flatten() == i)[0].to(self.device))

            # slice image_emb using cluster indices
            cluster_embs = []

            if attention_mode in ["local", "concat", "cls"]:
                pre_attention = self.BLIPcap.visual_encoder.patch_embed(image)
                B = pre_attention.shape[0]
                encoder =  self.BLIPcap.visual_encoder
                for i in range(len(cluster_indices)):
                    register_blk = -1
                    x = torch.index_select(pre_attention, 1, cluster_indices[i])
                    cls_tokens = encoder.cls_token.expand(
                        B, -1, -1
                    )  # stole cls_tokens impl from Phil Wang, thanks
                    x = torch.cat((cls_tokens, x), dim=1)

                    x = x + encoder.pos_embed[:, : x.size(1), :]
                    x = encoder.pos_drop(x)

                    for j, blk in enumerate(encoder.blocks):
                        x = blk(x, register_blk == j)
                    x = encoder.norm(x).squeeze()
                    if attention_mode == "local":
                        cluster_embs.append(x)
                    elif attention_mode == "concat":
                        global_attention_emb = image_emb[idx].squeeze()[cluster_indices[i]]
                        x = torch.concat((x, global_attention_emb),0)
                        cluster_embs.append(x)
                    elif attention_mode == "cls":
                        cluster_embs.append(x[0])

            else:
                for i in range(len(cluster_indices)):
                    cluster_embs.append(image_emb[idx].squeeze()[cluster_indices[i]])

            for emb in cluster_embs:
                # emb = emb.mean(axis=0)
                decoder_out = self.BLIPcap.text_decoder.generate_from_encoder(
                    tokenized_prompt=self.prompt,
                    visual_embeds=emb.clone().detach().unsqueeze(0),
                    sep_token_id=self.BLIPcap.tokenizer.sep_token_id,
                    pad_token_id=self.BLIPcap.tokenizer.pad_token_id,
                    use_nucleus_sampling=True,
                    num_beams=5,
                    max_length=15,
                    min_length=3,
                    top_p=0.9,
                    repetition_penalty=3.0,
                )
                outputs = self.BLIPcap.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)
                caption = [output[len(self.BLIPcap.prompt):] for output in outputs]
                captions.append(caption[0])
            captions_list.append(captions)
        return captions_list


def majority_filter(image, size):
    # Create a sliding window view of the image
    shape = (size, size)
    windowed_image = view_as_windows(image, shape)

    # Compute the mode in each window
    modes, _ = mode(windowed_image.reshape(-1, size * size), axis=1)
    modes = modes.reshape(image.shape[0] - size + 1, image.shape[1] - size + 1)

    # Pad modes to match the original image size
    pad_width = size // 2
    modes = np.pad(modes, pad_width, mode='edge')

    return modes


def clean_clusters(image, footprint_size=3):
    # Apply majority filter recursively until convergence
    for i in range(5):
        new_image = majority_filter(image.squeeze(), footprint_size)
        mask = np.abs(image - new_image) > 1e-5  # Tolerance for floating point errors
        if not np.any(mask):
            break
        image = new_image
    print("Number of iterations: {}".format(i + 1))
    return image

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
    wandb_track = False

    img_path = "images/eiffel.jpg"
    raw_image = Image.open(img_path)
    transform = Compose([
        ToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    raw_image = transform(raw_image).unsqueeze(0)

    # Parameters from best parameter sweep
    scales = [384, 512]
    cluster_range = range(2,8)
    smoothness_theta = 0.8
    smoothness_weight = 6.26
    pos_emb_dim = 256
    cleanup = True
    device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if wandb_track:
        run = wandb.init(
            # Set the project where this run will be logged
            project="maskblip",
            group="multiscale",
            # Track hyperparameters and run metadata
            config={
                "scales": scales
            })
    else:
        run = wandb.init(mode = "disabled")


    model = MaskBLIP(device, scales=scales, cluster_range=cluster_range, smoothness_theta=smoothness_theta, smoothness_weight=smoothness_weight, pos_emb_dim=pos_emb_dim)

    print("model loaded")
    clusters, captions = model(raw_image, clean=cleanup)

    print(captions)

    unique_clusters = np.unique(clusters)
    cmap = plt.cm.get_cmap('tab20', len(unique_clusters))  # 'tab20' is a good colormap for categorical data
    # Create a plot with a colorbar that has labels
    fig, axs = plt.subplots(1, 2, figsize=(25, 7))  # 1 row, 2 columns
    # The first subplot will display your raw image
    cax = axs[0].imshow(clusters.squeeze())
    axs[0].set_title('Segmentation')
    # This creates a colorbar for the segmentation plot
    cbar = fig.colorbar(cax, ax=axs[0], ticks=unique_clusters, spacing='proportional')
    # This sets the labels of the colorbar to correspond to your captions
    cbar.ax.set_yticklabels(captions[0])  # change fontsize and rotation as necessary
    axs[1].imshow(raw_image.squeeze().permute(1, 2, 0))
    axs[1].set_title('Raw Image')
    # Show the plot
    plt.tight_layout()
    plt.show()
