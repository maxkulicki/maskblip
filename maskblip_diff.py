from lavis.models import load_model_and_preprocess
import torch
import torch.nn as nn
from diff_slic.model import SSNClusteringModel, SlicClusteringModel
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import torch.nn.functional as F
from lavis.models.vit import Attention
from prompt_learner import CoOp

class CaptionAdapter(nn.Module):
    def __init__(self, device, embed_dim=768):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.layer1 = nn.Linear(embed_dim, embed_dim)
        self.layer1.to(device)
        self.layer2 = nn.Linear(embed_dim, embed_dim)
        self.layer2.to(device)
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

class ClusteringAdapter(nn.Module):
    def __init__(self, device, embed_dim=768):
        super(ClusteringAdapter, self).__init__()
        self.adapter = Attention(embed_dim+2, num_heads=7)
        self.adapter.to(device)
        self.batch_norm = nn.BatchNorm1d(576)
        self.batch_norm.to(device)
        self.device = device
    def forward(self, x):
        batch_size = x.shape[0]
        x_pos = torch.linspace(0, 23, 24).expand(24, 24).flatten().unsqueeze(0).unsqueeze(2).repeat(batch_size,1,1).to(self.device)
        y_pos = torch.linspace(0, 23, 24).expand(24, 24).T.flatten().unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1).to(self.device)
        x = torch.cat((x_pos, y_pos, x), dim=2)
        x = self.adapter(x)
        x = self.batch_norm(x)
        return x

class SLICParameterSelector(nn.Module):
    def __init__(self, input_size=768):
        super(SLICParameterSelector, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=3, stride=1, padding=1)
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
        # map to range (2,7)
        n_clusters = ((torch.sigmoid(x[:, 0])-0.5) * 6).int() + 5
        # map to int range (3,10)
        n_iters = 3 + (torch.sigmoid(x[:, 1]) * 7).int()
        # map to range (0,0.1)
        compactness = torch.sigmoid(x[:, 2]) * 0.01 #torch.sigmoid(x[:, 2]) * 0 + 0.00001

        return (n_clusters, n_iters, compactness)


class MaskBLIP(torch.nn.Module):
    def __init__(self, blip_model, device, text_encoder=None, use_ssn=True,  n_clusters=4, n_iter=3, compactness=0.001,
                 merging_threshold=None, captioning_adapter=False, clustering_adapter=False, parameter_selector=False):
        super().__init__()
        self.device = device
        self.BLIPcap = blip_model.to(device)
        self.captioning = True
        self.text_encoder = text_encoder
        if use_ssn:
            self.clustering_model = SSNClusteringModel(device, n_clusters=n_clusters, n_iter=n_iter, compactness=compactness, merging_threshold=merging_threshold)
        else:
            self.clustering_model = SlicClusteringModel(device, n_clusters=n_clusters, n_iter=n_iter, compactness=compactness, merging_threshold=merging_threshold)
        self.prompt = self.init_prompt()
        self.prompt_module = CoOp(768, self.BLIPcap.tokenizer, self.text_encoder, device)
        if captioning_adapter:
            self.captioning_adapter = CaptionAdapter(device)
        else:
            self.captioning_adapter = None
        if clustering_adapter:
            self.clustering_adapter = ClusteringAdapter(device)
        else:
            self.clustering_adapter = None
        if parameter_selector:
            self.parameter_selector = SLICParameterSelector(device)
        else:
            self.parameter_selector = None

    def init_prompt(self):
        prompt = [self.BLIPcap.prompt]
        # prompt_text = "A one-word summary of this image: "
        # prompt = [prompt_text]
        prompt = self.BLIPcap.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt.input_ids[:, 0] = self.BLIPcap.tokenizer.bos_token_id
        prompt.input_ids = prompt.input_ids[:, :-1]
        return prompt
    def forward_skip_clustering(self, image, mask):
        captions = []
        image_emb = self.BLIPcap.forward_encoder({"image": image})[:, :-1, :]
        clusters = mask.clone()
        unique_values = torch.unique(clusters)
        new_values = torch.arange(unique_values.shape[0])
        for i in range(unique_values.shape[0]):
            clusters[clusters == unique_values[i]] = new_values[i]

        # get flattened indices of each cluster
        cluster_indices = []
        for i in range(len(clusters.unique())):
            cluster_indices.append(torch.where(clusters.flatten() == clusters.unique()[i]))

        # slice image_emb using cluster indices
        cluster_embs = []
        for i in range(len(cluster_indices)):
            cluster_embs.append(image_emb.squeeze()[cluster_indices[i]])

        for emb in cluster_embs:
            #emb = emb.mean(axis=0).unsqueeze(0)
            if self.captioning_adapter is not None:
                emb = emb + self.captioning_adapter(emb)

            # generate caption for each cluster
            decoder_out = self.BLIPcap.text_decoder.generate_from_encoder(
                tokenized_prompt=self.prompt,#_module(emb),
                visual_embeds=emb.clone().detach().unsqueeze(0),
                sep_token_id=self.BLIPcap.tokenizer.sep_token_id,
                pad_token_id=self.BLIPcap.tokenizer.pad_token_id,
                use_nucleus_sampling=False,
                num_beams=3,
                max_length=15,
                min_length=3,
                top_p=0.9,
                repetition_penalty=1.0,
            )
            outputs = self.BLIPcap.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)
            caption = [output[len(self.BLIPcap.prompt):] for output in outputs]
            captions.append(caption[0])

        return captions

    def forward(self, image):
        captions = []
        image_emb = self.BLIPcap.forward_encoder({"image": image})[:, :-1, :]
        if self.clustering_adapter is not None:
            image_emb = image_emb + self.clustering_adapter(image_emb)[:,:,2:]
        if self.parameter_selector is not None:
            params = self.parameter_selector(image_emb)
        else:
            params = None
        clusters = self.clustering_model(image_emb, parameters=params)
        clusters = torch.stack(clusters, dim=0)
        #clusters.to(self.device)

        if self.captioning:
            # get flattened indices of each cluster
            cluster_indices = []
            for i in range(clusters.min(), clusters.max() + 1):
                cluster_indices.append(torch.where(clusters.flatten() == i))

            # slice image_emb using cluster indices
            cluster_embs = []
            for i in range(len(cluster_indices)):
                cluster_embs.append(image_emb.squeeze()[cluster_indices[i]])

            for emb in cluster_embs:
                emb = emb.mean(axis=0)
                if self.captioning_adapter is not None:
                    emb = emb + self.captioning_adapter(emb)
                # generate caption for each cluster
                decoder_out = self.BLIPcap.text_decoder.generate_from_encoder(
                    tokenized_prompt=self.prompt_module(emb),
                    visual_embeds=emb.clone().detach().unsqueeze(0),
                    sep_token_id=self.BLIPcap.tokenizer.sep_token_id,
                    pad_token_id=self.BLIPcap.tokenizer.pad_token_id,
                    use_nucleus_sampling=True,
                    num_beams=3,
                    max_length=15,
                    min_length=3,
                    top_p=0.9,
                    repetition_penalty=1.0,
                )
                outputs = self.BLIPcap.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)
                caption = [output[len(self.BLIPcap.prompt):] for output in outputs]
                captions.append(caption[0])

            return clusters.squeeze(), captions
        else:
            return clusters.squeeze()

if __name__ == "__main__":
    img_path = "images/img1.jpg"
    image = Image.open(img_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vis_processors, _ = load_model_and_preprocess("blip_caption", "base_coco")
    image = vis_processors["eval"](image).unsqueeze(0).to(device)

    compactness = 0.01
    n_clusters = 3
    n_iter = 6
    merging_threshold = None

    model = MaskBLIP(model, device, n_clusters=n_clusters, n_iter=n_iter, compactness=compactness, merging_threshold=merging_threshold)
    clusters, captions = model.forward(image)
    print(captions)
    model.clustering_model = SSNClusteringModel(device, n_clusters=n_clusters, n_iter=n_iter, compactness=compactness, merging_threshold=merging_threshold)
    soft_clusters, soft_captions = model.forward(image)
    print(soft_captions)

    fig, ax = plt.subplots(1, 2)
    fig.suptitle("compactness: {}, n_clusters: {}, n_iter: {}".format(compactness, n_clusters, n_iter))
    ax[0].imshow(clusters)
    ax[0].title.set_text("SLIC Clusters")
    ax[1].imshow(soft_clusters.cpu().detach().numpy())
    ax[1].title.set_text("Soft SLIC Clusters")
    plt.show()



    # spatial_importances = [1,3,5,7]
    # n_iters = [1,2,3]
    # for i in n_iters:
    #     for j in spatial_importances:
    #         model.clustering_model = SSNClusteringModel(4, i, j).to(device)
    #         model.forward(image)








