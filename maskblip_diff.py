from lavis.models import load_model_and_preprocess
import torch
from diff_slic.model import SSNClusteringModel, SlicClusteringModel
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import torch.nn.functional as F
class MaskBLIP(torch.nn.Module):
    def __init__(self, blip_model, device, use_ssn=True,  n_clusters=4, n_iter=3, compactness=0.001, merging_threshold=None):
        super().__init__()
        self.device = device
        self.BLIPcap = blip_model.to(device)
        if use_ssn:
            self.clustering_model = SSNClusteringModel(device, n_clusters=n_clusters, n_iter=n_iter, compactness=compactness, merging_threshold=merging_threshold)
        else:
            self.clustering_model = SlicClusteringModel(device, n_clusters=n_clusters, n_iter=n_iter, compactness=compactness, merging_threshold=merging_threshold)
        self.prompt = self.init_prompt()

        #self.captioning_adapter = torch.nn.Linear(768, 768)


    def init_prompt(self):
        prompt = [self.BLIPcap.prompt]
        # prompt_text = "A one-word summary of this image: "
        # prompt = [prompt_text]
        prompt = self.BLIPcap.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt.input_ids[:, 0] = self.BLIPcap.tokenizer.bos_token_id
        prompt.input_ids = prompt.input_ids[:, :-1]
        return prompt
    def forward(self, image):
        captions = []
        image_emb = self.BLIPcap.forward_encoder({"image": image})[:, :-1, :]
        clusters = self.clustering_model(image_emb)
        clusters.to(self.device)
        # get flattened indices of each cluster
        cluster_indices = []
        for i in range(clusters.min(), clusters.max() + 1):
            cluster_indices.append(torch.where(clusters.flatten() == i))

        # slice image_emb using cluster indices
        cluster_embs = []
        for i in range(len(cluster_indices)):
            cluster_embs.append(image_emb.squeeze()[cluster_indices[i]])

        for emb in cluster_embs:
            #emb = self.captioning_adapter(emb)
            # generate caption for each cluster
            decoder_out = self.BLIPcap.text_decoder.generate_from_encoder(
                tokenized_prompt=self.prompt,
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

        return clusters.squeeze(), captions

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








