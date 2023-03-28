from lavis.models import load_model_and_preprocess
import torch
from diff_slic.model import SSNModel
from PIL import Image
from matplotlib import pyplot as plt
class MaskBLIP(torch.nn.Module):
    def __init__(self, blip_model, device):
        super().__init__()
        self.device = device
        self.BLIPcap = blip_model.to(device)
        self.clustering_model = SSNModel().to(device)
        self.prompt = self.init_prompt()

    def init_prompt(self):
        prompt = [self.BLIPcap.prompt]
        # prompt_text = "A one-word summary of this image: "
        # prompt = [prompt_text]
        prompt = model.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt.input_ids[:, 0] = model.tokenizer.bos_token_id
        prompt.input_ids = prompt.input_ids[:, :-1]
        return prompt
    def forward(self, image):
        captions = []
        image_emb = self.BLIPcap.forward_encoder({"image": image})[:, :-1, :]
        x_pos = torch.linspace(-1, 1, 24).expand(24, 24).flatten().to(self.device) * self.clustering_model.spatial_importance
        y_pos = torch.linspace(-1, 1, 24).expand(24, 24).T.flatten().to(self.device) * self.clustering_model.spatial_importance
        image_emb_with_spatial = torch.cat(
            (image_emb, x_pos.unsqueeze(1).unsqueeze(0), y_pos.unsqueeze(1).unsqueeze(0)), dim=2)
        grid_image_emb = image_emb_with_spatial.unflatten(1, (24, 24))

        #TODO: include the image alongside the embedding
        #TODO: potentially add cluster merging
        clusters = self.clustering_model(grid_image_emb)

        plt.imshow(clusters.squeeze().cpu().detach().numpy())
        plt.title(str(self.clustering_model.spatial_importance) +  " " + str(self.clustering_model.n_iter))
        plt.show()

        # get flattened indices of each cluster
        cluster_indices = []
        for i in range(clusters.min(), clusters.max() + 1):
            cluster_indices.append(torch.where(clusters.flatten() == i))

        # slice image_emb using cluster indices
        cluster_embs = []
        for i in range(len(cluster_indices)):
            cluster_embs.append(image_emb.squeeze()[cluster_indices[i]])

        for emb in cluster_embs:

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

        print(captions)
        return captions

if __name__ == "__main__":
    img_path = "images/cat.jpg"
    image = Image.open(img_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vis_processors, _ = load_model_and_preprocess("blip_caption", "base_coco")
    image = vis_processors["eval"](image).unsqueeze(0).to(device)
    model = MaskBLIP(model, device)

    spatial_importances = [1,3,5,7]
    n_iters = [1,2,3]
    for i in n_iters:
        for j in spatial_importances:
            model.clustering_model = SSNModel(4, i, j).to(device)
            model.forward(image)








