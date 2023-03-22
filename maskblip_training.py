from lavis.models import load_model_and_preprocess
import torch
from diff_slic.model import SSNModel
from PIL import Image
class MaskBLIP(torch.nn.Module):
    def __init__(self, blip_model, device):
        super().__init__()
        self.device = device
        self.BLIPcap = blip_model.to(device)
        self.clustering_model = SSNModel(20, 100, 5).to(device)
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

        image_emb = self.BLIPcap.forward_encoder({"image": image})
        grid_image_emb = image_emb[:, :-1, :].unflatten(1, (24, 24)).squeeze()
        #TODO: include the image alongside the embedding
        #TODO: potentially add cluster merging
        emb_clusters = self.clustering_model(image)#grid_image_emb)
        # get flattened indices of each cluster
        cluster_indices = []
        for i in range(emb_clusters.min(), emb_clusters.max() + 1):
            cluster_indices.append(torch.where(emb_clusters.flatten() == i))
        # slice image_emb using cluster indices
        cluster_embs = []
        for i in range(len(cluster_indices)):
            cluster_embs.append(image_emb[:, :-1, :].squeeze()[cluster_indices[i]])


        decoder_out = model.text_decoder.generate_from_encoder(
            tokenized_prompt=self.prompt,
            visual_embeds=emb_clusters,
            sep_token_id=model.tokenizer.sep_token_id,
            pad_token_id=model.tokenizer.pad_token_id,
            use_nucleus_sampling=False,
            num_beams=3,
            max_length=15,
            min_length=3,
            top_p=0.9,
            repetition_penalty=1.0,
        )
        outputs = model.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)
        return outputs

if __name__ == "__main__":
    img_path = "images/cat.jpg"
    image = Image.open(img_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vis_processors, _ = load_model_and_preprocess("blip_caption", "base_coco")
    image = vis_processors["eval"](image).unsqueeze(0).to(device)
    model = MaskBLIP(model, device)
    model.forward(image)





