import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from skimage.segmentation import slic
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation
from nlp import get_noun_chunks, load_spacy
import spacy

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
        print("Merging clusters")
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

    return clusters

def refine_clusters(blip_clusters, image, n_segments=25, compactness=10, sigma=1):
    image_slic = np.asarray(slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma))
    new_clusters = np.zeros_like(image_slic)
    shape = (image_slic.shape[1], image_slic.shape[0])
    clustered_img = np.asarray(Image.fromarray(blip_clusters.squeeze().astype(np.uint8)).resize(shape, Image.NEAREST))
    for superpixel in np.unique(image_slic):
        superpixel_mask = image_slic == superpixel
        superpixel_label = np.argmax(np.bincount(clustered_img[superpixel_mask]))
        new_clusters[superpixel_mask] = superpixel_label

    return new_clusters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raw_image = Image.open("images/bear.jpg").convert("RGB")

# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# generate caption
image_emb = model.forward_encoder({"image": image})

grid_image_emb = image_emb[:, :-1, :].unflatten(1, (24,24)).squeeze().detach().cpu().numpy()
slic_clusters = slic(grid_image_emb, n_segments=10, compactness=0.001, sigma=1, channel_axis=2, enforce_connectivity=False)
#affinity_clusters = AffinityPropagation().fit_predict(image_emb[:, :-1, :].squeeze().detach().numpy()).reshape(24, 24)
dataset = image_emb[:, :-1, :].squeeze().detach().numpy()
clusters = slic_clusters

clusters = merge_clusters(clusters, grid_image_emb, threshold=0.999)

#get flattened indices of each cluster
cluster_indices = []
for i in range(clusters.min(), clusters.max() + 1):
    cluster_indices.append(np.where(clusters.flatten() == i))

#slice image_emb using cluster indices
cluster_embs = []
for i in range(len(cluster_indices)):
    cluster_embs.append(image_emb[:, :-1, :].squeeze().detach().cpu().numpy()[cluster_indices[i]])

prompt = [model.prompt]
# prompt_text = "A one-word summary of this image: "
# prompt = [prompt_text]
prompt = model.tokenizer(prompt, return_tensors="pt").to(device)
prompt.input_ids[:, 0] = model.tokenizer.bos_token_id
prompt.input_ids = prompt.input_ids[:, :-1]
captions = []
for emb in cluster_embs:
    decoder_out = model.text_decoder.generate_from_encoder(
        tokenized_prompt=prompt,
        visual_embeds=torch.tensor(emb).unsqueeze(0),
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
    caption = [output[len(model.prompt) :] for output in outputs]
    captions.append(caption[0])
for caption in captions:
    print(caption, end=", ")

spacy_model = load_spacy()
chunks = get_noun_chunks(captions, spacy_model)
print("\nChunks: ")
for chunk in chunks:
    print(chunk, end=", ")

new_clusters = refine_clusters(clusters, raw_image, n_segments=150, compactness=20, sigma=5)
fig, axs = plt.subplots(1,3, figsize=(15, 5))
axs[0].imshow(raw_image)
axs[0].imshow(new_clusters, alpha=0.5)
axs[1].imshow(new_clusters)
axs[2].imshow(clusters)
plt.tight_layout()
plt.show()


del model
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
#captions = chunks

caption_embeddings = []
for caption in captions:
    sample = {"image": image, "text_input": ["A picture of a " + caption]}
    cap_emb = model.extract_features(sample, mode="text").text_embeds
    caption_embeddings.append(cap_emb.mean(dim=1))
    #caption_embeddings.append(cap_emb[:,0,:]) # classification token only
caption_embeddings = torch.stack(caption_embeddings).squeeze()

caption_embeddings = caption_embeddings / caption_embeddings.norm(dim=1, keepdim=True)
sims = caption_embeddings @ image_emb.squeeze().t()
matches = torch.argmax(sims, dim=0)[:-1].unflatten(0, (24,24)).detach().cpu().numpy()


fig, axs = plt.subplots(3,1,figsize=(9, 14), gridspec_kw={'height_ratios': [1, 1, 1]})
implot = axs[0].imshow(raw_image)

#axs[0].set_title(full_caption)
cluster_plot = axs[1].imshow(clusters)
values = np.unique(matches.ravel())+1
colors = [ cluster_plot.cmap(cluster_plot.norm(value)) for value in values]
# create a patch (proxy artist) for every color
patches = [ mpatches.Patch(color=colors[i], label=captions[i] ) for i in range(len(values)) ]
# put those patched as legend-handles into the legend
segment_plot = axs[2].imshow(matches)
plt.legend(handles=patches, loc=8, bbox_to_anchor=(0.5, -0.35))

#plt.tight_layout()
plt.show()



