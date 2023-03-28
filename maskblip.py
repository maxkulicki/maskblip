from lavis.models import load_model_and_preprocess
import torch
from PIL import Image
from skimage.segmentation import slic
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation

from nlp import get_noun_chunks, load_spacy
import spacy

def merge_clusters(clusters, embs, device, threshold=0.995):
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
        # print("Similarity: ", similarities[most_similar])
        # print("Merging clusters")
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

def refine_clusters(blip_clusters, image, n_segments=150, compactness=10, sigma=1):
    image_slic = np.asarray(slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma))
    new_clusters = np.zeros_like(image_slic)
    shape = (image_slic.shape[1], image_slic.shape[0])
    clustered_img = np.asarray(Image.fromarray(blip_clusters.squeeze().astype(np.uint8)).resize(shape, Image.NEAREST))
    for superpixel in np.unique(image_slic):
        superpixel_mask = image_slic == superpixel
        superpixel_label = np.argmax(np.bincount(clustered_img[superpixel_mask]))
        new_clusters[superpixel_mask] = superpixel_label

    return new_clusters

def maskblip_segmentation(raw_image, model, device, vis_processors, n_segments=4, threshold=0.9995, refine=False, plot=False):
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    # generate caption
    image_emb = model.forward_encoder({"image": image})

    grid_image_emb = image_emb[:, :-1, :].unflatten(1, (24, 24)).squeeze().detach().cpu().numpy()
    slic_clusters = slic(grid_image_emb, n_segments=n_segments, compactness=0.01, sigma=1, channel_axis=2,
                         enforce_connectivity=False)
    clusters = slic_clusters
    print("clusters before merging: ", len(np.unique(clusters)))
    clusters = merge_clusters(clusters, grid_image_emb, device, threshold=threshold)
    print("clusters after merging: ", len(np.unique(clusters)))

    # get flattened indices of each cluster
    cluster_indices = []
    for i in range(clusters.min(), clusters.max() + 1):
        cluster_indices.append(np.where(clusters.flatten() == i))

    # slice image_emb using cluster indices
    cluster_embs = []
    for i in range(len(cluster_indices)):
        cluster_embs.append(image_emb[:, :-1, :].squeeze()[cluster_indices[i]])

    prompt = [model.prompt]
    # prompt_text = "A one-word summary of this image: "
    # prompt = [prompt_text]
    prompt = model.tokenizer(prompt, return_tensors="pt").to(device)
    prompt.input_ids[:, 0] = model.tokenizer.bos_token_id
    prompt.input_ids = prompt.input_ids[:, :-1]
    prompt.to(device)
    captions = []
    for emb in cluster_embs:
        decoder_out = model.text_decoder.generate_from_encoder(
            tokenized_prompt=prompt,
            visual_embeds=emb.clone().detach().unsqueeze(0),
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
        caption = [output[len(model.prompt):] for output in outputs]
        captions.append(caption[0])

    if refine:
        clusters = refine_clusters(clusters, np.asarray(raw_image), n_segments=150, compactness=20, sigma=5)
        captions = [captions[i-1] for i in np.unique(clusters)]
        unique = np.unique(clusters)
        new_clusters  = np.zeros_like(clusters)
        for i in range(len(unique)):
            new_clusters[clusters == unique[i]] = i
        clusters = new_clusters
    print("clusters after refining: ", len(np.unique(clusters)))

    if plot:
        plot_results(clusters, captions, raw_image)

    return clusters, captions

def plot_results(clusters, captions, raw_image):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].imshow(raw_image)
    axs[0].set_title("Original Image")
    cluster_plot = axs[1].imshow(clusters)
    axs[1].set_title("Segmented Image")
    values = np.unique(clusters.ravel())
    colors = [ cluster_plot.cmap(cluster_plot.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    patches = [ mpatches.Patch(color=colors[i], label=captions[i] ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches)
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_image = Image.open("images/bear.jpg").convert("RGB")

    # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
    # this also loads the associated image processors
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    # preprocess the image
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    clusters, captions = maskblip_segmentation(raw_image, model, device, vis_processors, refine=False)

    spacy_model = load_spacy()
    chunks = get_noun_chunks(captions, spacy_model)
    print("\nChunks: ")
    for chunk in chunks:
        print(chunk, end=", ")

    plot_results(clusters, captions, raw_image)