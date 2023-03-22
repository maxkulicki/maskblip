import os
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from skimage.segmentation import slic
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import json
from pycocotools.coco import COCO
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

def extract_nouns(text, lemmatizer):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Tag the words with their part-of-speech (POS)
    tagged_words = pos_tag(words)
    # Extract the nouns in their base form (singular or mass noun)
    nouns = []
    for word, pos in tagged_words:
        if pos.startswith('NN'):
            if word in ['people', 'man', 'woman']:
                nouns.append('person')
            else:
                nouns.append(lemmatizer.lemmatize(word, pos='n'))

    return list(set(nouns))
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

    similarities = cosine_similarity(cluster_embs)
    for i in range(len(similarities)):
        similarities[i, i] = 0
    most_similar = np.unravel_index(similarities.argmax(), similarities.shape)

    while similarities[most_similar] > threshold:
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

def get_cluster_captions(clusters, image_emb, model, device):
    # get flattened indices of each cluster
    cluster_indices = []
    for i in range(clusters.min(), clusters.max() + 1):
        cluster_indices.append(np.where(clusters.flatten() == i))

    # slice image_emb using cluster indices
    cluster_embs = []
    for i in range(len(cluster_indices)):
        cluster_embs.append(image_emb[:, :-1, :].squeeze()[cluster_indices[i]])

    prompt = [model.prompt]  # * image_embeds.size(0)
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
            max_length=20,
            min_length=3,
            top_p=0.9,
            repetition_penalty=1.0,
        )
        outputs = model.tokenizer.batch_decode(decoder_out, skip_special_tokens=True)
        caption = [output[len(model.prompt):] for output in outputs]
        captions += caption
    return captions

def synset_match(nouns, synset):
    if 'cow' in nouns:
        print("cow in nouns")
    for noun in nouns:
        for syn in synset:
            if noun.strip() in syn.split(" "):
                return True
    return False
def is_substring_in_list(substring, string_list):
    for string in string_list:
        if substring in string:
            return string
    return None

def get_recall_precision(caption_nouns, label_synsets):
    matches = 0
    for label_synset in label_synsets:
        if synset_match(caption_nouns, label_synset):
            matches += 1
        # else:
        #     print("Label not in captions: ", label_synset)
    if len(label_synsets) == 0:
        recall = 1
        precision = 1
    else:
        recall = matches / len(label_synsets)
        precision = matches / len(caption_nouns)
    return recall, precision

def get_ade20k_labels(annotations_path):
    labels = []
    with open(annotations_path, encoding="utf8") as f:
        for line in f:
            label = line.split("# ")[3].split(",")
            label = [l.strip() for l in label]
            if label not in labels:
                labels.append(label)
    return labels

def evaluate_captioning_method(model, device, lemmatizer, local=True, threshold=0.995, n_segments=10, n_samples=10, plot=True):
    recalls = []
    precisions = []
    avg_n_clusters = 0
    for image_path in os.listdir(ade20k_dir)[:n_samples]:
        # check if the image ends with png
        if (image_path.endswith(".jpg")):
            annotations_path = image_path.replace(".jpg", "_atr.txt")
            labels = get_ade20k_labels(ade20k_dir + annotations_path)
            #print("Ground truth labels: ", labels)
            img = Image.open(ade20k_dir + image_path)
            # check if image is grayscale
            if len(np.array(img).shape) == 2:
                img = img.convert("RGB")
            img = vis_processors["eval"](img).unsqueeze(0).to(device)
            if not local:
                caption = model.generate({"image": img})[0]
                #print("Caption: ", caption)
                caption_nouns = extract_nouns(caption, lemmatizer)
                recall, precision = get_recall_precision(caption_nouns, labels)
            else:
                image_emb = model.forward_encoder({"image": img})
                grid_image_emb = image_emb[:, :-1, :].unflatten(1, (24, 24)).squeeze().detach().cpu().numpy()
                slic_clusters = slic(grid_image_emb, n_segments=n_segments, compactness=0.001, sigma=1, channel_axis=2,
                                     enforce_connectivity=False)
                clusters = slic_clusters
                clusters = merge_clusters(clusters, grid_image_emb, threshold=threshold)
                captions = get_cluster_captions(clusters, image_emb, model, device)
                caption_nouns = []
                for caption in captions:
                    print("Caption: ", caption)
                    caption_nouns += extract_nouns(caption, lemmatizer)
                caption_nouns = list(set(caption_nouns))
                #print("Local caption nouns: ", caption_nouns)

                recall, precision = get_recall_precision(caption_nouns, labels)
                n_clusters = len(np.unique(clusters))
                avg_n_clusters += n_clusters

            recalls.append(recall)
            precisions.append(precision)
            #print("Recall: ", recall)

    avg_recall = round(np.mean(recalls), 5)
    avg_precision = round(np.mean(precisions), 5)
    avg_n_clusters = round(avg_n_clusters / n_samples, 5)

    print("Average recall: ", avg_recall)
    print("Average precision: ", avg_precision)
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        if local:
            fig.suptitle("Local captioning, {n_segments} segments, {threshold} threshold")
        else:
            fig.suptitle(f"Global captioning")

        axs[0].hist(recalls, bins=10)
        axs[0].set_xlabel("Recall")
        axs[0].set_title("Average recall: " + str(avg_recall))

        axs[1].hist(precisions, bins=10)
        axs[1].set_xlabel("Precision")
        axs[1].set_title("Average precision: " + str(avg_precision))
        plt.show()
    return avg_recall, avg_precision, avg_n_clusters

def plot_metric(metrics, metric_name, thresholds, segment_numbers):
    metrics = np.array(metrics).reshape(len(thresholds), len(segment_numbers))
    fig, ax = plt.subplots()
    ax.matshow(metrics, cmap='seismic')
    ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    ax.set_title(metric_name)
    ax.set_xticks(np.arange(len(segment_numbers)))
    ax.set_yticks(np.arange(len(thresholds)))
    ax.set_xticklabels(segment_numbers)
    ax.set_yticklabels(thresholds)
    ax.set_xlabel('Number of segments')
    ax.set_ylabel('Threshold')
    for (i, j), z in np.ndenumerate(metrics):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    plt.tight_layout()
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
lemmatizer = WordNetLemmatizer()

recalls = []
precisions = []
full_image_recalls = []
full_image_precisions = []
n_samples = 64

ade20k_dir = "datasets/original_ade20k/"
thresholds = [0.9995, 0.9999,  0.99995]
segment_numbers = [20, 25, 30, 35, 40]
clustering_methods = [(x, y) for x in thresholds for y in segment_numbers]

print("Evaluating global captioning")
evaluate_captioning_method(model, device, lemmatizer, local=False, n_samples=n_samples)
recalls = []
precisions = []
cluster_numbers = []

for threshold, n_segments in clustering_methods:
    print(f"Evaluating local captioning with threshold {threshold} and {n_segments} segments")
    recall, precision, cluster_number = evaluate_captioning_method(model, device, lemmatizer, local=True, threshold=threshold,\
                                            n_segments=n_segments, n_samples=n_samples, plot=False)
    recalls.append(recall)
    precisions.append(precision)
    cluster_numbers.append(cluster_number)

plot_metric(recalls, "Average Recall", thresholds, segment_numbers)
plot_metric(precisions, "Average precision", thresholds, segment_numbers)
plot_metric(cluster_numbers, "Average nr of clusters", thresholds, segment_numbers)