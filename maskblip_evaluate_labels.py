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
    two_word_classes = ['traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'baseball bat', 'baseball glove', \
                        'tennis racket', 'wine glass', 'hot dog', 'potted plant', 'dining table', 'cell phone', 'teddy bear', 'hair drier']
    # Tokenize the text into words
    words = word_tokenize(text)

    # Tag the words with their part-of-speech (POS)
    tagged_words = pos_tag(words)
    # Extract the nouns in their base form (singular or mass noun)
    nouns = []
    for word, pos in tagged_words:
        if pos.startswith('NN'):
            match = is_substring_in_list(word, two_word_classes)
            if match is not None:
                nouns.append(match)
            elif word in ['people', 'man', 'woman']:
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

def get_cluster_captions(clusters):
    # get flattened indices of each cluster
    cluster_indices = []
    for i in range(clusters.min(), clusters.max() + 1):
        cluster_indices.append(np.where(clusters.flatten() == i))

    # slice image_emb using cluster indices
    cluster_embs = []
    for i in range(len(cluster_indices)):
        cluster_embs.append(image_emb[:, :-1, :].squeeze().detach().cpu().numpy()[cluster_indices[i]])

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
            syn = syn.replace('_', ' ')
            if noun == syn:
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

# specify the path to the COCO annotation file and the directory containing the images
annFile = 'datasets/coco2017/instances_val2017.json'
imgDir = 'datasets/coco2017/val2017_imgs'

# initialize COCO api for instance annotations
coco = COCO(annFile)
synsets = json.load(open("datasets/coco2017/coco_to_synset.json", "r"))
lemmatizer = WordNetLemmatizer()
# get all image ids in the dataset
imgIds = coco.getImgIds()

recalls = []
precisions = []
full_image_recalls = []
full_image_precisions = []
n_samples = 100

np.random.seed(1)
random_indices = np.random.choice(len(imgIds), n_samples, replace=False)
for id in random_indices:
    img = coco.loadImgs(imgIds[id])[0]
    annIds = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(annIds)

    img = Image.open(imgDir + "/" + img['file_name'])
    #check if image is grayscale
    if len(np.array(img).shape) == 2:
        img = img.convert("RGB")
    img = vis_processors["eval"](img).unsqueeze(0).to(device)

    full_image_caption = model.generate({"image": img})[0]
    full_image_caption_nouns = extract_nouns(full_image_caption, lemmatizer)
    print("Full image caption nouns: ", full_image_caption_nouns)
    image_emb = model.forward_encoder({"image": img})
    grid_image_emb = image_emb[:, :-1, :].unflatten(1, (24, 24)).squeeze().detach().cpu().numpy()
    slic_clusters = slic(grid_image_emb, n_segments=10, compactness=0.001, sigma=1, channel_axis=2,
                         enforce_connectivity=False)
    dataset = image_emb[:, :-1, :].squeeze().detach().numpy()
    clusters = slic_clusters
    clusters = merge_clusters(clusters, grid_image_emb, threshold=0.999)
    captions = get_cluster_captions(clusters)
    caption_nouns = []
    for caption in captions:
        caption_nouns += extract_nouns(caption, lemmatizer)
    caption_nouns = list(set(caption_nouns))
    print("Local caption nouns: ", caption_nouns)

    # extract all class labels for the given image as a list
    label_synsets = []
    labels = []
    for ann in anns:
        label = coco.loadCats(ann['category_id'])[0]['name']
        if label not in labels and label != 'stop sign':
            labels.append(label)
            label_synset = wordnet.synset(synsets[label]['synset'])
            synonyms = label_synset.lemma_names()
            label_synsets.append(synonyms)

    print("Ground truth labels: ", labels)

    recall, precision = get_recall_precision(caption_nouns, label_synsets)
    full_image_recall, full_image_precision = get_recall_precision(full_image_caption_nouns, label_synsets)

    recalls.append(recall)
    precisions.append(precision)
    full_image_recalls.append(full_image_recall)
    full_image_precisions.append(full_image_precision)

    print("Recall: ", recall)

avg_recall = round(np.mean(recalls), 5)
avg_precision = round(np.mean(precisions), 5)

avg_full_image_recall = round(np.mean(full_image_recalls), 5)
avg_full_image_precision = round(np.mean(full_image_precisions), 5)
print("Average recall: ", avg_recall)
print("Average precision: ", avg_precision)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Local captioning")

axs[0].hist(recalls, bins=10)
axs[0].set_xlabel("Recall")
axs[0].set_title("Average recall: " + str(avg_recall))

axs[1].hist(precisions, bins=10)
axs[1].set_xlabel("Precision")
axs[1].set_title("Average precision: " + str(avg_precision))
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Full image captioning")

axs[0].hist(full_image_recalls, bins=10)
axs[0].set_xlabel("Recall")
axs[0].set_title("Average recall: " + str(avg_full_image_recall))

axs[1].hist(full_image_precisions, bins=10)
axs[1].set_xlabel("Precision")
axs[1].set_title("Average precision: " + str(avg_full_image_precision))
plt.show()




