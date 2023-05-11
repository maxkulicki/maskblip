import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from nlp import get_noun_chunks, load_spacy
import spacy
import en_core_web_sm
from maskblip import maskblip_segmentation, plot_results
from xdecoder_semseg import load_xdecoder_model, segment_image, plot_segmentation
from maskblip_diff import MaskBLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# raw_image = Image.open("images/napoleon.jpg").convert("RGB")
#
#
# model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
# # preprocess the image
# # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
# clusters, captions = maskblip_segmentation(raw_image, model, device, vis_processors, n_segments=20, threshold=0.9995, refine=True, plot=False)
# print("\nCaptions: ")
# for caption in captions:
#     print(caption, end=", ")
# spacy_model = load_spacy()
# chunks = get_noun_chunks(captions, spacy_model)
# print("\nChunks: ")
# for chunk in chunks:
#     print(chunk, end=", ")
#
# xdecoder_model = load_xdecoder_model(device)
# xdecoder_segments = segment_image(xdecoder_model, raw_image, chunks, plot=True)

# plot_results(clusters, captions, raw_image)
# plot_segmentation(xdecoder_segments, raw_image, chunks, plot=True)



spacy_model = load_spacy()

img_path = "images/cat.jpg"
raw_image = Image.open(img_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption", "base_coco")
model2, _, _ = load_model_and_preprocess(name="blip_feature_extractor", model_type="base",
                                         is_eval=True, device=device)
model = MaskBLIP(model, device, text_encoder=model2.text_encoder, captioning_adapter=True)
model.tokenizer = model2.tokenizer
model.to(device)
del model2
model.captioning = True

print("model loaded")

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
# compactness = 0.01
# n_clusters = 4
# n_iter = 3
# merging_threshold = None
# model.clustering_model = SSNClusteringModel(device, n_clusters=n_clusters, n_iter=n_iter, compactness=compactness,
#                                             merging_threshold=merging_threshold)

clusters, captions = model.forward(image)
print("\nCaptions: ")
for caption in captions:
    print(caption, end=", ")
chunks = get_noun_chunks(captions, spacy_model)
#chunks = captions
print("\nChunks: ")
for chunk in chunks:
    print(chunk, end=", ")

xdecoder_model = load_xdecoder_model(device)
xdecoder_segments = segment_image(xdecoder_model, raw_image, chunks, plot=True)