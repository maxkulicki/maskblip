import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from nlp import get_noun_chunks, load_spacy
import spacy
import en_core_web_sm
from xdecoder_semseg import load_xdecoder_model, segment_image, plot_segmentation
from maskblip_diff import MaskBLIP
from multiscale_maskblip_kmeans import MultiscaleMaskBLIPK
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode

spacy_model = load_spacy()

img_path = "images/flag.jpg"
raw_image = Image.open(img_path)
transform = Compose([
    ToTensor(),
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])
image = transform(raw_image).unsqueeze(0)
device = "cpu"
device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiscaleMaskBLIPK(device)

clusters, captions = model.forward(image)
print("\nCaptions: ")
for caption in captions:
    print(caption, end=", ")
chunks = get_noun_chunks(captions[0], spacy_model)
#chunks = captions
print("\nChunks: ")
for chunk in chunks:
    print(chunk, end=", ")
del model, clusters, captions, image
torch.cuda.empty_cache()

xdecoder_model = load_xdecoder_model(device2)
xdecoder_segments = segment_image(xdecoder_model, raw_image, chunks, plot=True)