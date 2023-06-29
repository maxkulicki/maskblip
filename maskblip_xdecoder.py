import torch
from PIL import Image
import spacy
from nlp import get_noun_chunks
from xdecoder_semseg import load_xdecoder_model, segment_image
from maskblip import MaskBLIP
from torchvision.transforms import Compose, ToTensor, Normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spacy_model = spacy.load("en_core_web_sm")
img_path = "images/bird.jpg"
raw_image = Image.open(img_path)
transform = Compose([
    ToTensor(),
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])
image = transform(raw_image).unsqueeze(0)
device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MaskBLIP(device)

clusters, captions = model(image, clean=True)
print("\nCaptions: ")
for caption in captions:
    print(caption, end=", ")
chunks = get_noun_chunks(captions[0], spacy_model)
print("\nChunks: ")
for chunk in chunks:
    print(chunk, end=", ")
del model, clusters, captions, image
torch.cuda.empty_cache()

xdecoder_model = load_xdecoder_model(device2)
xdecoder_segments = segment_image(xdecoder_model, raw_image, chunks, plot=True)