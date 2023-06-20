import os
import torch
from segmentation_dataset import SegmentationDataset
from cutler_dataset import CutlerDataset
from maskblip import MaskBLIP, plot_result
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, Normalize
from xdecoder_semseg import load_xdecoder_model, segment_image
from nlp import get_noun_chunks
import json

if __name__ == "__main__":
    n_captions = 5
    batch_size = 1
    wandb_track = False
    supervised = True

    device = 'cuda'
    model = MaskBLIP(device)
    model.captioning = False

    transform = Compose([
        ToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])

    if supervised:
        dataset_dir = os.path.join("../datasets", "VOC2012")
        dataset = SegmentationDataset(dataset_dir, transform=transform, img_size=model.output_size)
    else:
        dataset_dir = os.path.join("cutler", "maskcut")
        dataset = CutlerDataset(dataset_dir, transform=transform, img_size=model.output_size)


    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )
    result_dict = {}
    for batch in tqdm(data_loader):
        if batch[0].tolist() == [1, 1, 1] and batch[1].tolist() == [1, 1, 1]:
            print("Skipped: ", batch[-1])
            continue
        raw_images, annotations, paths = batch
        raw_images = raw_images.to(device)

        # output = cluster assigments
        output = model(raw_images, clean=True)

        chunk_list = []
        for i in range(n_captions):
            captions = model.generate_captions(raw_images, output)
            chunks = get_noun_chunks(captions[0], model.spacy_model)
            chunk_list += chunks
        chunk_list = list(set(chunk_list))
        stripped_name = paths[0].split("/")[-1]
        result_dict[stripped_name] = chunk_list

    with open('./results/maskblip_results_val.json', 'w') as fp:
        json.dump(result_dict, fp, indent=4)