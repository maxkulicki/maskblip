#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
sys.path.append('../')
import argparse
import numpy as np
import PIL.Image as Image
import torch
from torchvision import transforms
from scipy import ndimage
from detectron2.utils.colormap import random_color

import dino # model
#from third_party.TokenCut.unsupervised_saliency_detection import metric
from crf import densecrf
from maskcut import maskcut

# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])

def IoU(mask1, mask2):
    mask1, mask2 = (mask1>0.5).to(torch.bool), (mask2>0.5).to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    return (intersection.to(torch.float) / union).mean().item()

def vis_mask(input, mask, mask_color) :
    fg = mask > 0.5
    rgb = np.copy(input)
    rgb[fg] = (rgb[fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)
    return Image.fromarray(rgb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('MaskCut Demo')
    # default arguments
    parser.add_argument('--out-dir', type=str, help='output directory')
    parser.add_argument('--vit-arch', type=str, default='base', choices=['base', 'small'], help='which architecture')
    parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')
    parser.add_argument('--patch-size', type=int, default=8, choices=[16, 8], help='patch size')
    parser.add_argument('--img-path', type=str, default='imgs', help='single image visualization')
    parser.add_argument('--tau', type=float, default=0.05, help='threshold used for producing binary graph')

    # additional arguments
    parser.add_argument('--fixed_size', type=int, default=480, help='rescale the input images to a fixed size')
    parser.add_argument('--pretrain_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--N', type=int, default=5, help='the maximum number of pseudo-masks per image')
    parser.add_argument('--cpu', action='store_false', help='use cpu')
    parser.add_argument('--output_path', type=str,  default='', help='path to save outputs')

    args = parser.parse_args()
    print (args)

    if args.pretrain_path is not None:
        url = args.pretrain_path
    if args.vit_arch == 'base' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        feat_dim = 768
    elif args.vit_arch == 'small' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        feat_dim = 384

    backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)

    msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
    print (msg)
    backbone.eval()
    args.cpu = False
    if not args.cpu:
        backbone.cuda()

    for img_path in os.listdir(args.img_path):
        img_path = os.path.join(args.img_path, img_path)
        bipartitions, _, I_new = maskcut(img_path, backbone, args.patch_size, args.tau, \
            N=args.N, fixed_size=args.fixed_size, cpu=False)

        I = Image.open(img_path).convert('RGB')
        width, height = I.size
        pseudo_mask_list = []
        for idx, bipartition in enumerate(bipartitions):
            # post-process pesudo-masks with CRF
            pseudo_mask = densecrf(np.array(I_new), bipartition)
            pseudo_mask = ndimage.binary_fill_holes(pseudo_mask>=0.5)

            # filter out the mask that have a very different pseudo-mask after the CRF
            if not args.cpu:
                mask1 = torch.from_numpy(bipartition).cuda()
                mask2 = torch.from_numpy(pseudo_mask).cuda()
            else:
                mask1 = torch.from_numpy(bipartition)
                mask2 = torch.from_numpy(pseudo_mask)
            if IoU(mask1, mask2) < 0.5:
                pseudo_mask = pseudo_mask * -1

            # construct binary pseudo-masks
            pseudo_mask[pseudo_mask < 0] = 0
            pseudo_mask = Image.fromarray(np.uint8(pseudo_mask*255))
            pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))

            pseudo_mask = pseudo_mask.astype(np.uint8)
            upper = np.max(pseudo_mask)
            lower = np.min(pseudo_mask)
            thresh = upper / 2.0
            pseudo_mask[pseudo_mask > thresh] = upper
            pseudo_mask[pseudo_mask <= thresh] = lower
            if pseudo_mask.sum() > 0:
                pseudo_mask_list.append(pseudo_mask)

        file_name = os.path.splitext(os.path.basename(img_path))[0]

        input = np.array(I)
        for pseudo_mask in pseudo_mask_list:
            input = vis_mask(input, pseudo_mask, random_color(rgb=True))
        input.save(os.path.join("masked_imgs", f"{file_name}.jpg"))

        np.savez(f'annotations/{file_name}.npz', *pseudo_mask_list)
        print(f'Saved annotation: {file_name}.npz')

    # bipartitions, _, I_new = maskcut(args.img_path, backbone, args.patch_size, args.tau, \
    #     N=args.N, fixed_size=args.fixed_size, cpu=args.cpu)
    # I = Image.open(args.img_path).convert('RGB')
    # width, height = I.size
    # pseudo_mask_list = []
    # for idx, bipartition in enumerate(bipartitions):
    #     # post-process pesudo-masks with CRF
    #     pseudo_mask = densecrf(np.array(I_new), bipartition)
    #     pseudo_mask = ndimage.binary_fill_holes(pseudo_mask>=0.5)
    #
    #     # filter out the mask that have a very different pseudo-mask after the CRF
    #     if not args.cpu:
    #         mask1 = torch.from_numpy(bipartition).cuda()
    #         mask2 = torch.from_numpy(pseudo_mask).cuda()
    #     else:
    #         mask1 = torch.from_numpy(bipartition)
    #         mask2 = torch.from_numpy(pseudo_mask)
    #     if IoU(mask1, mask2) < 0.5:
    #         pseudo_mask = pseudo_mask * -1
    #
    #     # construct binary pseudo-masks
    #     pseudo_mask[pseudo_mask < 0] = 0
    #     pseudo_mask = Image.fromarray(np.uint8(pseudo_mask*255))
    #     pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))
    #
    #     pseudo_mask = pseudo_mask.astype(np.uint8)
    #     upper = np.max(pseudo_mask)
    #     lower = np.min(pseudo_mask)
    #     thresh = upper / 2.0
    #     pseudo_mask[pseudo_mask > thresh] = upper
    #     pseudo_mask[pseudo_mask <= thresh] = lower
    #     pseudo_mask_list.append(pseudo_mask)
    #
    # input = np.array(I)
    # for pseudo_mask in pseudo_mask_list:
    #     input = vis_mask(input, pseudo_mask, random_color(rgb=True))
    # input.save(os.path.join(args.output_path, "demo.jpg"))