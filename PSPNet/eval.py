#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from net.pspnet import PSPNet
from PIL import Image
import cv2

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

parser = argparse.ArgumentParser(description="Pyramid Scene Parsing Network")
parser.add_argument('--data-path', type=str, default='',help='Path to dataset folder')
parser.add_argument('--models-path', type=str, default='./checkpoints', help='Path for storing model snapshots')
parser.add_argument('--backend', type=str, default='densenet', help='Feature extractor')
parser.add_argument('--img_path', type=str, default='test/Tornado_Moore_Before_01.jpg', help='Path for image')
args = parser.parse_args()

def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        # if not epoch == 'last':
        #     epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch

def get_transform():
    transform_image_list = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]


    data_transforms = transforms.Compose(transform_image_list)
    return data_transforms

def main():
    # --------------- model --------------- #
    snapshot = os.path.join(args.models_path, args.backend, 'PSPNet_last')
    net, starting_epoch = build_network(snapshot, args.backend)
    net.eval()

    data_transform = get_transform()

    image_size = 256
    images = os.listdir("test/Beirut_Explosion")
    print(images)
    for i, img in enumerate(images):
        # print(type(i))
        img_name = img
        img = Image.open("test/Beirut_Explosion" + "/" + img)
        img = img.resize((512,512))

        if len(img.getbands()) != 3:
            img = img.convert('RGB')
        img = np.array(img)

        h, w, _ = img.shape
        mask_whole = np.zeros((h, w), dtype=np.uint8)

        stride = 256
        with torch.no_grad():
            for i in range(h // stride):
                for j in range(w // stride):
                    crop = img[i * stride:i * stride + image_size, j * stride:j * stride + image_size]
                    ch, cw, _ = crop.shape
                    crop = data_transform(crop)
                    crop = torch.unsqueeze(crop,dim=0)

                    pred_seg, pred_cls = net(crop.cuda())
                    pred_seg = pred_seg[0]
                    pred = pred_seg.cpu().numpy().transpose(1, 2, 0)
                    pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8).reshape((256, 256, 1))
                    pred = pred.reshape((ch,cw))

                    mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size] = pred[:, :]

            AAL = mask_whole.shape[0]*mask_whole.shape[1]*255
            mask_whole[mask_whole>0] = 255
            Have = mask_whole.sum()
            print(img_name,Have / AAL)
            Image.fromarray(mask_whole).save('demo'+'/'+img_name+'.png')

if __name__ == '__main__':
    main()