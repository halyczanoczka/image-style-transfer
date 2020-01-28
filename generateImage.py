import torch
from torchvision import models
import imageHelper
import argparse


def generateImage(content_img_file_path: str, style_img_file_path: str):

    # load vgg16 parameters
    vgg16 = models.vgg16(pretrained=True).features

    # freeze model parameters
    for p in vgg16.parameters():
        p.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg16.to(device)


    content = imageHelper.load_image(content_img_file_path).to(device)
    style = imageHelper.load_image(style_img_file_path, shape=content.shape[-2:]).to(device)

    # TO DO : display image here
    imageHelper.show_images(content, style)
    # TO DO : get features

    # TO DO : calculate gradient



# create parser
parser = argparse.ArgumentParser()

# add arguments to the parser
parser.add_argument("content")
parser.add_argument("style")
args = parser.parse_args()

generateImage(args.content, args.style)