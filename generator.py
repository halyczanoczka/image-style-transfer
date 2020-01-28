import torch
from torchvision import models
from imageHelper import load_image, show_images, save_image
import argparse
from train import train

# create parser
parser = argparse.ArgumentParser()

# add arguments to the parser
parser.add_argument("content")
parser.add_argument("style")
parser.add_argument("output_path")
args = parser.parse_args()

# check available device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load vgg16 parameters
vgg16 = models.vgg16(pretrained=True).features
# freeze model parameters
for p in vgg16.parameters():
    p.requires_grad_(False)
vgg16.to(device)

# load images and transform to tensor
content = load_image(args.content).to(device)
style = load_image(args.style, shape=content.shape[-2:]).to(device)

# display input images
show_images(content, style)

# you may choose to leave these as is

output = train(content, style, vgg16, device)
save_image(output, args.output_path)

