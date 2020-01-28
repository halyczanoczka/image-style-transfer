import torch
from torchvision import models
import imageHelpers


def generateImage(content_img_file_path, style_img_file_path):
    # load vgg16 parameters
    vgg16 = models.vgg16(pretrained=True).features

    # freeze model parameters
    for p in vgg16.parameters():
        p.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg16.to(device)

    content = imageHelpers.load_image(content_img_file_path).to(device)
    style = imageHelpers.load_image(style_img_file_path, shape=content.shape[-2:]).to(device)

    # TO DO : display image here

    # TO DO : get features

    # TO DO : calculate gradient
