from featuresHelper import get_features, gram_matrix
from imageHelper import show_images
from torch import optim
import torch

# -----------------------------------------------------
# modify this parameters to achieve satisfying results

style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.7,
                 'conv5_1': 0.2}

alpha = 1  # content_weight
beta = 10  # style_weight

steps = 5000

# -----------------------------------------------------


def train(content, style, model, device):

    # get features
    content_features = get_features(content, model)
    style_features = get_features(style, model)

    # calculate the gram matrices for each layer of our style representation
    # TODO: remove conv4_2 from style features
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # create a third "target" image and prep it for change
    target = content.clone().requires_grad_(True).to(device)

    # iteration hyperparameters
    optimizer = optim.Adam([target], lr=0.003)

    for ii in range(1, steps + 1):

        target_features = get_features(target, model)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        # the style loss
        # initialize the style loss to 0
        style_loss = 0
        # iterate through each style layer and add to the style loss
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape

            # Calculate the target gram matrix
            target_gram = gram_matrix(target_feature)

            # get the "style" style representation
            style_gram = style_grams[layer]

            # Calculate the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((style_gram - target_gram) ** 2)

            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)

        # calculate the *total* loss
        total_loss = alpha * content_loss + beta * style_loss

        # update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print(f'Step {ii}/{steps}')


    return target