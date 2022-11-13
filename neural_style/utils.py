import torch
from PIL import Image
import numpy as np


mean = [0.4763, 0.4507, 0.4094]
std = [0.2702, 0.2652, 0.2811]

def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img



def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return tensors


def deprocess(image_tensor):
    """ Denormalizes and rescales image tensor """
    img = denormalize(image_tensor)
    img *= 255
    image_np = torch.clamp(img, 0, 255).numpy().astype(np.uint8)
    image_np = image_np.transpose(1, 2, 0)
    return image_np

def save_image(filename, data):
    img = deprocess(data)
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

