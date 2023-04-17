import torch
from PIL import Image
import numpy as np


mean = [0.4763, 0.4507, 0.4094]
std = [0.2702, 0.2652, 0.2811]

def load_image(filename, size=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    return img


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def deprocess(image_tensor):
    """ Denormalizes and rescales image tensor """
    unnorm = UnNormalize(mean=mean, std=std)
    img = image_tensor
    unnorm(img)
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

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
