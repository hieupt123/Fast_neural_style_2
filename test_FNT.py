import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from neural_style import utils
from neural_style.transformer_net import TransformerNet
import glob
import os
import sys
from pathlib import Path


mean = np.array([0.4764, 0.4504, 0.4100])
std = np.array([0.2707, 0.2657, 0.2808])

def getNameFile(pathFile):
    path = Path(pathFile)
    return path.stem


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

def transform_style_image(path_image, path_model, path_save_image="result"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(path_save_image) == False:
        os.makedirs(path_save_image)

    name_image = Path(path_image).stem
    name_model = Path(path_model).stem

    # Create transform
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load image
    content_image = utils.load_image(path_image)
    content_image = style_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(path_model)
        style_model.load_state_dict(state_dict["state_dict"])

        print('Total loss: ', state_dict['total_loss'])

        style_model.to(device)
        style_model.eval()
        output = style_model(content_image).cpu()
        image_grid = denormalize(output.cpu())

        name = f'{name_image}_{name_model}.jpg'

    path_save = os.path.join(path_save_image, name)
    print(f"Save image to {path_save}")
    save_image(image_grid, path_save)

if __name__ == "__main__":
    path_image = 'images/content-images/amber.jpg'
    path_model = 'saved_models/candy/candy_ckpt_epoch_9_batch_id_4400.pth.tar'
    path_save_image = "styled.jpg"
    transform_style_image(path_image, path_model, path_save_image)