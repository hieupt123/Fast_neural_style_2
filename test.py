import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from neural_style import utils
from neural_style.transformer_net import TransformerNet
import glob
import os


mean = np.array([0.4764, 0.4504, 0.4100])
std = np.array([0.2707, 0.2657, 0.2808])

def getNameFile(pathFile):
    if '\\' in pathFile:
        name = pathFile.split('\\')[-1]
    else:
        name = pathFile.split('/')[-1]
    name, _ = os.path.splitext(name)
    return name

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

def stylize(path_images, root_model, path_save_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(path_save_image) == False:
        os.makedirs(path_save_image)

    # Create transform
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load image
    image_samples= dict()
    for path in glob.glob(f"{path_images}/*"):
        name = getNameFile(path)

        content_image = utils.load_image(path)
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)
        image_samples[name] = content_image

    # Get path model
    path_models = []
    for path in glob.glob(f"{root_model}/*.pth.tar"):
        path_models.append(path)

    for img in image_samples.items():
        for path in path_models:
            with torch.no_grad():
                style_model = TransformerNet()
                state_dict = torch.load(path)
                style_model.load_state_dict(state_dict["state_dict"])

                print('Total loss: ', state_dict['total_loss'])

                style_model.to(device)
                style_model.eval()
                output = style_model(img[1]).cpu()
                name = f'{img[0]}_epoch_{state_dict["current_epoch"]}_batch_idx_{state_dict["start_batch_idx"]-1}.jpg'

            print("Save image")
            save_image(os.path.join(path_save_image, name), output[0])



import time

t = time.time()

path_images = './images/content-images'
path_models = './saved_models/rain_princess'
path_save_image = "./result/rain_princess"

stylize(path_images, path_models, path_save_image)
print(time.time() - t)