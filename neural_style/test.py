import numpy as np
import torch
from torchvision import transforms
import torch.onnx
from PIL import Image
import utils
# from transformer_net import TransformerNet
from transformer_net_2 import TransformerNet


import sys

# style_model = TransformerNet()
# state_dict = torch.load('saved_models/ckpt_epoch_0_batch_id_300.pth')
# print(state_dict.keys())
# sys.exit()

mean = np.array([0.4764, 0.4504, 0.4100])
std = np.array([0.2707, 0.2657, 0.2808])

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

def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(args.model)
        style_model.load_state_dict(state_dict["state_dict"])

        print('Total loss: ', state_dict['total_loss'])

        style_model.to(device)
        style_model.eval()
        output = style_model(content_image).cpu()
    save_image(args.output_image, output[0])


import time

t = time.time()

args = type('', (), {})()
args.content_scale = None
args.cuda = 0
args.export_onnx = ""
args.content_image = 'images/content-images/hoovertowernight.jpg'
# args.output_image = "result/styled-water3.jpg"
args.output_image = "result/styled.jpg"

args.model = "saved_models/mosaic_ckpt_epoch_1_batch_id_4000.pth.tar"
stylize(args)
print(time.time() - t)