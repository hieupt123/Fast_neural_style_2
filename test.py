"""
Style transformer using method Faster Neural Style
"""

import torch
from torchvision import transforms
from neural_style import utils
from neural_style.transformer_net import TransformerNet

def transformstyle(path_image, path_save, weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_image = utils.load_image(path_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(weight)
        style_model.load_state_dict(state_dict["state_dict"])

        print('Total loss: ', state_dict['total_loss'])

        style_model.to(device)
        style_model.eval()
        output = style_model(content_image).cpu()
    utils.save_image(path_save, output[0])

import time
if __name__ == '__main__':
    t = time.time()
    path_image = 'images/content-images/amber.jpg'
    path_save = "style_trans.jpg"

    # load model
    TransformerNetWEIGHT = "saved_models/rain_princess/rain_princess.pth.tar"
    transformstyle(path_image, path_save, TransformerNetWEIGHT)

    print(time.time() - t)

