import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import sys
import torch.optim as optim
from torchvision.utils import save_image


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.maxPool = [4, 9, 18, 27, 36]
        self.chosen_features = [0, 5, 10, 19, 28]
        for i in self.maxPool:
            self.vgg_pretrained_features[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.model = self.vgg_pretrained_features[:29]
    def forward(self, X):
        features = []
        for layer_num, layer in enumerate(self.model):
            X = layer(X)

            if layer_num in self.chosen_features:
                features.append(X)

        return features


# model neural style
# # Set device
# device = ('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Hypeparemeters
# learning_rate = 0.001
# num_epochs = 6000
# size = 256
# a = 1
# b = 0.01
#
# transform = transforms.Compose([
#     transforms.Resize((256,256)),
#     transforms.ToTensor(),
# ])
# def load_img(path):
#     img = Image.open(path)
#     img = transform(img).unsqueeze(0)
#     return img.to(device)
#
# path_img= 'img/hoovertowernight.jpg'
# path_style = 'img/candy.jpg'
#
# original_img = load_img(path_img)
# style_img = load_img(path_style)
#
# generated = original_img.clone().requires_grad_(True)
#
# # Create model
# model = VGG().to(device).eval()
#
#
# optimize = optim.Adam([generated], lr=learning_rate)
#
# for epoch in range(num_epochs):
#     # Get features
#     original_features = model(original_img)
#     style_features = model(style_img)
#     gener_features = model(generated)
#
#     style_loss = content_loss = 0
#     for original_feature, style_feature, gener_feature in zip(original_features, style_features, gener_features):
#
#         content_loss += torch.mean((gener_feature-original_feature)**2)
#         batch_size, channel, height, width = gener_feature.shape
#
#         # print(gener_feature.shape, original_feature.shape)
#         G = gener_feature.view(channel, height*width).mm(
#             gener_feature.view(channel, height*width).t()
#         )
#
#         A = style_feature.view(channel, height*width).mm(
#             style_feature.view(channel, height*width).t()
#         )
#
#         style_loss += torch.mean((G-A)**2)
#         # print(G, A)
#
#     total_loss = a*content_loss + b*style_loss
#
#     optimize.zero_grad()
#     total_loss.backward()
#     optimize.step()
#
#     if epoch % 200 ==0:
#         print(total_loss)
#         save_image(generated, 'generated_3.png')