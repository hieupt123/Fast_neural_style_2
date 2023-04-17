"""
Using method Neural style transfer
"""
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from VGG import VGG

model = models.vgg19(pretrained=True)

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

def getFileName(path):
    name = path.split("/")[-1]
    file_name = os.path.splitext(name)[0]
    return file_name

device = torch.device("cuda" if torch.cuda.is_available else 'cpu')
image_size = 256

loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[], std=[])
    ]
)

path_img = 'images/content-images/amber.jpg'
path_style = 'images/New_style/weeping-woman-by-pablo-picasso.jpg'
original_img = load_image(path_img)
style_img = load_image(path_style)

# generated = torch.randn(original_img.shape, device=device, requires_grad=True)
generated = original_img.clone().requires_grad_(True)



model = VGG().to(device).eval()
# Hyperparameters
total_steps = 60000
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated], lr=learning_rate)

# Create dir to save image
# Get style name
style_name = getFileName(path_style)
path_dir_save_image = f'images/output-images/{style_name}/'
if os.path.exists(path_dir_save_image) == False:
    os.mkdir(path_dir_save_image)

# get file name
file_name = getFileName(path_img)

def cal_loss(gen_feature, orig_feature, style_feature):
    batch_size, channel, height, width = gen_feature.shape
    original_loss = torch.mean((gen_feature - orig_feature) ** 2)

    # Computer Gram Matrix
    G = gen_feature.view(channel, height * width).mm(
        gen_feature.view(channel, height * width).t()
    )

    A = style_feature.view(channel, height * width).mm(
        style_feature.view(channel, height * width).t()
    )
    style_loss = torch.mean((G - A) ** 2)
    return original_loss, style_loss

for step in range(total_steps):
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)

    style_loss = original_loss = 0
    # original_loss, style_loss = cal_loss(generated_features, original_img_features, style_features)

    for gen_feature, orig_feature, style_feature in zip(generated_features,
                                            original_img_features, style_features):
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature)**2)

        # Computer Gram Matrix
        G = gen_feature.view(channel, height*width).mm(
            gen_feature.view(channel, height*width).t()
        )

        A = style_feature.view(channel, height*width).mm(
            style_feature.view(channel, height*width).t()
        )
        style_loss += torch.mean((G -A)**2)


    total_loss = alpha*original_loss + beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 ==0:
        print(total_loss)
        save_image(generated, os.path.join(path_dir_save_image, f'{file_name}_step_{step}.png'))
