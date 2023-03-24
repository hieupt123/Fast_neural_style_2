"""
Image transformation networks
"""
import torch
from torch import nn

class Residual_block(nn.Module):
  """Residual block
  Architecture: https://arxiv.org/pdf/1610.02915.pdf
  """
  def __init__(self, channel):
    super(Residual_block, self).__init__()
    self.batchnorm = nn.BatchNorm2d(channel)
    self.conv2d = nn.Conv2d(in_channels=channel, out_channels=channel,
                            padding='same', kernel_size=3, stride=1)
    self.relu = nn.ReLU()

  def forward(self, x):
    residual = x
    out = self.relu(self.batchnorm(self.conv2d(self.batchnorm(x))))
    out = self.batchnorm(self.conv2d(out))
    return out + x

class TransformerNet(nn.Module):
  def __init__(self):
    super(TransformerNet, self).__init__()
    # Downsampling
    self.down_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding = 9//2) 
    self.BN_1 = torch.nn.BatchNorm2d(num_features=32)
    self.down_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding = 1)
    self.BN_2 = torch.nn.BatchNorm2d(num_features=64)
    self.down_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding = 1)
    self.BN_3 = torch.nn.BatchNorm2d(num_features=128)
    # Residual connect
    self.res_1 = Residual_block(128)
    self.res_2 = Residual_block(128)
    self.res_3 = Residual_block(128)
    self.res_4 = Residual_block(128)
    self.res_5 = Residual_block(128)
    # Upsampling
    self.up_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding= 1)
    self.BN_4 = torch.nn.BatchNorm2d(num_features=64)
    self.up_2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding = 1, output_padding= 1)
    self.BN_5 = torch.nn.BatchNorm2d(num_features=32)
    self.up_3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=9, stride=1, padding = 9//2)
    self.BN_6 = torch.nn.BatchNorm2d(num_features=3)

    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()
    

  def forward(self, x):
    y = self.relu(self.BN_1(self.down_1(x)))
    # print(y.shape)
    y = self.relu(self.BN_2(self.down_2(y)))
    # print(y.shape)
    y = self.relu(self.BN_3(self.down_3(y)))
    # print(y.shape)

    # print()
    y = self.res_1(y)
    # print(y.shape)
    y = self.res_2(y)
    # print(y.shape)
    y = self.res_3(y)
    # print(y.shape)
    y = self.res_4(y)
    # print(y.shape)
    y = self.res_5(y)
    # print(y.shape)

    # print()
    y = self.relu(self.BN_4(self.up_1(y)))
    # print(y.shape)
    y = self.relu(self.BN_5(self.up_2(y)))
    # print(y.shape)
    y = self.tanh(self.BN_6(self.up_3(y)))
    # print(y.shape)
    return y
