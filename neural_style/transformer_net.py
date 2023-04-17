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
    self.conv_1 = nn.Conv2d(in_channels=channel, out_channels=channel,
                            padding='same', kernel_size=3, stride=1)
    self.inst1 = nn.InstanceNorm2d(channel, affine=True)
    self.conv_2 = nn.Conv2d(in_channels=channel, out_channels=channel,
                            padding='same', kernel_size=3, stride=1)
    self.inst2 = nn.InstanceNorm2d(channel, affine=True)
    self.relu = nn.ReLU()

  def forward(self, x):
    residual = x
    out = self.relu(self.inst1(self.conv_1(x)))
    out = self.inst2(self.conv_2(out))
    return self.relu(out + residual)


class TransformerNet(nn.Module):
  def __init__(self):
    super(TransformerNet, self).__init__()
    # Downsampling
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding=9 // 2)
    self.BN_1 = nn.InstanceNorm2d(num_features=32, affine=True)
    self.down_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
    self.BN_2 = nn.InstanceNorm2d(num_features=64, affine=True)
    self.down_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
    self.BN_3 = nn.InstanceNorm2d(num_features=128, affine=True)
    # Residual connect
    self.res_1 = Residual_block(128)
    self.res_2 = Residual_block(128)
    self.res_3 = Residual_block(128)
    self.res_4 = Residual_block(128)
    self.res_5 = Residual_block(128)
    # Upsampling
    self.up_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                   output_padding=1)
    self.BN_4 = nn.InstanceNorm2d(num_features=64, affine=True)
    self.up_2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1,
                                   output_padding=1)
    self.BN_5 = nn.InstanceNorm2d(num_features=32, affine=True)
    # self.up_3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=9, stride=1, padding = 9//2)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=9, stride=1, padding=9 // 2)

    self.relu = nn.ReLU()

  def forward(self, x):
    y = self.relu(self.BN_1(self.conv1(x)))
    # print(y.shape)
    y = self.relu(self.BN_2(self.down_1(y)))
    # print(y.shape)
    y = self.relu(self.BN_3(self.down_2(y)))
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
    y = self.conv2(y)
    # print(y.shape)
    return y
