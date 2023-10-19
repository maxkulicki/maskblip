import torch
import torch.nn as nn

class Upsampler(nn.Module):
    def __init__(self, n_clusters=4):
        super(Upsampler, self).__init__()

        # Factor 2 upsampling layer 1
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')

        # 1x1 convolutional layer 1
        self.conv1 = nn.Conv2d(n_clusters, n_clusters, kernel_size=1)

        # ReLU activation
        self.relu = nn.ReLU()

        # Factor 2 upsampling layer 2
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # 1x1 convolutional layer 2
        self.conv2 = nn.Conv2d(n_clusters, n_clusters, kernel_size=1)

        self.upsample_skip = nn.Upsample(scale_factor=4, mode='bilinear')
    def forward(self, x):
        x = x.permute(0,3,1,2)
        x_skip = self.upsample_skip(x)
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.upsample2(x)
        x = self.conv2(x)
        x = torch.softmax(x + x_skip, dim=1)
        return x