import torch
import torch.nn as nn
import torch.nn.functional as F

#extracting features from the low-resolution clusters
class CNN_T(nn.Module):
    def __init__(self, n_clusters):
        super(CNN_T, self).__init__()

        self.conv1 = nn.Conv2d(n_clusters, 96, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(96, 48, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(48, 24, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = self.upsample(x)
        return x

#extracting image features for guidance
class CNN_G(nn.Module):
    def __init__(self):
        super(CNN_G, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(96, 48, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(48, 24, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=0.5, mode='nearest')
        return x

#combining the features from the low-resolution clusters and the image features
class AttentionUpsample(nn.Module):
    def __init__(self, in_dim):
        super(AttentionUpsample, self).__init__()
        self.cnn_m = nn.Conv2d(in_dim, in_dim*2, kernel_size=1)
        self.attention = nn.MultiheadAttention(in_dim, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self, cluster_features, image_features):
        x = torch.cat([cluster_features, image_features], dim=1)
        x = self.cnn_m(x)
        q, k = torch.split(x, x.shape[1]//2, dim=1)
        v = cluster_features
        x, _ = self.attention(q, k, v)
        x = self.upsample(x)
        return x

#final convolution
class CNN_F(nn.Module):
    def __init__(self):
        super(CNN_F, self).__init__()
        self.conv1 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x

#combine the modules above to form the guided upsampler
class GuidedUpsampler(nn.Module):
    def __init__(self):
        super(GuidedUpsampler, self).__init__()
        self.cnn_t = CNN_T(10)
        self.cnn_g = CNN_G()
        self.attention_upsample1 = AttentionUpsample(48)
        self.attention_upsample2 = AttentionUpsample(48)
        self.cnn_f = CNN_F()

    def forward(self, cluster_features, image_features):
        x = self.cnn_t(cluster_features)
        y = self.cnn_g(image_features)
        x = self.attention_upsample1(x, y)
        x = self.attention_upsample2(x, y)
        x = self.cnn_f(x)
        return x

if __name__ == '__main__':
    # Test guided upsampler
    upsampler = GuidedUpsampler()
    cluster_features = torch.randn(1, 10, 64, 64)
    image_features = torch.randn(1, 3, 256, 256)
    output = upsampler(cluster_features, image_features)
    print(output.shape)
