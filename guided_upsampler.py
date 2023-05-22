import torch
import torch.nn as nn
import torch.nn.functional as F

#extracting features from the low-resolution clusters
class CNN_T(nn.Module):
    def __init__(self, n_clusters):
        super(CNN_T, self).__init__()

        self.conv1 = nn.Conv2d(n_clusters, 96, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 48, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 24, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.upsample(x)
        return x

#extracting image features for guidance
class CNN_G(nn.Module):
    def __init__(self):
        super(CNN_G, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 48, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 24, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        return x

#combining the features from the low-resolution clusters and the image features
class AttentionUpsample(nn.Module):
    def __init__(self, in_dim):
        super(AttentionUpsample, self).__init__()
        self.cnn_m = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.attention = nn.MultiheadAttention(in_dim//2, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self, cluster_features, image_features):
        x = torch.cat([cluster_features, image_features], dim=1)
        x = self.cnn_m(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x_size = x.shape[2]
        batch_size = x.shape[0]
        q, k = torch.split(x, x.shape[1]//2, dim=1)
        v = cluster_features
        #x, _ = self.attention(q, k, v)
        def prep(x):
            return x.flatten(2).squeeze().transpose(-1, -2)
        x, _ = self.attention(prep(q), prep(k), prep(v))
        x = torch.relu(x)
        x = torch.reshape(x, (batch_size, x_size, x_size,-1)).permute(0,3,1,2)
        return x

#final convolution
class CNN_F(nn.Module):
    def __init__(self, out_dim=10):
        super(CNN_F, self).__init__()
        self.conv1 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, out_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        return x

#combine the modules above to form the guided upsampler
class GuidedUpsampler(nn.Module):
    def __init__(self, n_clusters=10):
        super(GuidedUpsampler, self).__init__()
        self.cnn_t = CNN_T(n_clusters)
        self.cnn_g = CNN_G()
        self.attention_upsample1 = AttentionUpsample(48)
        self.attention_upsample2 = AttentionUpsample(48)
        self.cnn_f = CNN_F(n_clusters)
        self.upsample_skip = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, cluster_features, image):
        image = F.interpolate(image, size=(96,96), mode='nearest')
        cluster_features = cluster_features.permute(0, 3, 1, 2)
        x_skip = self.upsample_skip(cluster_features)
        x = self.cnn_t(cluster_features)
        im_features = self.cnn_g(image)
        im_features_ds = F.interpolate(im_features, scale_factor=0.5, mode='nearest')
        x = self.attention_upsample1(x, im_features_ds)
        x = nn.Upsample(scale_factor=2, mode='bilinear')(x)
        x = self.attention_upsample2(x, im_features)
        x = self.cnn_f(x)
        x = torch.softmax(x + x_skip, dim=1)
        return x + x_skip

if __name__ == '__main__':
    # Test guided upsampler
    upsampler = GuidedUpsampler()
    cluster_features = torch.randn(1, 10, 24, 24)
    image_features = torch.randn(1, 3, 96, 96)
    output = upsampler(cluster_features, image_features)
    print(output.shape)