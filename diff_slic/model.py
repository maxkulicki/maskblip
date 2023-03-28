import torch
import torch.nn as nn
from diff_slic.lib.ssn.ssn import ssn


def conv_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )


class SSNModel(nn.Module):
    def __init__(self, nspix=4, n_iter=10, spatial_importance=1):
        super().__init__()
        self.nspix = nspix
        self.n_iter = n_iter
        self.spatial_importance = spatial_importance
        

    def forward(self, x):

        #pixel_f = self.feature_extract(x)
        # if self.training:
        #     return ssn_iter(pixel_f, self.nspix, self.n_iter)
        # else:
        #     return sparse_ssn_iter(pixel_f, self.nspix, self.n_iter)
        x[:, 0, :, :] = x[:, 0, :, :] * self.spatial_importance
        x[:, 1, :, :] = x[:, 1, :, :] * self.spatial_importance
        return ssn(x, self.nspix, self.n_iter)


    def feature_extract(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(s1)
        s3 = self.scale3(s2)

        s2 = nn.functional.interpolate(s2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        s3 = nn.functional.interpolate(s3, size=s1.shape[-2:], mode="bilinear", align_corners=False)

        cat_feat = torch.cat([x, s1, s2, s3], 1)
        feat = self.output_conv(cat_feat)

        return torch.cat([feat, x], 1)
