import torch
import json


def get_init_centroid(images, num_spixels, centroid_data):
    batchsize, channels, height, width = images.shape
    centroids = []
    init_label_maps = []
    for i in range(batchsize):
        init_label_map = torch.tensor(centroid_data[str(num_spixels[i].item())])
        img = images[i]

        c = [torch.mean(img.flatten(1)[:, init_label_map == j], dim=1) for j in range(num_spixels[i])]
        centroids.append(torch.stack(c).T)
        init_label_maps.append(init_label_map)
    init_label_map = torch.cat(init_label_maps, 0)
    init_label_map = init_label_map.reshape(batchsize, -1)
    return centroids, init_label_map




def calc_init_centroid(images, num_spixels_width, num_spixels_height):
    """
    calculate initial superpixels

    Args:
        images: torch.Tensor
            A Tensor of shape (B, C, H, W)
        spixels_width: int
            initial superpixel width
        spixels_height: int
            initial superpixel height

    Return:
        centroids: torch.Tensor
            A Tensor of shape (B, C, H * W)
        init_label_map: torch.Tensor
            A Tensor of shape (B, H * W)
        num_spixels_width: int
            A number of superpixels in each column
        num_spixels_height: int
            A number of superpixels int each raw
    """
    if 0 in num_spixels_height:
        print("num_spixels_height is 0")

    batchsize, channels, height, width = images.shape
    device = images.device
    centroids = []
    init_label_maps = []
    for i in range(batchsize):
        c = torch.nn.functional.adaptive_avg_pool2d(images[i], (num_spixels_height[i], num_spixels_width[i]))
        centroids.append(c.flatten(1))
        num_spixels = num_spixels_width[i] * num_spixels_height[i]
        labels = torch.arange(int(num_spixels), device=device).reshape(1, 1, *c.shape[-2:]).type_as(c)
        init_label_map = torch.nn.functional.interpolate(labels, size=(height, width), mode="nearest")
        init_label_maps.append(init_label_map)
    #centroids = torch.cat(centroids, 0)
    init_label_map = torch.cat(init_label_maps, 0)

    # with torch.no_grad():
    #     num_spixels = num_spixels_width * num_spixels_height
    #     labels = torch.arange(int(num_spixels), device=device).reshape(1, 1, *centroids.shape[-2:]).type_as(centroids)
    #     init_label_map = torch.nn.functional.interpolate(labels, size=(height, width), mode="nearest")
    #     init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)

    init_label_map = init_label_map.reshape(batchsize, -1)
    #centroids = centroids.reshape(batchsize, channels, -1)

    return centroids, init_label_map


@torch.no_grad()
def get_abs_indices(init_label_map, num_spixels_width):
    b, n_pixel = init_label_map.shape
    device = init_label_map.device
    r = torch.arange(-1, 2.0, device=device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)

    abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1).long()
    abs_spix_indices = (init_label_map[:, None] + relative_spix_indices[None, :, None]).reshape(-1).long()
    abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1).long()

    return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)


@torch.no_grad()
def get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width):
    relative_label = affinity_matrix.max(1)[1]
    r = torch.arange(-1, 2.0, device=affinity_matrix.device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)
    label = init_label_map + relative_spix_indices[relative_label]
    return label.long()

def ssn(pixel_features, num_spixels, n_iter, init_data, training=False):
    pixel_features = pixel_features.permute(0, 3, 1, 2)
    height, width = pixel_features.shape[-2:]
    spixel_features, init_label_map = get_init_centroid(pixel_features, num_spixels, init_data)
    s = torch.sqrt(height * width / num_spixels).int()

    pixel_features = pixel_features.flatten(start_dim=2).permute(0,2,1)
    results = []
    for idx, n in enumerate(n_iter):
        for i in range(n):
            feature_dist_matrix = torch.cdist(pixel_features[idx, :, 2:].clone(), spixel_features[idx][2:, :].T.clone())
            spatial_dist_matrix = torch.cdist(pixel_features[idx, :, :2].clone(), spixel_features[idx][:2, :].T.clone())
            dist_matrix = feature_dist_matrix + spatial_dist_matrix / s[idx]
            affinity_matrix = (-dist_matrix).softmax(1).unsqueeze(0)
            spixel_features_new = torch.bmm(affinity_matrix.permute(0, 2, 1),
                                            pixel_features[idx, :, :].unsqueeze(0)).permute(0, 2, 1)
            spixel_features_sum = affinity_matrix.sum(1)
            spixel_features[idx] = spixel_features_new.squeeze().clone() / spixel_features_sum.clone()

        if training:
            results.append(affinity_matrix.squeeze())
        else:
            hard_labels = affinity_matrix.argmax(2).unflatten(1,(width,height))
            results.append(hard_labels)

    #results = torch.stack(results, 1)
    return results

#get_initial_centroids(6, 24, 24)