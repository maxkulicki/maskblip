import json
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def get_initial_centroids(n, num_spixels_width, num_spixels_height, dict):
    points = list(dict[str(n)].values())
    points = [[int(p[0]*num_spixels_height), int(p[1]*num_spixels_width)] for p in points]
    return points

def closest_points(points, n):
    # Generate the grid points
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    grid_points = np.column_stack((x.ravel(), y.ravel()))

    # Compute the pairwise distances between the grid points and the points in the first list
    distances = cdist(points, grid_points)

    # Get the index of the closest point for each grid point
    closest_indices = np.argmin(distances, axis=0)

    return closest_indices

def make_json():
    dict = json.load(open("cluster_init.json"))
    result = {}
    for i in range(1,26):
        grid = closest_points(get_initial_centroids(i, 24, 24, dict), 24)
        result[i] = grid.tolist()
        plt.imshow(grid.reshape(24,24))
        plt.show()
    with open("../../centroid_data.dict", "w") as f:
        json.dump(result, f)

make_json()