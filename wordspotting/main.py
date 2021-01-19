import math
import pickle
import os
import cv2
import numpy as np
from collections import defaultdict

import PIL.Image as Image
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

from wordspotting.SIFT.compute_sift import compute_sift_descriptors

show_plots = False


class Patch:
    top_left = (0, 0)
    bottom_right = (0, 0)

    def __init__(self, center_x, center_y, width, height):
        delta_x = width / 2.0
        delta_y = height / 2.0

        self.top_left = (center_x - delta_x, center_y - delta_y)
        self.bottom_right = (center_x + delta_x, center_y + delta_y)


def inverse_file_structure(frames, labels):
    ifs = defaultdict(list)

    for coord, label in zip(frames, labels):
        ifs[label].append(coord)

    return ifs


def gen_heatmap(ifs, image_width, image_height, grid_width, grid_height, query_words):
    map = np.zeros((math.ceil(image_height / grid_height), math.ceil(image_width / grid_width)))
    for word in query_words:
        for (x, y) in ifs[word]:
            x_grid = math.floor(x / grid_width)
            y_grid = math.floor(y / grid_height)
            map[y_grid, x_grid] += 1
    return map


# query_word is a numpy matrix of the image data
def wordspotting(query_word, step_size, cell_size, grid_width, grid_height, threshold):

    document_image_filename = os.path.join(os.path.dirname(__file__), 'pages/2700270.png')
    image = Image.open(document_image_filename)
    # Fuer spaeter folgende Verarbeitungsschritte muss das Bild mit float32-Werten vorliegen.
    im_arr = np.asarray(image, dtype='float32')

    pickle_densesift_fn = 'SIFT/2700270-full_dense-%d_sift-%d_descriptors.p' % (step_size, cell_size)
    frames, descriptors = pickle.load(open(pickle_densesift_fn, 'rb'))

    n_centroids = 100
    visual_words, labels = kmeans2(descriptors, n_centroids, iter=20, minit='points')

    ifs = inverse_file_structure(frames, labels)

    query_frames, query_desc = compute_sift_descriptors(query_word, cell_size, step_size)

    query_words = np.argmin(cdist(visual_words, query_desc), axis=0)

    # Find image size
    heatmap = gen_heatmap(ifs, im_arr.shape[1], im_arr.shape[0], grid_width, grid_height, query_words)

    heatmap_normalized = heatmap / np.max(heatmap)
    heatmap_threshold = np.where(heatmap_normalized >= threshold, heatmap_normalized, 0.0)

    heatmap_threshold = cv2.resize(heatmap_threshold, dsize=(image.width, image.height), interpolation=cv2.INTER_CUBIC)
    heatmap_interpolated = cv2.resize(heatmap_normalized, dsize=(image.width, image.height), interpolation=cv2.INTER_CUBIC)

    if show_plots:
        plt.imshow(heatmap_interpolated)
        plt.show()
        plt.imshow(image)
        plt.imshow(heatmap_threshold, alpha=0.5)
        plt.show()
        plt.imshow(image)
        plt.imshow(heatmap_interpolated, alpha=0.5)
        plt.show()

    patch_width = query_word.shape[1]
    patch_height = query_word.shape[0]
    patches = []

    for (x, y), v in np.ndenumerate(heatmap_interpolated):
        if v > threshold:
            patches.append(Patch(x, y, patch_width, patch_height))

    print("Done")


def main():
    step_size = 5
    cell_size = 15

    grid_height = 100
    grid_width = 100

    threshold = 0.75

    input = cv2.imread("inputs/the.png")
    wordspotting(input, step_size, cell_size, grid_width, grid_height, threshold)


if __name__ == '__main__':
    main()
