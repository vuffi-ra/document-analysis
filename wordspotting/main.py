import math
import pickle
import os
import cv2
import numpy as np
import tqdm
from collections import defaultdict

import PIL.Image as Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

from wordspotting.SIFT.compute_sift import compute_sift_descriptors

show_plots = True


class Patch:
    top_left = (0, 0)
    bottom_right = (0, 0)
    score = 0

    def __init__(self, center_x, center_y, width, height, score):
        delta_x = width / 2.0
        delta_y = height / 2.0

        self.top_left = (center_x - delta_x, center_y - delta_y)
        self.bottom_right = (center_x + delta_x, center_y + delta_y)
        self.score = score

    @staticmethod
    def from_corners(x_1, y_1, x_2, y_2, score):
        patch = Patch(0, 0, 0, 0, score)
        patch.top_left = (x_1, y_1)
        patch.bottom_right = (x_2, y_2)
        return patch

    def area(self):
        return (self.bottom_right[0] - self.top_left[0]) * (self.bottom_right[1] - self.top_left[1])

    def intersection(self, other):
        left = max(self.top_left[0], other.top_left[0])
        right = min(self.bottom_right[0], other.bottom_right[0])
        top = max(self.top_left[1], other.top_left[1])
        bottom = min(self.bottom_right[1], other.bottom_right[1])
        width = (right - left)
        height = (bottom - top)
        if width < 0.0 or height < 0.0:
            # No overlap
            return 0.0
        else:
            return width * height

    def union(self, other):
        return self.area() + other.area() - self.intersection(other)

    def iou(self, other):
        return self.intersection(other) / self.union(other)


def nms(patches, threshold):
    patches.sort(key=lambda p: p.score, reverse=True)
    result = []
    while patches:
        best_patch = patches[0]
        patches.remove(best_patch)
        result.append(best_patch)
        tmp = [patch for patch in patches if best_patch.iou(patch) <= threshold]
        patches = tmp

    return result


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


def show_patches(image, patches, heatmap):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap="Greys_r")
    ax.imshow(heatmap, cmap="Reds", alpha=0.4)

    for patch in patches:
        lower_left = (patch.top_left[0], patch.bottom_right[1])
        width = (patch.bottom_right[0] - patch.top_left[0])
        height = (patch.bottom_right[1] - patch.top_left[1])
        rect = Rectangle(lower_left, width, -height, alpha=0.5)
        ax.add_patch(rect)

    plt.show()


# query_word is a numpy matrix of the image data
def wordspotting(page, query_word, step_size, cell_size, grid_width, grid_height, threshold):

    document_image_filename = os.path.join(os.path.dirname(__file__), 'pages/%s.png' % page)
    image = Image.open(document_image_filename)
    # Fuer spaeter folgende Verarbeitungsschritte muss das Bild mit float32-Werten vorliegen.
    im_arr = np.asarray(image, dtype='float32')

    pickle_densesift_fn = 'SIFT/%s-full_dense-%d_sift-%d_descriptors.p' % (page, step_size, cell_size)
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

    start_x = int(patch_width / 2.0)
    start_y = int(patch_height / 2.0)

    step_x = int(patch_width / 8.0)
    step_y = int(patch_height / 8.0)

    for x in range(start_x, im_arr.shape[1], step_x):
        for y in range(start_y, im_arr.shape[0], step_y):
            patch = Patch(x, y, patch_width, patch_height, 0.0)
            hmax = heatmap_interpolated[int(patch.top_left[1]):int(patch.bottom_right[1]),
                          int(patch.top_left[0]):int(patch.bottom_right[0])].max()
            patch.score = heatmap_interpolated[int(patch.top_left[1]):int(patch.bottom_right[1]),
                          int(patch.top_left[0]):int(patch.bottom_right[0])].sum()
            if hmax > threshold:
                patches.append(patch)

    print("Patches found")

    nms_patches = nms(patches, 0.1)

    if show_plots:
        show_patches(im_arr, nms_patches, heatmap_threshold)

    print("Done")
    return nms_patches


def find_relevant_patches(patches, gt):
    relevancy_list = []
    for patch in patches:
        relevant = False
        for truth in gt:
            if patch.iou(truth) >= 0.5:
                relevant = True

        relevancy_list.append(1 if relevant else 0)

    return relevancy_list


def analyze(patches, gt):
    if not patches or not gt:
        return 0.0, 0.0, 0.0
    relevancy_list = find_relevant_patches(patches, gt)
    precisions = []
    for i in range(1, len(relevancy_list) + 1):
        precisions.append(sum(relevancy_list[0:i]) / i)

    precision = precisions[-1]
    recall = sum(relevancy_list) / len(gt)
    average_precision = 0
    for i in range(len(precisions)):
        average_precision += precisions[i] if relevancy_list[i] else 0.0
    average_precision /= len(gt)

    return precision, recall, average_precision


def read_groundtruth(page):
    gt_file = os.path.join(os.path.dirname(__file__), 'GT/%s.gtp' % page)
    result = defaultdict(list)

    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.split(' ')
            x_1 = int(parts[0])
            y_1 = int(parts[1])
            x_2 = int(parts[2])
            y_2 = int(parts[3])
            word = parts[4].strip()
            result[word].append(Patch.from_corners(x_1, y_1, x_2, y_2, 0.0))

    return result


def main():
    page = "2700270"
    step_size = 5
    cell_size = 15

    grid_height = 100
    grid_width = 100

    threshold = 0.8

    input = cv2.imread("inputs/the.png")
    patches = wordspotting(page, input, step_size, cell_size, grid_width, grid_height, threshold)
    gt = read_groundtruth(page)

    precision, recall, average_precision = analyze(patches, gt["the"])
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Average Precision: ", average_precision)


if __name__ == '__main__':
    main()
