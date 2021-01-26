import math
import pickle
import os
from collections import defaultdict

import cv2
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

from wordspotting.SIFT.compute_sift import compute_sift_descriptors


class WordspottingSettings:
    """Sift Step Size. Number of pixels between two SIFT descriptors."""
    sift_step_size = 5
    """Sift Cell Size. Width and height in pixels of a SIFT descriptor cell. Each descriptor is 4x4 cells large."""
    sift_cell_size = 15
    """Width of a heatmap cell."""
    heatmap_grid_width = 100
    """Height of a heatmap cell."""
    heatmap_grid_height = 100
    """Heatmap threshold. 
    Only patches that contain at least one pixel with a value above the threshold are considered.
    """
    patch_threshold = 0.75
    """Number of entries in the visual dictionary. Equivalent to number of centroids in Lloyd's Algorithm"""
    dictionary_size = 100
    """Number of iterations for LLoyd's Algorithm"""
    lloyd_iterations = 20
    """Non-Maximum-Supression Threshold. NMS will remove all patches that overlap more than this threshold"""
    nms_threshold = 0.1
    """Determines how the heatmap is interpolated to the image resolution"""
    heatmap_interpolation_method = cv2.INTER_CUBIC
    """Should plots be shown while calculating patches"""
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
    hmap = np.zeros((math.ceil(image_height / grid_height), math.ceil(image_width / grid_width)))
    for word in query_words:
        for (x, y) in ifs[word]:
            x_grid = math.floor(x / grid_width)
            y_grid = math.floor(y / grid_height)
            hmap[y_grid, x_grid] += 1

    hmap = hmap / np.max(hmap)
    return hmap


# query_word is a numpy matrix of the image data
def wordspotting(page, query_word, settings: WordspottingSettings):
    document_image_filename = os.path.join(os.path.dirname(__file__), 'pages/%s.png' % page)
    image = Image.open(document_image_filename)
    # Fuer spaeter folgende Verarbeitungsschritte muss das Bild mit float32-Werten vorliegen.
    im_arr = np.asarray(image, dtype='float32')

    pickle_densesift_fn = 'SIFT/%s-full_dense-%d_sift-%d_descriptors.p' \
                          % (page, settings.sift_step_size, settings.sift_cell_size)
    frames, descriptors = pickle.load(open(pickle_densesift_fn, 'rb'))

    visual_words, labels = kmeans2(descriptors, settings.dictionary_size, iter=settings.lloyd_iterations,
                                   minit='points')

    ifs = inverse_file_structure(frames, labels)

    query_frames, query_desc = compute_sift_descriptors(query_word, settings.sift_cell_size, settings.sift_step_size)

    # noinspection PyUnresolvedReferences
    query_words = np.argmin(cdist(visual_words, query_desc), axis=0)

    heatmap = gen_heatmap(ifs, im_arr.shape[1], im_arr.shape[0], settings.heatmap_grid_width,
                          settings.heatmap_grid_height, query_words)
    heatmap = cv2.resize(heatmap, dsize=(image.width, image.height),
                         interpolation=settings.heatmap_interpolation_method)

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
            hmax = heatmap[int(patch.top_left[1]):int(patch.bottom_right[1]),
                           int(patch.top_left[0]):int(patch.bottom_right[0])].max()
            patch.score = heatmap[int(patch.top_left[1]):int(patch.bottom_right[1]),
                                  int(patch.top_left[0]):int(patch.bottom_right[0])].sum()
            if hmax > settings.patch_threshold:
                patches.append(patch)

    nms_patches = nms(patches, settings.nms_threshold)

    if settings.show_plots:
        fig = plt.figure()
        heatmap_plot = fig.add_subplot(1, 3, 1)
        threshold_plot = fig.add_subplot(1, 3, 2)
        patch_plot = fig.add_subplot(1, 3, 3)

        heatmap_plot.set_title("Heatmap (Interpolated)")
        heatmap_plot.imshow(image, cmap="Greys_r")
        heatmap_plot.imshow(heatmap, cmap="Reds", alpha=0.4)

        heatmap_threshold = np.where(heatmap >= settings.patch_threshold, heatmap, 0.0)
        threshold_plot.set_title("Heatmap (Interpolated, Threshold: %g)" % settings.patch_threshold)
        threshold_plot.imshow(image, cmap="Greys_r")
        threshold_plot.imshow(heatmap_threshold, cmap="Reds", alpha=0.4)

        patch_plot.set_title("NMS Patches")
        patch_plot.imshow(image, cmap="Greys_r")
        patch_plot.imshow(heatmap, cmap="Reds", alpha=0.4)

        for patch in nms_patches:
            lower_left = (patch.top_left[0], patch.bottom_right[1])
            width = (patch.bottom_right[0] - patch.top_left[0])
            height = (patch.bottom_right[1] - patch.top_left[1])
            rect = Rectangle(lower_left, width, -height, alpha=0.5)
            patch_plot.add_patch(rect)

        plt.show()

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

    input_word = cv2.imread("inputs/the.png")
    patches = wordspotting(page, input_word, WordspottingSettings())
    gt = read_groundtruth(page)

    precision, recall, average_precision = analyze(patches, gt["the"])
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Average Precision: ", average_precision)


if __name__ == '__main__':
    main()
