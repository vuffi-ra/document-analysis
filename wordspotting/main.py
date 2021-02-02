from __future__ import annotations

import math
import pickle
import os
from collections import defaultdict
from typing import List, Tuple, Dict, Union

import cv2
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

Coords = List[Tuple[int, int]]


class WordspottingSettings:
    """Sift Step Size. Number of pixels between two SIFT descriptors."""
    sift_step_size: int = 5
    """Sift Cell Size. Width and height in pixels of a SIFT descriptor cell. Each descriptor is 4x4 cells large."""
    sift_cell_size: int = 15
    """Width of a heatmap cell."""
    heatmap_grid_width: int = 100
    """Height of a heatmap cell."""
    heatmap_grid_height: int = 100
    """Heatmap threshold. 
    Only patches that contain at least one pixel with a value above the threshold are considered. [0, 1]
    """
    patch_threshold: float = 0.75
    """Patch Step. Determines the step size used when walking the heatmap with possible patches.
    Step in pixels when value above 1, else factor of query patch size.
    """
    patch_step: Tuple[Union[int, float], Union[int, float]] = 1 / 8, 1 / 8
    """Number of entries in the visual dictionary. Equivalent to number of centroids in Lloyd's Algorithm"""
    dictionary_size: int = 100
    """Number of iterations for LLoyd's Algorithm"""
    lloyd_iterations: int = 20
    """Non-Maximum-Supression Threshold. NMS will remove all patches that overlap more than this threshold. [0, 1]"""
    nms_threshold: float = 0.1
    """Determines how the heatmap is interpolated to the image resolution"""
    heatmap_interpolation_method = cv2.INTER_CUBIC
    """Should plots be shown while calculating patches"""
    show_plots: bool = True


class Patch:
    top_left: Tuple[int, int] = (0, 0)
    bottom_right: Tuple[int, int] = (0, 0)
    score: float = 0

    def __init__(self, center_x: int, center_y: int, width: int, height: int, score: float):
        delta_x = width / 2.0
        delta_y = height / 2.0

        self.top_left = (int(center_x - delta_x), int(center_y - delta_y))
        self.bottom_right = (int(center_x + delta_x), int(center_y + delta_y))
        self.score = score

    @staticmethod
    def from_corners(x_1: int, y_1: int, x_2: int, y_2: int, score: float) -> Patch:
        patch = Patch(0, 0, 0, 0, score)
        patch.top_left = (x_1, y_1)
        patch.bottom_right = (x_2, y_2)
        return patch

    def area(self) -> float:
        return (self.bottom_right[0] - self.top_left[0]) * (self.bottom_right[1] - self.top_left[1])

    def intersection(self, other: Patch) -> float:
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

    def union(self, other: Patch) -> float:
        return self.area() + other.area() - self.intersection(other)

    def iou(self, other: Patch) -> float:
        return self.intersection(other) / self.union(other)

    def width(self):
        return self.bottom_right[0] - self.top_left[0]

    def height(self):
        return self.bottom_right[1] - self.top_left[1]

    def contains(self, point: Tuple[int, int]):
        x, y = point

        return self.top_left[0] <= x <= self.bottom_right[0] and self.top_left[1] <= y <= self.bottom_right[1]


def nms(patches: List[Patch], threshold: float) -> List[Patch]:
    patches.sort(key=lambda p: p.score, reverse=True)
    result = []
    while patches:
        best_patch = patches[0]
        patches.remove(best_patch)
        result.append(best_patch)
        tmp = [patch for patch in patches if best_patch.iou(patch) <= threshold]
        patches = tmp

    return result


def inverse_file_structure(frames: Coords, labels) -> Dict[int, Coords]:
    ifs = defaultdict(list)

    for coord, label in zip(frames, labels):
        ifs[label].append(coord)

    return ifs


def gen_heatmap(ifs: Dict[int, Coords], image_width: int, image_height: int, grid_width: int, grid_height: int,
                query_words):
    hmap = np.zeros((math.ceil(image_height / grid_height), math.ceil(image_width / grid_width)))
    for word in query_words:
        for (x, y) in ifs[word]:
            x_grid = math.floor(x / grid_width)
            y_grid = math.floor(y / grid_height)
            hmap[y_grid, x_grid] += 1

    hmap = hmap / np.max(hmap)
    return hmap


def patch_index(a, patch):
    """Indexes the given np array with the given patch"""
    return a[patch.top_left[1]:patch.bottom_right[1], patch.top_left[0]:patch.bottom_right[0]]


def preprocess(frames, descriptors, settings):
    visual_vocabulary, labels = kmeans2(descriptors, settings.dictionary_size, iter=settings.lloyd_iterations,
                                        minit='points')
    ifs = inverse_file_structure(frames, labels)

    return ifs, visual_vocabulary


# query_word is a numpy matrix of the image data
def wordspotting(image_array, frames: Coords, descriptors, ifs: Dict[int, Coords], vocab, query_patch: Patch,
                 settings: WordspottingSettings):
    query_desc = []
    # NOTE: We walk over all descriptors here. We could instead calculate which descriptors lie in the query_patch
    for (p, descriptor) in zip(frames, descriptors):
        if query_patch.contains(p):
            query_desc.append(descriptor)

    # noinspection PyUnresolvedReferences
    query_words = np.argmin(cdist(vocab, np.array(query_desc)), axis=0)

    height, width = image_array.shape
    heatmap = gen_heatmap(ifs, width, height, settings.heatmap_grid_width,
                          settings.heatmap_grid_height, query_words)
    heatmap = cv2.resize(heatmap, dsize=(width, height),
                         interpolation=settings.heatmap_interpolation_method)

    patch_width = query_patch.width()
    patch_height = query_patch.height()
    patches = []

    start_x = int(query_patch.width() / 2.0)
    start_y = int(query_patch.height() / 2.0)

    step_x = int(patch_width * settings.patch_step[0] if settings.patch_step[0] < 1.0 else settings.patch_step[0])
    step_y = int(patch_height * settings.patch_step[1] if settings.patch_step[1] < 1.0 else settings.patch_step[1])

    for x in range(start_x, width, step_x):
        for y in range(start_y, height, step_y):
            patch = Patch(x, y, patch_width, patch_height, 0.0)
            patch_array = patch_index(heatmap, patch)
            hmax = patch_array.max()
            patch.score = patch_array.sum()
            if hmax > settings.patch_threshold:
                patches.append(patch)

    nms_patches = nms(patches, settings.nms_threshold)

    if settings.show_plots:
        plt.imshow(patch_index(image_array, query_patch), cmap="Greys_r")
        plt.show()
        fig = plt.figure()
        heatmap_plot = fig.add_subplot(1, 3, 1)
        threshold_plot = fig.add_subplot(1, 3, 2)
        patch_plot = fig.add_subplot(1, 3, 3)

        heatmap_plot.set_title("Heatmap (Interpolated)")
        heatmap_plot.imshow(image_array, cmap="Greys_r")
        heatmap_plot.imshow(heatmap, cmap="Reds", alpha=0.4)

        heatmap_threshold = np.where(heatmap >= settings.patch_threshold, heatmap, 0.0)
        threshold_plot.set_title("Heatmap (Interpolated, Threshold: %g)" % settings.patch_threshold)
        threshold_plot.imshow(image_array, cmap="Greys_r")
        threshold_plot.imshow(heatmap_threshold, cmap="Reds", alpha=0.4)

        patch_plot.set_title("NMS Patches")
        patch_plot.imshow(image_array, cmap="Greys_r")
        patch_plot.imshow(heatmap, cmap="Reds", alpha=0.4)

        for patch in nms_patches:
            lower_left = (patch.top_left[0], patch.bottom_right[1])
            rect = Rectangle(lower_left, patch.width(), -patch.height(), alpha=0.5)
            patch_plot.add_patch(rect)

        plt.show()

    return nms_patches


def find_relevant_patches(patches: List[Patch], gt: List[Patch]) -> List[int]:
    relevancy_list = []
    for patch in patches:
        relevant = False
        for truth in gt:
            if patch.iou(truth) >= 0.5:
                relevant = True

        relevancy_list.append(1 if relevant else 0)

    return relevancy_list


def analyze(patches: List[Patch], gt: List[Patch]) -> Tuple[float, float, float]:
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


def read_groundtruth(page: str) -> Dict[str, List[Patch]]:
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


def load_page(page: str, settings: WordspottingSettings):
    gt = read_groundtruth(page)

    document_image_filename = os.path.join(os.path.dirname(__file__), 'pages/%s.png' % page)
    image = Image.open(document_image_filename)
    # Fuer spaeter folgende Verarbeitungsschritte muss das Bild mit float32-Werten vorliegen.
    im_arr = np.asarray(image, dtype='float32')

    assert (image.size[0] == im_arr.shape[1])
    assert (image.size[1] == im_arr.shape[0])

    pickle_densesift_fn = 'SIFT/%s-full_dense-%d_sift-%d_descriptors.p' \
                          % (page, settings.sift_step_size, settings.sift_cell_size)
    frames, descriptors = pickle.load(open(pickle_densesift_fn, 'rb'))

    return im_arr, frames, descriptors, gt


def analyze_page(page: str, settings: WordspottingSettings):
    precision, recall, average_precision = 0.0, 0.0, 0.0
    runs = 0
    im_arr, frames, descriptors, gt = load_page(page, settings)
    ifs, vocab = preprocess(frames, descriptors, settings)

    for word in gt:
        for i, patch in enumerate(gt[word]):
            patches = wordspotting(im_arr, frames, descriptors, ifs, vocab, patch, settings)
            run_precision, run_recall, run_average_precision = analyze(patches, gt[word])
            print(word, " (", i + 1, " / ", len(gt[word]), "): Precision: ", run_precision, ", Recall: ", run_recall,
                  ", Average Precision: ", run_average_precision)
            runs += 1
            precision += run_precision
            recall += run_recall
            average_precision += run_average_precision

    return precision / runs, recall / runs, average_precision / runs


def main():
    page = "2700270"
    settings = WordspottingSettings()
    settings.show_plots = False
    precision, recall, average_precision = analyze_page(page, settings)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Average Precision: ", average_precision)
    print("Values averaged over all possible input words")


if __name__ == '__main__':
    main()
