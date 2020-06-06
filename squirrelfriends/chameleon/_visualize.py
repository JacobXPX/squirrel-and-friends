import glob
import os
from math import ceil

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_image_list(path, extension="jpg"):
    """Get the list of images in the path.

    Args:
        path (str): the path of folder with images.
        extension (str): the format of image file.

    Returns:
        img_lst (list of str): image paths.
    """

    img_lst = [os.path.basename(name) for name in glob.glob(
        os.path.join(path, "*.%s" % extension))]

    return img_lst


def draw_rectangles(img, boxes, color=[255, 0, 0]):
    """Draw rectangles on image.

    Args:
        img (array): image.
        boxes (list): list of top, left, bottom, right line.
        color (list): color of box. tuple of RGB.

    Returns:
        img (array): image with rectangles.
    """

    img = img.copy()
    boxes = np.array(boxes)

    for bbox in boxes:
        top_left = (int(bbox[0]), int(bbox[1]))
        bottom_right = (int(bbox[2]), int(bbox[3]))
        thickness = int(max(img.shape[:2]) / 200)
        img = cv2.rectangle(img, top_left, bottom_right, color, thickness)

    return img


def imgs_plot(imgs, ncols, titles, suptitle=None):
    """Plot multiple images.

    Args:
        imgs (list of :obj:): list of img arrays.
        ncols (int): number of columns of subplot.
        tiltes (list): list of titiles for each subplot.
        suptitle (str): suptitle of plot.

    Returns:
        grid (matplotlib Axes): axes object with the distplot.
    """

    grid, ax = plt.subplots(figsize=(20, 15),
                            nrows=ceil(len(imgs) / ncols),
                            ncols=ncols)
    if suptitle is not None:
        grid.suptitle(suptitle, fontsize=30)

    grid.subplots_adjust(wspace=0.3)
    grid.subplots_adjust(hspace=0.3)

    for i, (img, title) in enumerate(zip(imgs, titles)):
        ax[i // ncols][i % ncols].imshow(img)
        ax[i // ncols][i % ncols].set_title(title, fontsize=15)

    return grid
