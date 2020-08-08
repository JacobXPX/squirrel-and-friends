
""" Predict with Test Time Augmentation (TTA)
   Additional to the original test/validation images, apply image augmentation to them
   (just like for training images). The intent
   is to increase the accuracy of predictions by examining the images using multiple
   perspectives.
"""

from itertools import product

import numpy as np


class horizontalFlip:
    def __init__(self, image_size):
        self.image_size = image_size

    def augment(self, image):
        return image.flip(1)

    def batch_augment(self, images):
        return images.flip(2)

    def deaugment_boxes(self, boxes):
        boxes[:, [1, 3]] = self.image_size - boxes[:, [3, 1]]
        return boxes


class verticalFlip:
    def __init__(self, image_size):
        self.image_size = image_size

    def augment(self, image):
        return image.flip(2)

    def batch_augment(self, images):
        return images.flip(3)

    def deaugment_boxes(self, boxes):
        boxes[:, [0, 2]] = self.image_size - boxes[:, [2, 0]]
        return boxes


class rotate90:
    def __init__(self, image_size):
        self.image_size = image_size

    def augment(self, image):
        return np.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return np.rot90(images, 1, (2, 3))

    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0, 2]] = self.image_size - boxes[:, [1, 3]]
        res_boxes[:, [1, 3]] = boxes[:, [2, 0]]
        return res_boxes


class ttaCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def augment(self, image):
        res_images = image.copy()
        for transform in self.transforms:
            image = transform.augment(res_images)
        return res_images

    def batch_augment(self, images):
        res_images = images.copy()
        for transform in self.transforms:
            images = transform.batch_augment(res_images)
        return res_images

    def prepare_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, 0] = np.min(boxes[:, [0, 2]], axis=1)
        res_boxes[:, 2] = np.max(boxes[:, [0, 2]], axis=1)
        res_boxes[:, 1] = np.min(boxes[:, [1, 3]], axis=1)
        res_boxes[:, 3] = np.max(boxes[:, [1, 3]], axis=1)
        return res_boxes

    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        for transform in self.transforms[::-1]:
            res_boxes = transform.deaugment_boxes(res_boxes)
        return self.prepare_boxes(res_boxes)


def get_tta_transformers(image_size):
    """
    Get transformers of TTA*8, 8 possible perspectives of an image.

    Args:
        image_size (int): image size.

    Returns:
        tta_transforms(list(ttaCompose)): list of 8 ttaCompose.
    """

    tta_transforms = []
    for combo in product([horizontalFlip(image_size), None],
                         [verticalFlip(image_size), None],
                         [rotate90(image_size), None]
                         ):
        tta_transforms.append(ttaCompose([trans for trans in combo if trans]))
    return tta_transforms
