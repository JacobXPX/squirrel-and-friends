import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class datasetRetriever(Dataset):
    """Retrieve Dataset.

    Attributes:
        image_path
        image_ids
        transforms
        phase (str): "training", "validation", "testing"
        bbox: The `pascal_voc` format: `[x_min, y_min, x_max, y_max]`
    """

    def __init__(self,
                 image_path,
                 image_ids,
                 transforms=None,
                 origin_mosaic_mixup_prob=[0.6, 0.3, 0.1],
                 phase="training",
                 bboxes=None,
                 llabels=None):
        super().__init__()

        self.image_path = image_path
        self.image_ids = image_ids
        self.transforms = transforms
        self.origin_mosaic_mixup_prob = np.array(
            origin_mosaic_mixup_prob) / sum(origin_mosaic_mixup_prob)
        self.phase = phase
        self.bboxes = bboxes
        self.llabels = llabels
        self.image_size = self.load_image(0)[:2]

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        if self.phase == "testing":
            image = self.load_image_only(index)
            return image, image_id

        elif self.phase == "validation":
            image, boxes, labels = self.load_image_and_boxes(index)
            target = self.generate_target(index, image, boxes, labels)
            return image, target, image_id

        elif self.phase == "training":
            p = random.random()

            if p <= self.origin_mosaic_mixup_prob[0]:
                image, boxes, labels = self.load_image_and_boxes(index)
            elif p <= sum(self.origin_mosaic_mixup_prob[0:2]):
                image, boxes, labels = self.load_mosaic_image_and_boxes(index)
            else:
                image, boxes, labels = self.load_mixup_image_and_boxes(index)

            target = self.generate_target(index, image, boxes, labels)
            return image, target, image_id
        else:
            raise Exception("please set a right phase")

        return image, target, image_id

    def generate_target(self, index, image, boxes, labels):
        image, boxes, labels = self.apply_transform(image, boxes, labels)
        target = {}
        target["image_id"] = torch.tensor([index])
        if boxes is not None:
            target["boxes"] = boxes
        if labels is not None:
            target["labels"] = labels
        return target

    def apply_transform(self, image, boxes, labels):
        if self.transforms:
            if boxes is not None and len(boxes) > 0:
                for _ in range(10):
                    sample = self.transforms(**{
                        "image": image,
                        "bboxes": boxes,
                        "labels": labels
                    })
                    if len(sample["bboxes"]) > 0:
                        image = sample["image"]
                        boxes = torch.stack(
                            tuple(map(torch.tensor, zip(*sample["bboxes"])))).permute(1, 0)
                        # boxes[:, [0, 1, 2, 3]] = target["boxes"][:, [
                        #     1, 0, 3, 2]]  # yxyx: be warning
                        labels = torch.stack(
                            tuple(map(torch.tensor, zip(*sample["labels"])))).permute(1, 0)
                        break
            else:
                boxes = None
                if labels is not None:
                    sample = self.transforms(**{
                        "image": image,
                        "labels": labels
                    })
                    image = sample["image"]
                    labels = torch.tensor(sample["labels"])
                else:
                    labels = None
                    sample = self.transforms(**{
                        "image": image,
                    })
                    image = sample["image"]

        return image, boxes, labels

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image(self, index):
        image_id = self.image_ids[index]
        image_id = (image_id if "pseudo_" not in image_id
                    else image_id.replace("pseudo_", ""))
        image = cv2.imread(
            f"{self.image_path}/{image_id}.jpg", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        return image

    def load_image_only(self, index):
        image = self.load_image(index)
        if self.transforms:
            sample = {"image": image}
            sample = self.transforms(**sample)
            image = sample["image"]
        return image

    def load_image_and_boxes(self, index):

        image_id = self.image_ids[index]
        image = self.load_image(index)

        if self.bboxes is not None:
            bbox = self.bboxes[self.bboxes["image_id"] == image_id]
            boxes = bbox[["x_min", "y_min", "x_max", "y_max"]].values
        else:
            boxes = None

        if self.llabels is not None:
            llabel = self.llabels[self.llabels["image_id"] == image_id]
            labels = llabel[["label"]].values
        else:
            labels = None

        return image, boxes, labels

    def load_mosaic_image_and_boxes(self, index):

        w, h = self.image_size[0], self.image_size[1]

        xc, yc = [int(random.uniform(w * 0.25, h * 0.75))
                  for _ in range(2)]  # center x, y
        indexes = [
            index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full(
            (w, h, 3), 1, dtype=np.float32)

        result_boxes = []
        result_labels = []

        for i, index in enumerate(indexes):
            image, boxes, labels = self.load_image_and_boxes(index)
            if i == 0:  # bottom left <- top right
                x1a, y1a, x2a, y2a = 0, 0, xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # bottom right <- top left
                x1a, y1a, x2a, y2a = xc, 0, w, yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), x2a - x1a, h
            elif i == 2:  # top left <- bottom right
                x1a, y1a, x2a, y2a = 0, yc, xc, h
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, y2a - y1a
            elif i == 3:  # top right <- bottom left
                x1a, y1a, x2a, y2a = xc, yc, w, h
                x1b, y1b, x2b, y2b = 0, 0, x2a - x1a, y2a - y1a

            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            if boxes is not None:
                padw = x1a - x1b
                padh = y1a - y1b
                boxes[:, 0] += padw
                boxes[:, 1] += padh
                boxes[:, 2] += padw
                boxes[:, 3] += padh
                result_boxes.append(boxes)

            if labels is not None:
                result_labels.append(labels)

        if result_boxes:
            result_boxes = np.concatenate(result_boxes, 0)
            result_labels = np.concatenate(result_labels, 0)

            # correct boxes
            np.clip(result_boxes[:, [0, 2]], 0,
                    self.image_size[0], out=result_boxes[:, [0, 2]])
            np.clip(result_boxes[:, [1, 3]], 0,
                    self.image_size[1], out=result_boxes[:, [1, 3]])
            result_boxes = result_boxes.astype(np.int32)

            valid_boxes = (result_boxes[:, 2]-result_boxes[:, 0]) * \
                (result_boxes[:, 3]-result_boxes[:, 1]) > 0
            result_boxes = result_boxes[np.where(valid_boxes)]
            result_labels = result_labels[np.where(valid_boxes)]
        else:
            result_boxes = None
            if result_labels:
                areas = ([xc * yc, (w - xc) * yc,
                          xc * (h - yc), (w - xc) * (h - yc)])
                max_area = np.argmax(areas)
                result_labels = result_labels[max_area]
            else:
                result_labels = None

        return result_image, result_boxes, result_labels

    def load_mixup_image_and_boxes(self, index):

        origin_frac = np.clip(np.random.beta(1.0, 1.0), 0.35, 0.65)

        image, boxes, labels = self.load_image_and_boxes(index)
        r_image, r_boxes, r_labels = self.load_image_and_boxes(
            random.randint(0, self.image_ids.shape[0] - 1))

        result_image = origin_frac * image + (1 - origin_frac) * r_image

        if boxes is not None:
            result_boxes = np.vstack((boxes, r_boxes)).astype(np.int32)
            result_labels = np.vstack((labels, r_labels)).astype(np.int32)
        else:
            result_boxes = None
            if labels is not None:
                result_labels = labels if origin_frac >= 0.5 else r_labels
            else:
                result_labels = None

        return result_image, result_boxes, result_labels
