import os
import numpy as np
from PIL import Image
from boxaug import transforms
from boxaug import utils


class LabelImgDataset():
    def __init__(self, path, transform, min_boxes=1, use_labels=None):
        self.samples = utils.load_labelimg(path, min_boxes, use_labels)
        self.tfm = transform or transforms.Identity()

    def __getitem__(self, index):
        path, bboxes, labels = self.samples[index]

        image = Image.open(path).convert('RGB')
        image_arr = np.asarray(image)

        # Boxes to points
        bboxes_4pts = utils.bboxes_as_4pts(bboxes)
        points = bboxes_4pts.reshape(-1, 2)

        # Apply transform
        image_out, points_out = self.tfm(image_arr, points)

        # Points back to bboxes
        bboxes_4pts = points_out.reshape(-1, 8)
        bboxes_out = utils.bboxes_as_2pts(bboxes_4pts, align=True)

        return image_out, bboxes_out, labels

    def __len__(self):
        return len(self.samples)
