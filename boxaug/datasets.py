import os
import numpy as np
from PIL import Image
from boxaug import transforms
from boxaug import utils
from boxaug.exceptions import BoxaugError


class LabelImgDataset():
    def __init__(self, path, transform, min_boxes=1, use_labels=None):
        self._samples = utils.load_labelimg(path, min_boxes, use_labels)
        self._tfm = transform or transforms.Identity()

    def __getitem__(self, index):
        path, bboxes, labels = self._samples[index]

        image = Image.open(path).convert('RGB')
        image_arr = np.asarray(image)

        image_out, bboxes_out = self._tfm.apply_to_bboxes(image_arr, bboxes)

        return image_out, bboxes_out, labels

    def __len__(self):
        return len(self._samples)


class AutoCropDataset():
    """
    This dataset will crop and resize images automatically, just pass
    desired width and height to constructor.

    Images for which it's impossible to keep all bounding
    boxes in focus after cropping will be skipped.
    """

    def __init__(self, path, transform, width, height,
                 min_boxes=1, use_labels=None):

        samples = utils.load_labelimg(path, min_boxes, use_labels)
        self._ar = width / height
        self._samples, self._crop_coords =  self._preprocess_samples(samples)
        self._tfm = transforms.Compose(
            transforms.Resize(width, height),
            transform or transforms.Identity()
        )
        
    def _preprocess_samples(self, samples):
        samples_out = []
        crop_coords = []

        fail = 0

        for s in samples:
            img_path, bboxes, _ = s

            image = Image.open(img_path).convert('RGB')
            image_arr = np.asarray(image)
            points = bboxes.reshape(-1, 2)

            try:
                coords = utils.auto_crop(image_arr, points, self._ar, True)
            except BoxaugError:
                fail += 1
            else:
                samples_out.append(s)
                crop_coords.append(coords)

        print('Images skipped:', fail)
        print('Images loaded:', len(samples_out))

        return samples_out, crop_coords

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        im_path, bboxes, labels = self._samples[index]

        x0, y0, x1, y1 = self._crop_coords[index]

        image = Image.open(im_path).convert('RGB').crop([x0, y0, x1, y1])
        image_arr = np.asarray(image)

        image_out, bboxes_out = self._tfm.apply_to_bboxes(
            image_arr,
            bboxes - [x0, y0, x0, y0]
        )

        return image_out, bboxes_out, labels
