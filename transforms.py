import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw


class TransformBase():
    """Transform Interface"""

    def __call__(self, X):
        """
        Parameters:
            image: np.ndarray of shape (H, W, 3)
            boxes: np.ndarray of shape (N, 2)
        """
        raise NotImplementedError


class AdaptiveCrop(TransformBase):
    def __init__(self, width, height):
        self.w = width
        self.h = height

    def __call__(self, image, boxes):
        w, h = self.w, self.h
        raise NotImplementedError


class Compose(TransformBase):
    def __init__(self, *tfms):
        self.tfms = tfms

    def __call__(self, image, boxes):
        if isinstance(image, Image.Image):
            image = np.asarray(image)

        for t in self.tfms:
            image, boxes = t(image, boxes)

        return image, boxes


class Flip(TransformBase):
    def __init__(self, axis):
        self.axis = axis
    
    def __call__(self, image, boxes):
        w, h = image.shape[1], image.shape[0]

        image_out = np.flip(image, axis=self.axis)
        boxes_out = -1 * boxes + [w, h]

        return image_out, boxes_out


class ToPIL(TransformBase):
    def __call__(self, image, boxes):
        image_out = Image.fromarray(image)

        draw = ImageDraw.Draw(image_out)
        for b in boxes:
            x, y = b
            draw.ellipse([x-20, y-20, x+20, y+20], fill=(255, 0, 255), width=5)

        return image_out, boxes


class RandomAffine(TransformBase):
    def __init__(self, deg, shx, shy):
        self.deg = deg
        self.shx = shx
        self.shy = shy
        self.mode = 'constant'
    
    def __call__(self, image, boxes):

        rad = np.deg2rad(self.deg)

        h, w, _ = image.shape

        # Translate origin to center
        translate_1 = np.identity(4)
        translate_1[0, -1] = h / 2
        translate_1[1, -1] = w / 2

        # Rotate
        rotate = np.identity(4)
        rotate[0, 0] = np.cos(rad)
        rotate[0, 1] = np.sin(rad)
        rotate[1, 0] = -np.sin(rad)
        rotate[1, 1] = np.cos(rad)

        # Translate center to origin
        translate_2 = np.identity(4)
        translate_2[0, -1] = -h / 2 - (self.shy * h) / 2
        translate_2[1, -1] = -w / 2 - (self.shx * w) / 2

        # Shear
        shear = np.identity(4)
        shear[0, 1] = self.shy
        shear[1, 0] = self.shx

        # Final image transform matrix
        img_tfm = translate_1 @ rotate @ translate_2 @ shear

        # Final boxes transform matrix
        idx = [0, 1, 3]
        box_tfm = np.linalg.inv(img_tfm[idx][:, idx])

        # [[x, y], ...] -> [[y, ...], [x, ...], [1.0, ...]]
        n = boxes.shape[0]
        boxes = np.concatenate([boxes.T[::-1], np.ones(n).reshape(1, -1)], 0)

        # Apply transforms
        image_out = ndimage.affine_transform(image, img_tfm)
        boxes_out = (box_tfm @ boxes)[:-1][::-1].T
        
        return image_out, boxes_out
