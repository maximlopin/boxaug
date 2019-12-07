import numpy as np
from scipy import ndimage
from PIL import Image
from boxaug import utils

__all__ = ['Compose', 'Affine', 'Identity', 'Flip', 'Resize', 'Crop']


class TransformBase():
    """Transform Interface"""

    def __call__(self, image, points):
        """
        Args:
            image: np.ndarray of shape (H, W, 3)
            points: np.ndarray of shape (N, 2)

        Returns:
            tuple(transformed image, transformed points)
        """
        raise NotImplementedError


class Compose(TransformBase):
    """Stack multiple transforms"""

    def __init__(self, *tfms):
        self.tfms = tfms

    def __call__(self, image, points):
        for t in self.tfms:
            image, points = t(image, points)
        return image, points


class Flip(TransformBase):
    def __init__(self, axis, p=0.5):
        assert axis in [0, 1], 'flip axis must be 0 or 1'
        self.axis = axis
        self.p = p

    def __call__(self, image, points):
        w, h = image.shape[1], image.shape[0]

        if np.random.random() < self.p:
            image = np.flip(image, axis=self.axis)

            if self.axis == 0:
                points = [1, -1] * points + [0, h]
            else:
                points = [-1, 1] * points + [w, 0]

        return image, points


class Affine(TransformBase):
    def __init__(self, deg=0, shear=0, resampling='constant'):
        """
        Args:
            deg (scalar or tuple): rotation angle

            shear (scalar or tuple): scalar(xrange) or tuple(xrange, yrange) 
            or tuple(xmin, xmax, ymin, ymax)

            resampling: 'reflect' | 'constant' | 'nearest' | 'mirror' | 'wrap'
        """

        if isinstance(deg, (int, float)):
            self.deg = (-deg, deg)
        elif isinstance(deg, tuple):
            self.deg = (deg[0], deg[1])
        else:
            raise TypeError('degrees must be either tuple or scalar')

        if isinstance(shear, (int, float)):
            self.shx = (-shear, shear)
            self.shy = (0, 0)
        elif isinstance(shear, tuple):
            if len(shear) == 2:
                self.shx = (-shear[0], shear[0])
                self.shy = (-shear[1], shear[1])
            elif len(shear) == 4:
                self.shx = (shear[0], shear[1])
                self.shy = (shear[2], shear[3])
            else:
                raise TypeError('shear must be a tuple of 2 or 4 scalars')
        else:
            raise TypeError('shx must be either tuple or scalar')

        scipy_modes = ['reflect', 'constant', 'nearest', 'mirror', 'wrap']

        assert resampling in scipy_modes, 'use a valid resampling mode'

        self.mode = resampling

    def __call__(self, image, points):
        shx = np.random.uniform(*self.shx)
        shy = np.random.uniform(*self.shy)
        deg = np.random.uniform(*self.deg)

        rad = np.deg2rad(deg)

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
        translate_2[0, -1] = -h / 2 - (shy * h) / 2
        translate_2[1, -1] = -w / 2 - (shx * w) / 2

        # Shear
        shear = np.identity(4)
        shear[0, 1] = shy
        shear[1, 0] = shx

        # Final image transform matrix
        img_tfm = translate_1 @ rotate @ translate_2 @ shear

        # Final points transform matrix
        idx = [0, 1, 3]
        point_tfm = np.linalg.inv(img_tfm[idx][:, idx])

        # [[x, y], ...] -> [[y, ...], [x, ...], [1.0, ...]]
        n = points.shape[0]
        points = np.concatenate([points.T[::-1], np.ones(n).reshape(1, -1)], 0)

        # Apply transforms
        image_out = ndimage.affine_transform(image, img_tfm, mode=self.mode)
        points_out = (point_tfm @ points)[:-1][::-1].T

        return image_out, points_out


class Identity(TransformBase):
    def __call__(self, image, points):
        return image, points


class Resize(TransformBase):
    def __init__(self, width, height):
        self.w = width
        self.h = height

    def __call__(self, image, points):
        w, h = image.shape[1], image.shape[0]

        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((self.w, self.h), Image.BILINEAR)
        image_out = np.asarray(pil_img)

        points_out = points * [self.w / w, self.h / h]

        return image_out, points_out


class Crop(TransformBase):
    def __init__(self, x0, y0, x1, y1):
        self.x0 = int(x0)
        self.y0 = int(y0)
        self.x1 = int(x1)
        self.y1 = int(y1)

    def __call__(self, image, points):
        image_out = image[self.y0:self.y1, self.x0:self.x1]
        points_out = points - [self.x0, self.y0]

        return image_out, points_out


class AutoCrop(TransformBase):
    """
    Crops image to desired aspect ratio keeping all points inside the crop.

    WARNING: If such crop is impossible, exceptions.BoxaugError is raised.
    Before using this transform you might want to remove samples from your
    dataset that cause TransformError.
    """

    def __init__(self, aspect_ratio):
        assert aspect_ratio > 0
        self.ar = aspect_ratio

    def __call__(self, image, points):
        image, points = utils.auto_crop(image, points, self.ar)
        return image, points
