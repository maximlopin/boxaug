import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

__all__ = ['Compose', 'Affine', 'Identity', 'Flip', 'Resize']


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


# TODO
# class AdaptiveCrop(TransformBase):
#     def __init__(self, width, height):
#         self.w = width
#         self.h = height

#     def __call__(self, image, boxes):
#         w = image.shape[1]
#         h = image.shape[0]

#         # Total box coordinates
#         tb_x0 = boxes[:, 0].min()
#         tb_y0 = boxes[:, 1].min()
#         tb_x1 = boxes[:, 0].max()
#         tb_y1 = boxes[:, 1].max()

#         # Total box center coordinates
#         center_x, center_y = np.array([0.5, 0.5]) @ [[tb_x0, tb_y0],
#                                                      [tb_x1, tb_y1]]
#         # Crop coordinates
#         x0 = np.dot([center_x, crop_w], [1.0, -0.5])
#         y0 = np.dot([center_y, crop_h], [1.0, -0.5])
#         x1 = np.dot([center_x, crop_w], [1.0, +0.5])
#         y1 = np.dot([center_y, crop_h], [1.0, +0.5])

#         # Crop offset
#         offset_x = min(0, w - x1) - min(0, x0)
#         offset_y = min(0, h - y1) - min(0, y0)

#         x0 += offset_x
#         y0 += offset_y
#         x1 += offset_x
#         y1 += offset_y

#         # Crop shift
#         shift_x = (-1 * min(x1 - tb_x1, x0), min(tb_x0 - x0, w - x1))
#         shift_y = (-1 * min(y1 - tb_y1, y0), min(tb_y0 - y0, h - y1))

#         return (x0, y0, x1, y1), shift_x, shift_y


class Compose(TransformBase):
    """Stack multiple transforms"""

    def __init__(self, *tfms):
        self.tfms = tfms

    def __call__(self, image, points):
        for t in self.tfms:
            image, points = t(image, points)
        return image, points


class Flip(TransformBase):
    def __init__(self, axis):
        assert axis in [0, 1], 'flip axis must be 0 or 1'
        self.axis = axis

    def __call__(self, image, points):
        w, h = image.shape[1], image.shape[0]

        if np.random.random() > 0.5:
            image = np.flip(image, axis=self.axis)

            if self.axis == 0:
                points = [1, -1] * points + [0, h]
            else:
                points = [-1, 1] * points + [w, 0]

        return image, points


class Affine(TransformBase):
    def __init__(self, deg, shear, resampling='constant'):
        """
        Args:
            deg (scalar or tuple): rotation angle
            shear (tuple): (xrange, yrange) or (xmin, xmax, ymin, ymax)
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


def split_bboxes(bboxes):
    """
    Splits array of bounding boxes of format [[x0, y0, x1, y1], ...]
    into 2 arrays: center coordinates of bboxes, widths and heights of bboxes

    Args:
        np.ndarray of shape (N, 4)

    Returns:
        tuple(
            np.ndarray of shape(N, 2): [[x, y], ...]
            np.ndarray of shape(N, 2): [[w, h], ...]
        )
    """
    assert len(bboxes.shape) == 2
    assert bboxes.shape[1] == 4

    xy_tfm = np.array([[0.5, 0.0],
                       [0.0, 0.5],
                       [0.5, 0.0],
                       [0.0, 0.5]])

    wh_tfm = np.array([[-1.0, 0.0],
                       [0.0, -1.0],
                       [1.0, 0.0],
                       [0.0, 1.0]])

    xy = bboxes @ xy_tfm

    wh = bboxes @ wh_tfm

    return xy, wh


def merge_bboxes(xy_array, wh_array):
    """
    Args:
        xy_array: np.ndarray of shape (N, 2), center coordinates of bboxes
        wh_array: np.ndarray of shape (N, 2), widths and heights of bboxes

    Returns:
        np.ndarray of shape (N, 4) like this: [[x0, y0, x1, y1], ...]
    """
    assert xy_array.shape[0] == xy_array.shape[0]
    assert len(xy_array.shape) == len(wh_array.shape) == 2
    assert xy_array.shape[1] == wh_array.shape[1] == 2

    tfm = np.array([[1, 0, -0.5, 0],
                    [0, 1, 0, -0.5],
                    [1, 0, 0.5, 0],
                    [0, 1, 0, 0.5]])

    bboxes = tfm @ np.concatenate((xy_array, wh_array), axis=1).T

    return bboxes.T


def to_pil(image, bboxes):
    """
    Args:
        image: np.ndarray of shape (H, W, 3)
        bboxes: np.ndarray of shape (N, 4)

    Returns:
        PIL Image
    """
    assert len(bboxes.shape) == 2
    assert bboxes.shape[1] == 4

    pil_image = Image.fromarray(image)

    draw = ImageDraw.Draw(pil_image)

    for x0, y0, x1, y1 in bboxes:
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=3)

    return pil_image
