import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

__all__ = ['Compose', 'Affine', 'Identity', 'Flip', 'ToPIL']


class TransformBase():
    """Transform Interface"""

    def __call__(self, image, boxes):
        """
        Args:
            image: np.ndarray of shape (H, W, 3)
            boxes: np.ndarray of shape (N, 2)

        Returns (same shapes):
            image: np.ndarray of shape (H, W, 3)
            boxes: np.ndarray of shape (N, 2)
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

    def __call__(self, image, boxes):
        for t in self.tfms:
            image, boxes = t(image, boxes)
        return image, boxes


class Flip(TransformBase):
    def __init__(self, axis):
        assert axis in [0, 1], 'flip axis must be 0 or 1'
        self.axis = axis

    def __call__(self, image, boxes):
        w, h = image.shape[1], image.shape[0]

        if np.random.random() > 0.5:
            image = np.flip(image, axis=self.axis)

            if self.axis == 0:
                boxes = [1, -1] * boxes + [0, h]
            else:
                boxes = [-1, 1] * boxes + [w, 0]

        return image, boxes


class ToPIL(TransformBase):
    """Converts image and boxes to a PIL image"""

    def __call__(self, image, boxes):

        d = 0.05 * min(image.shape[0], image.shape[1])
        fill = (255, 255, 0)

        image_out = Image.fromarray(image)

        draw = ImageDraw.Draw(image_out)
        for b in boxes:
            x, y = b
            draw.ellipse([x-d, y-d, x+d, y+d], fill=fill, width=3)

        return image_out, boxes


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

    def __call__(self, image, boxes):
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

        # Final boxes transform matrix
        idx = [0, 1, 3]
        box_tfm = np.linalg.inv(img_tfm[idx][:, idx])

        # [[x, y], ...] -> [[y, ...], [x, ...], [1.0, ...]]
        n = boxes.shape[0]
        boxes = np.concatenate([boxes.T[::-1], np.ones(n).reshape(1, -1)], 0)

        # Apply transforms
        image_out = ndimage.affine_transform(image, img_tfm, mode=self.mode)
        boxes_out = (box_tfm @ boxes)[:-1][::-1].T

        return image_out, boxes_out


class Identity(TransformBase):
    def __call__(self, image, boxes):
        return image, boxes


class Resize(TransformBase):
    def __init__(self, width, height):
        self.w = width
        self.h = height

    def __call__(self, image, boxes):
        w, h = image.shape[1], image.shape[0]

        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((self.w, self.h), Image.BILINEAR)
        image_out = np.asarray(pil_img)

        boxes_out = boxes * [self.w / w, self.h / h]

        return image_out, boxes_out
