import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw


class TransformBase():
    def __call__(self, X):
        raise NotImplementedError


class Rotate(TransformBase):
    def __init__(self, deg):
        """
        Parameters:
            deg: tuple(scalar, scalar) or scalar
        """
        if isinstance(deg, tuple):
            self.min, self.max = deg
        elif isinstance(deg, (int, float)):
            self.min, self.max = -deg / 2, deg / 2
        else:
            raise TypeError('deg must be either tuple or scalar')


    def __call__(self, image, boxes):
        """
        Parameters:
            image: np.ndarray of shape (H, W, 3)
            boxes: np.ndarray of shape (N, 2)
        """
        deg = np.random.uniform(self.min, self.max)

        rad = np.deg2rad(deg)

        c, s = np.cos(rad), np.sin(rad)
        w, h = image.shape[1] / 2, image.shape[0] / 2

        image_out = ndimage.rotate(image, -deg, reshape=False)
        boxes_out = np.dot([[c, -s], [s, c]], (boxes - [w, h]).T)

        return image_out, boxes_out.T + [w, h]


class Shear(TransformBase):
    def __init__(self, x_shear, y_shear):
        """
        Parameters:
            x_shear: tuple(scalar, scalar) or scalar
            y_shear: tuple(scalar, scalar) or scalar
        """
        if isinstance(x_shear, tuple):
            self.xmin, self.xmax = x_shear
        elif isinstance(x_shear, (int, float)):
            self.xmin, self.xmax = -x_shear / 2, x_shear / 2
        else:
            raise TypeError('x_shear must be either tuple or float')

        if isinstance(y_shear, tuple):
            self.ymin, self.ymax = y_shear
        elif isinstance(y_shear, (int, float)):
            self.ymin, self.ymax = -y_shear / 2, y_shear / 2
        else:
            raise TypeError('y_shear must be either tuple or float')


    def __call__(self, image, boxes):
        """
        Parameters:
            image: np.ndarray of shape (H, W, 3)
            boxes: np.ndarray of shape (N, 2)
        """
        a = np.random.uniform(self.xmin, self.xmax)
        b = np.random.uniform(self.ymin, self.ymax)

        m = np.array([[1, b, 0],
                      [a, 1, 0],
                      [0, 0, 1]])

        w, h = image.shape[1], image.shape[0]

        o = np.array([w, h]) * [-b / 2, -a / 2]

        image_out = ndimage.affine_transform(image, m, [*o, 0])
        boxes_out = np.dot([[1, -a], [-b, 1]], boxes.T).T + [-o[1], -o[0]]

        return image_out, boxes_out


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
    def __init__(self):
        pass

    def __call__(self, image, boxes):
        image_out = Image.fromarray(image)

        draw = ImageDraw.Draw(image_out)
        for b in boxes:
            print([*b])
            draw.ellipse([*(b - 20), *(b + 20)], fill=(255, 0, 255), width=5)

        return image_out, boxes