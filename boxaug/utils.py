import numpy as np
from PIL import Image, ImageDraw


class CropError(Exception):
    pass


def IOU(a_wh, b_wh):
    """
    Intersection over Union

    Args:
        a_wh: (width, height) of box A
        b_wh: (width, height) of box B

    Returns float.
    """
    aw, ah = a_wh
    bw, bh = b_wh

    I = min(aw, bw) * min(ah, bh)

    area_a = aw * ah
    area_b = bw * bh

    U = area_a + area_b - I

    return I / U


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


def bbox_safe_crop(image, bboxes, ar):
    """
    Crops image to desired aspect ratio keeping all bounding boxes inside
    the crop. If such crop is impossible, CropError is raised.

    Args:
        image: np.ndarray of shape (H, W, 3)
        bboxes: np.ndarray of shape (N, 4)
        ar (scalar): desired aspect ratio of the crop
    
    Returns:
        (np.ndarray) image, (np.ndarray) bboxes

    """
    assert len(bboxes.shape) == 2
    assert bboxes.shape[1] == 4
    assert ar > 0

    w, h = image.shape[1] - 1, image.shape[0] - 1

    if ar > 1:
        crop_w = min(w, h)
        crop_h = int(crop_w / ar)
    else:
        crop_h = min(w, h)
        crop_w = int(crop_h * ar)

    # Total box coordinates
    tb_x0 = bboxes[:, 0].min()
    tb_y0 = bboxes[:, 1].min()
    tb_x1 = bboxes[:, 2].max()
    tb_y1 = bboxes[:, 3].max()

    if (tb_x1 - tb_x0) > crop_w or (tb_y1 - tb_y0) > crop_h:
            raise CropError('Cannot fit all boxes inside crop.')

    # Total box center coordinates
    center_x, center_y = np.array([0.5, 0.5]) @ [[tb_x0, tb_y0],
                                                 [tb_x1, tb_y1]]
    # Crop coordinates
    x0 = np.dot([center_x, crop_w], [1.0, -0.5])
    y0 = np.dot([center_y, crop_h], [1.0, -0.5])
    x1 = np.dot([center_x, crop_w], [1.0, +0.5])
    y1 = np.dot([center_y, crop_h], [1.0, +0.5])

    # Crop offset
    offset_x = min(0, w - x1) - min(0, x0)
    offset_y = min(0, h - y1) - min(0, y0)

    x0 += offset_x
    y0 += offset_y
    x1 += offset_x
    y1 += offset_y

    image_out = image[int(y0):int(y1), int(x0):int(x1)]

    bboxes_out = bboxes - [x0, y0, x0, y0]

    return image_out, bboxes_out
