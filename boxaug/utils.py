import os
import numpy as np
from lxml import etree
from PIL import Image, ImageDraw
from boxaug.exceptions import BoxaugError


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


def auto_crop(image, points, ar, return_coordinates=False):
    """
    Crops image to desired aspect ratio keeping all points inside the crop.

    WARNING: If such crop is impossible, exceptions.BoxaugError is raised.

    Args:
        image: np.ndarray of shape (H, W, 3)
        points: np.ndarray of shape (N, 4)
        ar (scalar): desired aspect ratio of the crop
    
    Returns:
        (np.ndarray) image, (np.ndarray) points

        or (if return_coordinates is True)

        tuple of ints: (x0, y0, x1, y1)
    """
    assert len(points.shape) == 2
    assert points.shape[1] == 2
    assert ar > 0

    w, h = image.shape[1] - 1, image.shape[0] - 1

    if ar > 1:
        crop_w = min(w, h)
        crop_h = int(crop_w / ar)
    else:
        crop_h = min(w, h)
        crop_w = int(crop_h * ar)

    # Total box coordinates
    tb_x0 = points[:, 0].min()
    tb_y0 = points[:, 1].min()
    tb_x1 = points[:, 0].max()
    tb_y1 = points[:, 1].max()

    if (tb_x1 - tb_x0) > crop_w or (tb_y1 - tb_y0) > crop_h:
            raise BoxaugError('Cannot fit all boxes inside crop.')

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

    if return_coordinates is True:
        return int(x0), int(y0), int(x1), int(y1)

    image_out = image[int(y0):int(y1), int(x0):int(x1)]

    points_out = points - [x0, y0]

    return image_out, points_out


def bboxes_as_4pts(bboxes):
    """
    Converts 2-point bboxes to 4-point format.

    Args:
        bboxes: np.ndarray of shape (N, 4)

    Returns:
        np.ndarray of shape (N, 8)
    """
    assert len(bboxes.shape) == 2
    assert bboxes.shape[1] == 4

    a = np.identity(4)
    b = np.identity(4)[[0, 3, 2, 1]]

    m = np.concatenate((a, b), axis=0)

    bboxes_out = (m @ bboxes.T).T

    return bboxes_out


def bboxes_as_2pts(bboxes, align=True):
    """
    Converts 4-point bboxes to 2-point format with or without alignment.

    Args:
        bboxes: np.ndarray of shape (N, 8)

    Returns:
        np.ndarray of shape (N, 4)
    """
    assert len(bboxes.shape) == 2
    assert bboxes.shape[1] == 8

    if align is True:
        for i in range(bboxes.shape[0]):
            points = bboxes[i].reshape(-1, 2)

            cx, cy = points.mean(axis=0)
            points -= [cx, cy]

            dx, dy = (points[0] - points[2])
            theta = -1 * np.arctan(dx / dy)

            s, c = np.sin(theta), np.cos(theta)
            rotate = np.array([[c, s], [-s, c]])

            points = (rotate @ points.T).T + [cx, cy]
            bboxes[i] = points.reshape(8)
    
    return bboxes[:, :4]


def load_labelimg(path, min_boxes=1, use_labels=None):
        """
        Args:
            path (str): path to directory with LabelImg .xml files

        Returns:
            list of tuples: (
                img_path (str), 
                boxes (np.ndarray of shape (N, 4)),
                labels (list of str)
            )
        """
        samples = []

        for filename in os.listdir(path):
            if filename.endswith('xml'):
                xml_path = os.path.join(path, filename)

                tree = etree.parse(xml_path)

                img_path = tree.find('path').text

                bboxes = []
                labels = []

                for obj in tree.findall('object'):
                    x0 = int(obj.find('bndbox/xmin').text)
                    y0 = int(obj.find('bndbox/ymin').text)
                    x1 = int(obj.find('bndbox/xmax').text)
                    y1 = int(obj.find('bndbox/ymax').text)
                    label = obj.find('name').text

                    if use_labels is None or label in use_labels:
                        bboxes.append((x0, y0, x1, y1))
                        labels.append(label)

                if len(bboxes) >= min_boxes:
                    samples.append((img_path, np.array(bboxes), labels))

        return samples


def assign_anchors(bboxes_wh, anchor_bboxes_wh):
    """
    Args:
        bboxes: np.ndarray of shape (N, 2)
        anchor_bboxes: np.ndarray of shape (M, 2)

    Returns:
        list of N integers (indices of anchor boxes)
    """
    return [np.argmax([IOU(a_wh, b_wh) for b_wh in anchor_bboxes_wh])
                                       for a_wh in bboxes_wh]
