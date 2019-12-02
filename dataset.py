import os
import numpy as np
from PIL import Image, ImageDraw
from lxml import etree
from sklearn.cluster import k_means
from exceptions import SkipImage, CropError


class Dataset():
    def __init__(self, path, use_labels, anchors, transform=lambda x: x, ar=1.0):
        """
        path: path to folder with labelimg .xml files

        anchors: list of normalized (width, height) pairs

        use_labels: labels that are not in this list will be ignored

        transform: a transform that will be applied to image (must implement 
        __call__ method)
        """
        samples, itos, stoi = self._load_images(path, use_labels, anchors, ar)
        self.samples = samples
        self.itos = itos
        self.stoi = stoi
        self.transform = transform
        self.anchors = np.array(anchors)

    def _load_images(self, path, use_labels, anchors, ar):
        skipped_counter = 0
        crop_fail_counter = 0
        successful_images = 0

        samples = []
        all_labels = set()

        for filename in os.listdir(path):
            if filename.endswith('xml'):
                xml_path = os.path.join(path, filename)
                try:
                    *rest, boxes, labels, w, h = self._process_xml(xml_path, use_labels, ar)

                    boxes_wh = [-1.0, 1.0] @ boxes.reshape(-1, 2, 2)
                    boxes_xy = [0.5, 0.5] @ boxes.reshape(-1, 2, 2)

                    # Assign best (highest IOU) anchor box to each box
                    anchor_indices = []
                    for b_wh in boxes_wh / [w, h]:
                        i = np.argmax([IOU(b_wh, a_wh) for a_wh in anchors])
                        anchor_indices.append(i)

                    image = [*rest, anchor_indices, labels, boxes_xy]

                except SkipImage:
                    skipped_counter += 1

                except CropError:
                    crop_fail_counter += 1

                else:
                    successful_images += 1

                    samples.append(image)
                    all_labels.update(labels)

        itos = list(all_labels)
        stoi = dict((label, i) for i, label in enumerate(itos))

        print('Images loaded:', successful_images)
        print('Images skipped:', skipped_counter)
        print('Images failed to crop:', crop_fail_counter)

        return samples, itos, stoi

    def _process_xml(self, path, use_labels, ar):
        tree = etree.parse(path)
        im_path = tree.find('path').text

        boxes = []
        labels = []

        for obj in tree.findall('object'):
            x0 = int(obj.find('bndbox/xmin').text)
            y0 = int(obj.find('bndbox/ymin').text)
            x1 = int(obj.find('bndbox/xmax').text)
            y1 = int(obj.find('bndbox/ymax').text)
            label = obj.find('name').text

            if label in use_labels:
                boxes.append([x0, y0, x1, y1])
                labels.append(label)

        if not boxes:
            raise SkipImage

        image = Image.open(im_path)
        boxes = np.array(boxes)

        (x0, y0, x1, y1), shx, shy = self._compute_crop(image, boxes, ar)

        w, h = (image.width - 1), (image.height - 1)
    
        return im_path, (x0, y0, x1, y1), (shx, shy), boxes, labels, w, h

    def _compute_crop(self, image, boxes, ar):
        """
        1. Creates a window of size [w = ar*min(image.size), h = min(image.size)]
        2. Places the window on image so that it encloses ALL boxes in it,
        if that's not possible raises CropError
        3. Defines a random uniform range that the window can be shifted
        without losing boxes out of scope

        Parameters:
            image: PIL.Image instance
            boxes: np.ndarray of shape (N, 4)
            ar: desired aspect ratio of cropped image

        Returns:
            tuple(
                tuple(x0, y0, x1, y1): crop coordinates
                tuple(float, float): x axis safe shift range
                tuple(float, float): y axis safe shift range
            )
        """
        w = image.width - 1
        h = image.height - 1

        # Size of crop
        crop_h = min(w, h)
        crop_w = crop_h * ar

        # Total box coordinates
        tb_x0 = boxes[:, 0].min()
        tb_y0 = boxes[:, 1].min()
        tb_x1 = boxes[:, 2].max()
        tb_y1 = boxes[:, 3].max()

        if (tb_x1 - tb_x0) > crop_w or (tb_y1 - tb_y0) > crop_h:
            raise CropError('Cannot fit all boxes in crop.')

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

        # Crop shift
        shift_x = (-1 * min(x1 - tb_x1, x0), min(tb_x0 - x0, w - x1))
        shift_y = (-1 * min(y1 - tb_y1, y0), min(tb_y0 - y0, h - y1))

        return (x0, y0, x1, y1), shift_x, shift_y

    def __len__(self):
        return len(self.samples)

    def preview(self, index):
        path, (x0, y0, x1, y1), (shx, shy), anchor_indices, labels, boxes_xy = self.samples[index]

        shx = np.random.uniform(*shx)
        shy = np.random.uniform(*shy)

        crop_coords = np.add([x0, y0, x1, y1], [shx, shy, shx, shy])
        image = Image.open(path).convert('RGB').crop([*crop_coords])

        w, h = (x1 - x0), (y1 - y0)

        anchors_wh = np.array([self.anchors[i] for i in anchor_indices])

        a = boxes_xy @ [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]
        b = anchors_wh @ [[-w/2, 0.0, w/2, 0.0], [0.0, -h/2, 0.0, h/2]]
        boxes = a + b - [x0, y0, x0, y0] - [shx, shy, shx, shy]

        draw = ImageDraw.Draw(image)
        for box in boxes:
            draw.rectangle([*box], outline=(255, 255, 0), width=3)

        return image

    def __getitem__(self, index):
        path, (x0, y0, x1, y1), (shx, shy), anchor_indices, labels, boxes_xy = self.samples[index]

        shx = np.random.uniform(*shx)
        shy = np.random.uniform(*shy)

        crop_coords = np.add([x0, y0, x1, y1], [shx, shy, shx, shy])
        image = Image.open(path).convert('RGB').crop([*crop_coords])

        # 50% horizontal flip
        if np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            boxes_xy = ([w1, 0.0, w1, 0.0] - boxes) * [1.0, -1.0, 1.0, -1.0]

        out = np.zeros((1, 16, 16))

        norm_boxes_xy = (boxes_xy + [x0 - shx, y0 - shy]) / [x1 - x0, y1 - y0]

        for (x, y), i in zip(norm_boxes_xy, anchor_indices):
            w, h = self.anchors[i] * [16, 16]
            out[0, int(x), int(y)] = 1.0

        return image, out

    def __call__(self, out_w, out_h, stoi, shift=True, random_flip=True):
        """
        Parameters:
            out_w, int: width of feature space boxes will be mapped to
            out_h, int: height of feature space boxes will be mapped to
            stoi, dict: dictionary mapping labels to integers
            shift, bool: shift crop or not (safe for boxes)

        Returns:
            if stoi is set to None:
                PIL.Image, np.ndarray(5, out_h, out_w)
            else:
                PIL.Image, np.ndarray(5 + len(stoi), out_h, out_w)
        """

        if shift is True:
            shx = np.random.uniform(*self._shift_x)
            shy = np.random.uniform(*self._shift_y)
        else:
            shx = 0.0
            shy = 0.0

        shift = [shx, shy, shx, shy]

        image = Image.open(self._path).crop(np.array(self._crop) + shift)
        
        w0, h0 = image.width, image.height
        w1, h1 = out_w, out_h
        norm = [w1/w0, h1/h0, w1/w0, h1/h0]
        
        boxes = (self.boxes - shift) * norm

        if random_flip is True and np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = ([w1, 0.0, w1, 0.0] - boxes) * [1.0, -1.0, 1.0, -1.0]
            
        coordinates = [0.5, 0.5] @ boxes.reshape(-1, 2, 2)

        if len(stoi) > 1:
            out = self._multi_class_output(coordinates, out_h, out_w, stoi)
        else:
            out = self._single_class_output(coordinates, out_h, out_w)

        return image.convert('RGB'), out

    def _single_class_output(self, coordinates, out_h, out_w):
        shape = (3 * len(self.anchors), out_h, out_w)
        out = np.zeros(shape)

        for i, (x, y) in zip(self.anchor_indices, coordinates):

            x_offset = x - int(x)
            y_offset = y - int(y)

            s = len(self.anchors) + i * 2

            # Confidence
            out[i, int(y), int(x)] = 1.0

            # Offset x
            out[s, int(y), int(x)] = x_offset

            # Offset y
            out[s + 1, int(y), int(x)] = y_offset

        return out

    def _multi_class_output(self, coordinates, out_h, out_w, stoi):
        shape = (4 * len(self.anchors), out_h, out_w)
        out = np.zeros(shape)

        for i, L, (x, y) in zip(self.anchor_indices, self.labels, coordinates):

            x_offset = x - int(x)
            y_offset = y - int(y)

            s = len(self.anchors) + i * 3

            # Confidence
            out[i, int(y), int(x)] = 1.0

            # Offset x
            out[s + 1, int(y), int(x)] = x_offset

            # Offset y
            out[s + 2, int(y), int(x)] = y_offset

            # Label
            out[s + 3, int(y), int(x)] = stoi[L]

        return out



def IOU(a_wh, b_wh):
    aw, ah = a_wh
    bw, bh = b_wh

    I = min(aw, bw) * min(ah, bh)

    area_a = aw * ah
    area_b = bw * bh

    U = area_a + area_b - I

    return I / U


def k_means_anchors(boxes, k):
    """
    Parameters:
        boxes: np.ndarray of shape (N, 4)
        k: desired number of anchor boxes
    Returns:
        np.ndarray of shape (N, 2)
    """
    wh = [-1.0, 1.0] @ boxes.reshape(-1, 2, 2)
    anchors, *_ = k_means(wh, k)
    return anchors
