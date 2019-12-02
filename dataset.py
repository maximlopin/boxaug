import os
import numpy as np
from PIL import Image, ImageDraw
from lxml import etree
from sklearn.cluster import k_means
from exceptions import SkipImage, CropError


class ImageWithBoxes():
    def __init__(self, path, boxes, labels, ar=1.0):
        """
        Parameters:
            path (str): path to image
            boxes (iterable): list of non-normalized box coordinates (x0, y0, x1, y1)
            lables (list of strings): labels for boxes
            ar (float): desired aspect ratio of crops
        """
        image = Image.open(path)
        boxes = np.array(boxes)

        crop, shift_x, shift_y, = self._compute_crop(image, boxes, ar)

        x0, y0, *_ = crop

        self._shift_x = shift_x
        self._shift_y = shift_y
        self._crop = crop
        self.width = [-1.0, 0.0, 1.0, 0.0] @ np.array(crop)
        self.height = [0.0, -1.0, 0.0, 1.0] @ np.array(crop)

        self.anchors = None
        self.anchor_indices = None
        self.boxes = boxes - [x0, y0, x0, y0]
        self.labels = labels

        self._path = path

    @property
    def norm_boxes(self):
        w, h = self.width, self.height
        return self.boxes / [w, h, w, h]

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
            out[s + , int(y), int(x)] = y_offset

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

    def preview(self):
        shx = np.random.uniform(*self._shift_x)
        shy = np.random.uniform(*self._shift_y)
        shift = [shx, shy, shx, shy]

        image = Image.open(self._path).crop(np.array(self._crop) + shift)
        boxes = self.boxes - shift

        draw = ImageDraw.Draw(image)
        for i, box in zip(self.anchor_indices, boxes):

            # Center coordinates of box
            x, y = [0.5, 0.5] @ box.reshape(2, 2)

            # Size of corresponding anchor box
            w, h = self.anchors[i]

            # Coordinates of the anchor box
            x0 = (x / self.width - w / 2) * image.width
            y0 = (y / self.height - h / 2) * image.height
            x1 = (x / self.width + w / 2) * image.width
            y1 = (y / self.height + h / 2) * image.height

            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 255), width=3)

            draw.rectangle([*box], outline=(255, 255, 0), width=3)

        return image

    @classmethod
    def from_xml(cls, path, use_labels):
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
                boxes.append(np.array([x0, y0, x1, y1]))
                labels.append(label)

        if not boxes:
            raise SkipImage

        return cls(im_path, boxes, labels)


class Dataset():
    def __init__(self, path, anchors, use_labels, transform=lambda x: x):
        """
        path: path to folder with labelimg .xml files

        anchors: list of normalized (width, height) pairs

        use_labels: labels that are not in this list will be ignored

        transform: a transform that will be applied to image (must implement 
        __call__ method)
        """
        samples, itos, stoi = self._load_images(path, use_labels)
        self.samples = samples
        self.itos = itos
        self.stoi = stoi
        self.transform = transform
        self.anchors = anchors

        # Assign anchors to each box
        for s in samples:
            boxes = [-1.0, 1.0] @ s.norm_boxes.reshape(-1, 2, 2)
            indices = [np.argmax([IOU(a, b) for a in anchors])
                                                for b in boxes]
            s.anchors = anchors
            s.anchor_indices = indices

    def boxes(self):
        return np.concatenate([s.norm_boxes for s in samples], axis=0)

    def mean_iou(self):
        iou_list = []
        for s in self.samples:
            boxes = [-1.0, 1.0] @ s.norm_boxes.reshape(-1, 2, 2)
            for i, a_wh in zip(s.anchor_indices, boxes):
                b_wh = s.anchors[i]
                iou = IOU(a_wh, b_wh)
                iou_list.append(iou)
        return np.mean(iou_list)

    def _load_images(self, path, use_labels):
        images = []
        labels = set()
        for filename in os.listdir(path):
            if filename.endswith('xml'):
                xml_path = os.path.join(path, filename)
                try:
                    image_wb = ImageWithBoxes.from_xml(xml_path, use_labels)
                except (SkipImage, CropError):
                    continue

                images.append(image_wb)
                labels.update(image_wb.labels)

        itos = list(labels)
        stoi = dict((label, i) for i, label in enumerate(itos))

        return images, itos, stoi
        
    def sample(self, index):
        return self.samples[index].preview()

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        image_wb = self.samples[index]
        image, y = image_wb(16, 16, self.stoi)
        x = self.transform(image)
        return x, y


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
