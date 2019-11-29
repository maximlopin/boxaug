import os
from PIL import Image, ImageDraw
import numpy as np
from lxml import etree
from sklearn.cluster import k_means


def IOU(a_wh, b_wh):
    aw, ah = a_wh
    bw, bh = b_wh

    I = min(aw, bw) * min(ah, bh)

    area_a = aw * ah
    area_b = bw * bh

    U = area_a + area_b - I

    return I / U


class SkipImage(Exception):
    pass


class CropError(Exception):
    pass


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

        (x0, y0, x1, y1) = crop

        w = [-1.0, 0.0, 1.0, 0.0] @ np.array(crop)
        h = [0.0, -1.0, 0.0, 1.0] @ np.array(crop)

        crop_boxes = boxes - [x0, y0, x0, y0]

        self._shift_x = shift_x
        self._shift_y = shift_y
        self._crop = crop

        self.width = w
        self.height = h

        self._path = path

        self.anchors = None

        self.boxes = crop_boxes
        self.labels = labels

    @property
    def norm_boxes(self):
        w, h = self.width, self.height
        return self.boxes / [w, h, w, h]

    def __call__(self, out_w, out_h, stoi=None, shift=True):
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
        image = Image.open(self._path)

        if shift is True:
            shift_x = np.random.uniform(*self._shift_x)
            shift_y = np.random.uniform(*self._shift_y)
            shift = [shift_x, shift_y]

            boxes = self.boxes - shift
            x0, y0, x1, y1 = np.array(self._crop) + shift
        else:
            boxes = self.boxes
            x0, y0, x1, y1 = np.array(self._crop)
            
        crop = image.crop([*crop])

        boxes = boxes / [crop.width, crop.height, crop.width, crop.height]
        boxes = boxes * [out_w, out_h, out_w, out_h]
        
        if stoi is None:
            out = self._single_class_output(boxes, out_h, out_w)
        else:
            out = self._multi_class_output(boxes, stoi, out_h, out_w)

        return crop, out

    def _single_class_output(self, boxes, out_h, out_w):
        shape = (5, out_h, out_w)
        out = np.zeros(shape)

        for box in boxes:
            x, y = np.array([0.5, 0.5]) @ box.reshape((2, 2))

            # Confidence
            out[0, int(y), int(x)] = 1.0

            # Offset x
            out[1, int(y), int(x)] = 1.0

            # Offset y
            out[2, int(y), int(x)] = 1.0

            # Width
            out[3, int(y), int(x)] = 1.0

            # Height
            out[4, int(y), int(x)] = 1.0

        return out

    def _multi_class_output(self, boxes, stoi, out_h, out_w):
        shape = (5 + len(stoi), out_h, out_w)
        out = np.zeros(shape)

        for box, label in zip(boxes, self.labels):
            x, y = np.array([0.5, 0.5]) @ box.reshape((2, 2))

            # Confidence
            out[0, int(y), int(x)] = 1.0

            # Offset x
            out[1, int(y), int(x)] = 1.0

            # Offset y
            out[2, int(y), int(x)] = 1.0

            # Width
            out[3, int(y), int(x)] = 1.0

            # Height
            out[4, int(y), int(x)] = 1.0

            # Class
            out[5 + stoi[label], int(y), int(x)] = 1.0

        return out

    def _compute_crop(self, image, boxes, ar):
        """
        1. Creates a window of size [w = ar*min(image.size), h = min(image.size)]
        2. Places the windows on image so that it encloses ALL boxes in it,
        if that's not possible raises CropError
        3. Defines a random uniform range that the window can be shifted
        without losing boxes out of scope

        # TODO: Only raise CropError if it's impossible to get rid of
        # some box that prevents you from enclosing all other boxes without
        # shifting out of image size range (or rather do this optionally)

        Parameters:
            image: PIL.Image instance
            boxes: np.ndarray of shape (N, 4)
            ar: desired aspect ratio of cropped image

        Returns:
            tuple(
                np.ndarray([x0, y0, x1, y1]): crop coordinates
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
        shift_x = np.random.uniform(*self._shift_x)
        shift_y = np.random.uniform(*self._shift_y)
        shift = [shift_x, shift_y] * 2

        image = Image.open(self._path)
        crop = image.crop(np.array(self._crop) + shift)

        boxes = self.boxes - shift

        draw = ImageDraw.Draw(crop)
        for i, box in enumerate(boxes):
            if self.anchors is not None:
                x, y = [0.5, 0.5] @ box.reshape((2, 2))

                w, h = self.anchors[i]
            
                x0 = (x / self.width - w / 2) * crop.width
                y0 = (y / self.height - h / 2) * crop.height
                x1 = (x / self.width + w / 2) * crop.width
                y1 = (y / self.height + h / 2) * crop.height

                draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 255), width=3)

            draw.rectangle([*box], outline=(255, 255, 0), width=3)

        return crop

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
    def __init__(self, path, use_labels=[], transform=lambda x: x, 
                 anchors=None, k=3):
        """
        path: path to folder with labelimg .xml files

        use_labels: list of labels that you want to use (others will be ignored)

        transform: an object implementing __call__ method.
        It's only used to transform images, boxes are not affected.
        (do not use translational/deformational/distortional transformations)
        """
        samples, itos, stoi = self._load_images(path, use_labels)
        self.samples = samples
        self.itos = itos
        self.stoi = stoi
        self.transform = transform

        if anchors is None:
            norm_boxes = np.concatenate([s.norm_boxes for s in samples], axis=0)
            anchors = self._k_means_anchors(norm_boxes, k)
        else:
            anchors = np.array(anchors)

        for s in samples:
            boxes = [-1.0, 1.0] @ s.norm_boxes.reshape(-1, 2, 2)
            indices = [np.argmax([IOU(a, b) for a in anchors]) 
                                                for b in boxes]
            s.anchors = [anchors[i] for i in indices]

    def mean_iou(self):
        iou_list = []
        for s in self.samples:
            box_sizes = [-1.0, 1.0] @ s.norm_boxes.reshape((-1, 2, 2))
            for a_wh, b_wh in zip(s.anchors, box_sizes):
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

    def _k_means_anchors(self, boxes, k):
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

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        image_wb = self.samples[index]
        image, y = image_wb(16, 16, self.stoi)
        x = self.transform(image)
        return x, y
