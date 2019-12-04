import os
import numpy as np
from PIL import Image
from lxml import etree
from boxaug.utils import IOU
import boxaug.transforms as transforms


class LabelImgDataset():
    def __init__(self, path, use_labels, anchors, transform=None):
        """
        Args:
            path: path to folder with labelimg .xml files
            use_labels: labels that are not in this list will be ignored
            anchors: list of normalized (width, height) pairs
            transform: see boxaug.transforms
        """
        samples, itos, stoi = self._load_samples(path, use_labels, anchors)

        if transform is None:
            tfm = transforms.Identity()
        else:
            tfm = transform

        self.samples = samples
        self.itos = itos
        self.stoi = stoi
        self.tfm = tfm
        self.anchors = np.array(anchors)

    def _load_samples(self, path, use_labels, anchors):
        skipped_counter = 0

        samples = []
        all_labels = set()

        for filename in os.listdir(path):
            if filename.endswith('xml'):
                xml_path = os.path.join(path, filename)

                tree = etree.parse(xml_path)

                im_path = tree.find('path').text
                w = int(tree.find('size/width').text)
                h = int(tree.find('size/height').text)

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
                    skipped_counter += 1
                    continue

                boxes = np.array(boxes)

                boxes_wh = [-1.0, 1.0] @ boxes.reshape(-1, 2, 2)
                boxes_xy = [0.5, 0.5] @ boxes.reshape(-1, 2, 2)

                # Assign best (highest IOU) anchor box to each box
                anchor_indices = []
                for b_wh in boxes_wh / [w, h]:
                    i = np.argmax([IOU(b_wh, a_wh) for a_wh in anchors])
                    anchor_indices.append(i)

                image = (im_path, anchor_indices, boxes_xy, labels)

                samples.append(image)
                all_labels.update(labels)

        itos = list(all_labels)
        stoi = dict((label, i) for i, label in enumerate(itos))

        print('Images loaded:', len(samples))
        print('Images skipped:', skipped_counter)

        return samples, itos, stoi

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns:
            tuple(
                (np.ndarray) image
                (np.ndarray) center coordinates of boxes
                (np.ndarray) widths and heights of boxes
                (list) labels of boxes
            )
        """
        im_path, anchor_indices, boxes_xy, labels = self.samples[index]

        image = Image.open(im_path).convert('RGB')
        image_arr = np.asarray(image)

        tfm_image, tfm_boxes = self.tfm(image_arr, boxes_xy)

        anchor_boxes = np.array([self.anchors[i] for i in anchor_indices])

        return tfm_image, tfm_boxes, anchor_boxes, labels
