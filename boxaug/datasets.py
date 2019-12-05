import os
import numpy as np
from PIL import Image
from lxml import etree
from boxaug import transforms


class LabelImgDataset():
    def __init__(self, path, transform=None, use_labels=None):
        """
        Args:
            path: path to folder with LabelImg .xml files

            use_labels: bounding boxes whose labels are not in this list
            will be ignored.

            transform: see boxaug.transforms
        """
        samples, labels_set = self._load_samples(path, use_labels)

        itos = list(labels_set)
        stoi = dict((x, i) for i, x in enumerate(itos))

        self.samples = samples
        self.itos = itos
        self.stoi = stoi
        self.tfm = transform or transforms.Identity()

    def _load_samples(self, path, use_labels=None):
        samples = []
        labels_set = set()

        for filename in os.listdir(path):
            if filename.endswith('xml'):
                xml_path = os.path.join(path, filename)

                tree = etree.parse(xml_path)

                im_path = tree.find('path').text

                boxes = []
                labels = []

                for obj in tree.findall('object'):
                    x0 = int(obj.find('bndbox/xmin').text)
                    y0 = int(obj.find('bndbox/ymin').text)
                    x1 = int(obj.find('bndbox/xmax').text)
                    y1 = int(obj.find('bndbox/ymax').text)
                    label = obj.find('name').text

                    if use_labels is None or label in use_labels:
                        boxes.append([x0, y0, x1, y1])
                        labels.append(label)

                if boxes:
                    samples.append((im_path, np.array(boxes), labels))
                    labels_set.update(labels)

        return samples, labels_set

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, bboxes, labels = self.samples[index]

        image = Image.open(path).convert('RGB')
        image_arr = np.asarray(image)

        xy_array, wh_array = transforms.split_bboxes(bboxes)

        image_out, xy_array_out = self.tfm(image_arr, xy_array)

        w_ratio = image_out.shape[1]/image_arr.shape[1]
        h_ratio = image_out.shape[0]/image_arr.shape[0]

        wh_array_out = wh_array * [w_ratio, h_ratio]

        bboxes_out = transforms.merge_bboxes(xy_array_out, wh_array_out)

        return image_out, bboxes_out, labels
