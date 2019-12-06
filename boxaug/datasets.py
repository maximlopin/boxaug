import os
import numpy as np
from PIL import Image
from lxml import etree
from boxaug import transforms
from boxaug import utils


class Dataset():
    def __init__(self, samples, transform):
        self.samples = samples
        self.tfm = transform or transforms.Identity()

    def __getitem__(self, index):
        path, bboxes, labels = self.samples[index]

        image = Image.open(path).convert('RGB')
        image_arr = np.asarray(image)

        # Boxes to points
        bboxes_4pts = utils.bboxes_as_4pts(bboxes)
        points = bboxes_4pts.reshape(-1, 2)

        image_out, points_out = self.tfm(image_arr, points)

        # Points back to bboxes
        bboxes_4pts = points_out.reshape(-1, 8)
        bboxes_out = utils.bboxes_as_2pts(bboxes_4pts, align=True)

        return image_out, bboxes_out, labels

    def __len__(self):
        return len(self.samples)

    @classmethod
    def from_list(cls, samples, *args, **kwargs):
        """
        Args:
            samples: list of tuple(
                img_path (str),
                bboxes (np.ndarray of shape (N, 4)),
                labels (list of N labels)
            ) 
        """
        return cls(samples, *args, **kwargs)

    @classmethod
    def from_labelimg(cls, path, *args, **kwargs):
        """
        Args:
            path (str): path to directory with LabelImg .xml files
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

                    bboxes.append((x0, y0, x1, y1))
                    labels.append(label)

                samples.append((img_path, np.array(bboxes), labels))

        return cls(samples, *args, **kwargs)
