### About

Transforms similar to torchvision's, but applicable to bounding boxes as well.

```
boxaug.transforms.Compose
boxaug.transforms.Flip
boxaug.transforms.Affine
boxaug.transforms.Resize
boxaug.transforms.Crop
boxaug.transforms.Color
```

Check out two sample datasets.

```datasets.LabelImgDataset```: simple dataset that supports .xml files produced my LabelImg annotation tool (github.com/tzutalin/labelImg).

```datasets.AutoCropDataset```: adjust your dataset to the same aspect ratio and size keeping all bounding boxes in focus.

### Install

```
pip install boxaug
```

### Usage

See example of usage in ```example.ipynb```.
