from io import BytesIO
import json
import base64
import pathlib
import collections
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import albumentations as alb
from PIL import Image, ImageDraw


class ToTensor(alb.BasicTransform):
    def __init__(self):
        super().__init__(always_apply=True)
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def apply(self, image, **params):
        return self.image_transform(image)

    def apply_to_mask(self, mask, **params):
        return transforms.ToTensor()(mask)

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
        }


class Transform:
    default_transform = transforms.Compose([
        transforms.Resize((1088, 1920)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # augmentation_transforms = [ToTensor()]
    # if self.augmentation:
    #     augmentation_transforms = [
    #         alb.HueSaturationValue(always_apply=True),
    #         alb.RandomBrightnessContrast(always_apply=True),
    #         alb.HorizontalFlip(),
    #         alb.RandomGamma(always_apply=True),
    #         alb.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, always_apply=True),
    #         alb.PadIfNeeded(min_height=image_height, min_width=image_width, always_apply=True),
    #         alb.RandomCrop(image_height, image_width, always_apply=True),
    #     ] + augmentation_transforms
    # else:
    #     augmentation_transforms = [
    #         alb.PadIfNeeded(min_height=image_height, min_width=image_width, always_apply=True),
    #         alb.CenterCrop(image_height, image_width, always_apply=True),
    #     ] + augmentation_transforms

    # augmentation_transforms = [ToTensor()] + augmentation_transforms
    # self.transforms = alb.Compose(transforms=augmentation_transforms)


class LabelMeDataset(data.Dataset):
    def __init__(self, directory: str, transform=None, image_height: int = 1088, image_width: int = 1920):
        self.directory = pathlib.Path(directory)
        assert self.directory.exists()
        assert self.directory.is_dir()

        self.width, self.height = image_width, image_height
        self.transform = transform
        self.toTensor = transforms.Compose([transforms.ToTensor()])
        self.resize = transforms.Compose([transforms.Resize((self.height, self.width), Image.NEAREST)])
        self.toTensorResize = transforms.Compose([transforms.Resize((self.height, self.width)), transforms.ToTensor()])

        self.labelme_paths = []
        self.categories = collections.defaultdict(list)

        required_keys = ['version', 'flags', 'shapes', 'imagePath', 'imageData', 'imageHeight', 'imageWidth']
        for labelme_path in self.directory.rglob('*.json'):
            with open(labelme_path, 'r') as labelme_file:
                labelme_json = json.load(labelme_file)

                assert all(key in labelme_json for key in required_keys), (required_keys, labelme_json.keys())

                self.labelme_paths.append(labelme_path)

                for shape in labelme_json['shapes']:
                    label = shape['label']
                    self.categories[label].append(labelme_path)

        self.categories = sorted(list(self.categories.keys()))
        self.un_class = 1 + len(self.categories)  # +1 for background

    def __len__(self):
        return len(self.labelme_paths) - 1

    def __getitem__(self, idx: int):
        labelme_path = self.labelme_paths[idx]

        with open(labelme_path, 'r') as labelme_file:
            labelme_json = json.load(labelme_file)

        image_width = labelme_json['imageWidth']
        image_height = labelme_json['imageHeight']

        image = Image.open(BytesIO(base64.b64decode(labelme_json['imageData'])))
        assert image.size == (image_width, image_height) or (1, 1) == (image_width, image_height)

        num_categories = len(self.categories)
        mask = torch.zeros((num_categories, self.height, self.width), dtype=torch.long)

        # label_color = [(1+label)*int(255/num_categories) for label in range(num_categories)]  # +1 for background
        label_color = [(1 + label) for label in range(num_categories)]  # 0 is for background

        for shape in labelme_json['shapes']:
            label = self.categories.index(shape['label'])

            points = tuple(map(tuple, shape['points']))  # torch.tensor(shape['points'], dtype=torch.int)
            img_mask = Image.new('L', image.size, 0)
            raster = ImageDraw.Draw(img_mask)
            raster.polygon(points, 255)  # label_color[label]
            mask[label] = torch.max(mask[label], label_color[label] * self.toTensorResize(img_mask).squeeze().long())
            # mask[label] += label_color[label] * self.toTensorResize(img_mask).squeeze().long()
            # mask[label] += torch.tensor(np.array(self.resize(img_mask)))

        if self.transform:
            image = self.transform(image)
        return self.toTensorResize(image), torch.max(mask, dim=0).values

    def getpath(self, idx: int):
        return self.labelme_paths[idx]

    def newpath(self, idx, out=None, extension=None):
        if extension and out:
            return pathlib.Path.joinpath(pathlib.Path(out), pathlib.Path(self.getpath(idx)).name).with_suffix(extension)
        if extension:
            return pathlib.Path(self.getpath(idx)).with_suffix(extension)
        if out:
            return pathlib.Path.joinpath(pathlib.Path(self.getpath(idx)), pathlib.Path(self.getpath(idx)).name)
        return pathlib.Path(self.getpath(idx)).name
