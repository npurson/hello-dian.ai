import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

from PIL import Image


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bbox):
        for t in self.transforms:
            image, bbox = t(image, bbox)
        return image, bbox


class LoadImage(object):
    def __call__(self, image, bbox):
        image = Image.open(image)
        return image, bbox


class Resize(object):
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize((size, size))

    def __call__(self, image, bbox):
        bbox = [b * self.size / image.size[-1] for b in bbox]
        image = self.resize(image)
        return image, bbox


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, bbox):
        w, h = image.size[-2:]
        ratio = random.uniform(self.min_size, self.max_size)
        hp = int(h * ratio)
        wp = int(w * ratio)
        size = (hp, wp)
        image = F.resize(image, size)
        bbox = [b * hp / h for b in bbox]
        return image, bbox


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, bbox):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            bbox = [
                image.size[-1] - bbox[2], bbox[1], image.size[-1] - bbox[0],
                bbox[2]
            ]
        return image, bbox


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, bbox):
        image = pad_if_smaller(image, self.size[0])
        crop_params = transforms.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        bbox = [
            bbox[0] - crop_params[0], bbox[1] - crop_params[1],
            bbox[2] - crop_params[0], bbox[3] - crop_params[1]
        ]
        return image, bbox


class ToTensor(object):
    def __call__(self, image, bbox):
        image = F.to_tensor(image)
        bbox = torch.Tensor(bbox)
        return image, bbox


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, bbox):
        image = F.normalize(image, mean=self.mean, std=self.std)
        bbox /= 128
        return image, bbox
