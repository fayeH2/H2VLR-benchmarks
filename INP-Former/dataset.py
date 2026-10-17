import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')

_IMAGE_SUFFIXES = ("*.png", "*.PNG", "*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.bmp", "*.BMP",
                   "*.tif", "*.tiff")
_MASK_SUFFIXES = ("*.png", "*.PNG", "*.bmp", "*.BMP", "*.jpg", "*.JPG")
_NORMAL_DIRS = {"good", "ok"}


def _glob_images(root):
    paths = []
    for pattern in _IMAGE_SUFFIXES:
        paths.extend(glob.glob(os.path.join(root, pattern)))
    return sorted(paths)


def _glob_masks(root):
    paths = []
    for pattern in _MASK_SUFFIXES:
        paths.extend(glob.glob(os.path.join(root, pattern)))
    return sorted(paths)


def _find_mask(mask_dir, image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    candidates = [stem + "_mask.png", stem + ".png", stem + "_mask.bmp", stem + ".bmp",
                  stem + "_mask.jpg", stem + ".jpg"]
    for name in candidates:
        candidate = os.path.join(mask_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type.lower() in _NORMAL_DIRS:
                img_paths = _glob_images(os.path.join(self.img_path, defect_type))
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))
            else:
                img_paths = _glob_images(os.path.join(self.img_path, defect_type))
                mask_dir = os.path.join(self.gt_path, defect_type)
                all_masks = _glob_masks(mask_dir)
                mask_by_stem = {}
                for mask_path in all_masks:
                    stem = os.path.splitext(os.path.basename(mask_path))[0]
                    key = stem[:-len("_mask")] if stem.endswith("_mask") else stem
                    mask_by_stem.setdefault(key, mask_path)
                gt_paths = []
                for image_path in img_paths:
                    stem = os.path.splitext(os.path.basename(image_path))[0]
                    mask_path = mask_by_stem.get(stem)
                    if mask_path is None:
                        mask_path = _find_mask(mask_dir, image_path)
                    if mask_path is None:
                        raise FileNotFoundError(
                            f"No mask for {image_path} under {mask_dir}"
                        )
                    gt_paths.append(mask_path)
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path



