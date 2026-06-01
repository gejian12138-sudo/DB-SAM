import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from utils.utils import get_boxes_from_mask, ResizeLongestSide
from config import config_dict


def _elastic_deform(image, mask, alpha=120, sigma=10):
    h, w = image.shape[:2]
    dx = (np.random.rand(h, w).astype(np.float32) * 2 - 1) * alpha
    dy = (np.random.rand(h, w).astype(np.float32) * 2 - 1) * alpha
    ksize = int(sigma * 6) | 1
    dx = cv2.GaussianBlur(dx, (ksize, ksize), sigma)
    dy = cv2.GaussianBlur(dy, (ksize, ksize), sigma)
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(x + dx, 0, w - 1)
    map_y = np.clip(y + dy, 0, h - 1)
    image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return image, mask


def _motion_blur(image, kernel_size, angle):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    M = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel /= kernel.sum() + 1e-6
    return cv2.filter2D(image, -1, kernel)


def _add_specular_highlight(image):
    h, w = image.shape[:2]
    overlay = image.astype(np.float32).copy()
    for _ in range(random.randint(1, 3)):
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        rx = random.randint(10, 60)
        ry = random.randint(10, 40)
        intensity = random.uniform(0.4, 0.9)
        spot = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(spot, (cx, cy), (rx, ry), random.uniform(0, 180), 0, 360, 1.0, -1)
        spot = cv2.GaussianBlur(spot, (21, 21), 0)
        for c in range(3):
            overlay[..., c] = (overlay[..., c] * (1 - spot * intensity) + 255 * spot * intensity)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def augment_image_and_mask(image, mask):
    image_size = config_dict['img_size']
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    angle = random.uniform(-90, 90)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    if random.random() < 0.5:
        image, mask = _elastic_deform(image, mask)
    scale = random.uniform(0.5, 2.0)
    target = int(image_size * scale)
    h, w = image.shape[:2]
    ratio = target / max(h, w)
    new_h = max(1, int(h * ratio + 0.5))
    new_w = max(1, int(w * ratio + 0.5))
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    h, w = image.shape[:2]
    pad_h = max(0, image_size - h)
    pad_w = max(0, image_size - w)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(123, 116, 104))
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    h, w = image.shape[:2]
    y0 = random.randint(0, h - image_size) if h > image_size else 0
    x0 = random.randint(0, w - image_size) if w > image_size else 0
    image = image[y0:y0 + image_size, x0:x0 + image_size]
    mask = mask[y0:y0 + image_size, x0:x0 + image_size]
    return image, mask


def augment_color(image):
    pil_img = TF.to_pil_image(image)
    if random.random() < 0.8:
        pil_img = TF.adjust_brightness(pil_img, random.uniform(0.6, 1.4))
        pil_img = TF.adjust_contrast(pil_img, random.uniform(0.6, 1.4))
        pil_img = TF.adjust_saturation(pil_img, random.uniform(0.6, 1.4))
        pil_img = TF.adjust_hue(pil_img, random.uniform(-0.1, 0.1))
    if random.random() < 0.2:
        pil_img = TF.to_grayscale(pil_img, num_output_channels=3)
    image = np.array(pil_img)
    if random.random() < 0.3:
        ksize = random.choice([3, 5, 7])
        image = cv2.GaussianBlur(image, (ksize, ksize), random.uniform(0.5, 2.0))
    if random.random() < 0.2:
        image = _motion_blur(image, kernel_size=random.choice([5, 7, 9]),
                             angle=random.uniform(0, 180))
    if random.random() < 0.3:
        image = _add_specular_highlight(image)
    return image


class PolypTrainDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int = config_dict['img_size'],
                 requires_name: bool = False):
        self.image_size = image_size
        self.requires_name = requires_name
        image_dir = os.path.join(data_dir, 'imgs')
        mask_dir = os.path.join(data_dir, 'gts')
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith('.png') or f.endswith('.jpg')])
        self.label_paths = [
            [os.path.join(mask_dir, os.path.basename(p))] for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.cvtColor(cv2.imread(self.image_paths[index]), cv2.COLOR_BGR2RGB)
        mask_path = self.label_paths[index][0]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask file: {mask_path}")
        if mask.max() == 255:
            mask = mask / 255
        mask = mask.astype('uint8')
        for _ in range(10):
            aug_image, aug_mask = augment_image_and_mask(image, mask)
            if aug_mask.sum() > 0:
                break
        else:
            aug_image = cv2.resize(image, (self.image_size, self.image_size),
                                   interpolation=cv2.INTER_LINEAR)
            aug_mask = cv2.resize(mask, (self.image_size, self.image_size),
                                  interpolation=cv2.INTER_NEAREST)
        image, mask = aug_image, aug_mask
        image = augment_color(image)
        img_t = torch.as_tensor(image).permute(2, 0, 1).contiguous()
        boxes = get_boxes_from_mask(mask, max_pixel=0)
        mask_t = torch.from_numpy(mask)
        sample = {
            "image": img_t,
            "original_size": (self.image_size, self.image_size),
            "labels": mask_t.unsqueeze(0).unsqueeze(0),
            "boxes": boxes.unsqueeze(0),
            "point_coords": None,
            "point_labels": None,
        }
        if self.requires_name:
            sample["name"] = os.path.basename(self.image_paths[index])
        return sample


class PolypTestDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int = config_dict['img_size'],
                 requires_name: bool = False):
        self.image_size = image_size
        self.requires_name = requires_name
        image_dir = os.path.join(data_dir, 'imgs')
        mask_dir = os.path.join(data_dir, 'gts')
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith('.png') or f.endswith('.jpg')])
        self.label_paths = [os.path.join(mask_dir, os.path.basename(p))
                            for p in self.image_paths]

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        image = cv2.cvtColor(cv2.imread(self.image_paths[index]), cv2.COLOR_BGR2RGB)
        mask_path = self.label_paths[index]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask file: {mask_path}")
        if mask.max() == 255:
            mask = mask / 255
        mask = mask.astype(np.uint8)
        h, w = image.shape[:2]
        transforms = ResizeLongestSide(self.image_size)
        img_t = torch.as_tensor(transforms.apply_image(image)).permute(2, 0, 1).contiguous()
        boxes = transforms.apply_boxes_torch(get_boxes_from_mask(mask, max_pixel=0), (h, w))
        mask_resized = transforms.apply_image(mask)
        h_new, w_new = mask_resized.shape[:2]
        mask_t = F.pad(torch.from_numpy(mask_resized),
                       (0, self.image_size - w_new, 0, self.image_size - h_new)).unsqueeze(0)
        sample = {
            "image_path": self.image_paths[index],
            "image": img_t,
            "original_size": (h, w),
            "labels": mask_t.unsqueeze(0),
            "boxes": boxes,
            "label_path": os.path.dirname(mask_path),
            "point_coords": None,
            "point_labels": None,
        }
        if self.requires_name:
            sample["name"] = os.path.basename(mask_path)
        return sample


def collate_fn(batch):
    return batch
