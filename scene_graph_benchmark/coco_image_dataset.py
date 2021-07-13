import torch
import torch.utils.data as udata
import os
import cv2
from PIL import Image
import json

class ExtractCocoDataset(udata.Dataset):
    def __init__(self, image_dir, transforms):
        # super(ExtractCocoDataset, self).__init__()
        self.image_dir = image_dir

        self.images = []
        self.train_dir = os.path.join(image_dir, 'train2014')
        self.val_dir = os.path.join(image_dir, 'val2014')

        for img_file in os.listdir(self.train_dir):
            img_id = str(int(img_file.split('_')[-1][:-4]))
            path = os.path.join(self.train_dir, img_file)
            self.images.append((img_id, path))

        for img_file in os.listdir(self.val_dir):
            img_id = str(int(img_file.split('_')[-1][:-4]))
            path = os.path.join(self.val_dir, img_file)
            self.images.append((img_id, path))

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id, img_file = self.images[idx]
        cv2_img = cv2.imread(img_file)
        img_height = cv2_img.shape[0]
        img_width = cv2_img.shape[1]
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img, _ = self.transforms(img, target=None)
        return img, img_height, img_width, img_id


class ExtractCocoDatasetTargetPath(udata.Dataset):
    def __init__(self, images, transforms):
        # super(ExtractCocoDataset, self).__init__()
        self.image_dir = "/home/gzx/data/Dataset/image_caption/coco/images/val2014"
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # img_id, img_file = self.images[idx]
        img_file = os.path.join(self.image_dir, self.images[idx])
        cv2_img = cv2.imread(img_file)
        img_height = cv2_img.shape[0]
        img_width = cv2_img.shape[1]
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img, _ = self.transforms(img, target=None)
        # print(img.shape)
        return img, img_height, img_width


class ExtractVizwizDataset(udata.Dataset):
    def __init__(self, image_dir, data_json, transforms):
        self.image_dir = image_dir
        with open(data_json, 'r') as f:
            self.raw_json = json.load(f)['images']

        self.images = [k for k in self.raw_json]
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.images[idx]
        img_file = os.path.join(self.image_dir, self.raw_json[img_id]['filename'])
        cv2_img = cv2.imread(img_file)
        img_height = cv2_img.shape[0]
        img_width = cv2_img.shape[1]
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img, _ = self.transforms(img, target=None)
        return img, img_height, img_width, img_id


if __name__ == '__main__':
    # from maskrcnn_benchmark.data.transforms import build_transforms
    # transforms = build_transforms(cfg, is_train=False)
    # ExtractCocoDataset("/home/gzx/data/Dataset/image_caption/coco/images", "", transforms)
    pass