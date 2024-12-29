# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import numbers
import pandas as pd
import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import os
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import InterpolationMode

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.loader = loader
        self.train=train
        patch=1
        if train:
            csv_file = r"/media/xqh/data/train.csv"
        else:
            csv_file = r"/media/xqh/data/test.csv"
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root
        data = []
        for i in range(len(self.annotations)):
            # 获取索引为 index 的行的所有数据，并存入一个列表中

            row_data = str(self.annotations.iloc[i, 0])
            # 将该行数据添加到 data 列表中
            data.append(row_data)
        mos = []
        for i in range(len(self.annotations)):
            # 获取索引为 index 的行的所有数据，并存入一个列表中
            annotations = self.annotations.iloc[i, 1]
            annotations = annotations.astype('float').reshape(-1, 1)

            # 取出嵌套列表中的第一个元素
            annotations = annotations[0]
            annotations = annotations / 10
            annotations.astype('float32')
            # 将该行数据添加到 data 列表中
            mos.append(annotations)
        mos = torch.tensor(mos)
        sample = []
        for i in range(len(self.annotations)):
            if train:
                for aug in range(patch):
                    data_value = data[i]
                    mos_value = mos[i]
                    sample.append((os.path.join(self.root_dir, str(data_value)), mos_value))
            else:
                data_value = data[i]
                mos_value = mos[i]
                sample.append((os.path.join(self.root_dir, str(data_value)), mos_value))
        self.samples = sample
        self.target_transform = target_transform
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        x=sample
        width, height = sample.size
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
        return sample, target
#     # __getitem__ and __len__ inherited from ImageFolder
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    if args.data_set == 'ICAA17K':

        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                               transform=transform)
        nb_classes=1000
    if args.data_set == 'AVA':
        a="/media/xqh/data/AVA_Files/AVA_data_official_test.csv"
        b="/media/xqh/data/images"
        dataset = AVADataset(a,b, is_train)
        nb_classes=1000
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset,nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        # transform = create_transform(
        #     input_size=args.input_size,
        #     is_training=True,
        #     color_jitter=args.color_jitter,
        #     auto_augment=args.aa,
        #     interpolation=args.train_interpolation,
        #     re_prob=args.reprob,
        #     re_mode=args.remode,
        #     re_count=args.recount,
        # )
        # if not resize_im:
        #     # replace RandomResizedCropAndInterpolation with
        #     # RandomCrop
        #     transform.transforms[0] = transforms.RandomCrop(
        #         args.input_size, padding=4)
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize((224,224)),
        )
        t.append(transforms.ToTensor())
    t.append(transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    return transforms.Compose(t)
class Crop_patches(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, input_size, img_shape):

        img = img.astype(dtype=np.float32)
        if len(img_shape) == 2:
            H, W, = img_shape
            ch = 1
        else:
            H, W, ch = img_shape
        if ch == 1:
            img = np.asarray([img, ] * 3, dtype=img.dtype)

        stride = int(input_size / 2)
        hIdxMax = H - input_size
        wIdxMax = W - input_size
        return H, W, stride, hIdxMax, wIdxMax

    def __call__(self, image):
        input_size = 224

        img = self.to_numpy(image)
        img_shape = img.shape
        H, W, stride, hIdxMax, wIdxMax = self.get_params(img, input_size, img_shape)

        hIdx = [i * stride for i in range(int(hIdxMax / stride) + 1)]
        if H - input_size != hIdx[-1]:
            hIdx.append(H - input_size)
        wIdx = [i * stride for i in range(int(wIdxMax / stride) + 1)]
        if W - input_size != wIdx[-1]:
            wIdx.append(W - input_size)
        patches_numpy = [img[hId:hId + input_size, wId:wId + input_size, :]
                         for hId in hIdx
                         for wId in wIdx]
        patches_tensor = [
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(transforms.ToTensor()(p)) for p
            in patches_numpy]
        patches_tensor = torch.stack(patches_tensor, 0).contiguous()

        return patches_tensor

    def to_numpy(self, image):
        p = image.numpy()
        return p.transpose((1, 2, 0))
#
# class INatDataset(ImageFolder):
#     def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
#                  category='name', loader=default_loader):
#         self.loader = loader
#         self.train=train
#         patch=1
#         if train:
#             csv_file = r"/media/xqh/data/spaqTrain.csv"
#         else:
#             csv_file = r"/media/xqh/data/spaqTest.csv"
#         self.annotations = pd.read_csv(csv_file)
#         self.root_dir = root
#         data = []
#         for i in range(len(self.annotations)):
#             # 获取索引为 index 的行的所有数据，并存入一个列表中
#
#             row_data = str(self.annotations.loc[i, 'image_id'])
#             # 将该行数据添加到 data 列表中
#             data.append(row_data)
#         mos = []
#         for i in range(len(self.annotations)):
#             # 获取索引为 index 的行的所有数据，并存入一个列表中
#             annotations = self.annotations.loc[i,'Colorfulness']
#             annotations = annotations.astype('float').reshape(-1, 1)
#
#             # 取出嵌套列表中的第一个元素
#             annotations = annotations[0]
#             annotations = annotations / 10
#             annotations.astype('float32')
#             # 将该行数据添加到 data 列表中
#             mos.append(annotations)
#         mos = torch.tensor(mos)
#         sample = []
#         for i in range(len(self.annotations)):
#             if train:
#                 for aug in range(patch):
#                     data_value = data[i]
#                     mos_value = mos[i]
#                     sample.append((os.path.join(self.root_dir, str(data_value)), mos_value))
#             else:
#                 data_value = data[i]
#                 mos_value = mos[i]
#                 sample.append((os.path.join(self.root_dir, str(data_value)), mos_value))
#
#         self.samples = sample
#         self.transform = transform
#
#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         if self.transform is not None:
#             sample = self.transform(sample)
#         # if not self.train:
#         #     resize_transform = transforms.Compose([
#         #         transforms.Resize((224, 224)),
#         #         transforms.ToTensor()
#         #     ])
#         #     sample1 = resize_transform(x)
#         #     sample1 = torch.unsqueeze(sample1, 0)  # 添加额外的维度
#         #     sample = torch.cat((sample1, sample), dim=0)
#         # 自定义处理或转换逻辑
#         # 在这里可以对样本或目标进行额外的处理
#
#         return sample, target
#     # __getitem__ and __len__ inherited from ImageFolder
