import os
import time

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from timm.data import create_transform, create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def dataloader_factory(args):
    if args.dataset == "mnist":
        train_dataset, test_dataset = get_mnist_dataset(args)
        num_classes = 10
        input_shape = (1, 28, 28)
    elif args.dataset == "cifar10":
        train_dataset, test_dataset = get_cifar10_dataset(args)
        num_classes = 10
        input_shape = (3, 32, 32)
    elif args.dataset == "cifar100":
        train_dataset, test_dataset = get_cifar100_dataset(args)
        num_classes = 100
        input_shape = (3, 32, 32)
    elif args.dataset == "tinyimagenet":
        train_dataset, test_dataset = get_tinyimagenet_dataset(args)
        num_classes = 200
        input_shape = (3, 64, 64)
    elif args.dataset == "imagenet":
        train_dataset, test_dataset = get_imagenet_dataset(args)
        num_classes = 1000
        input_shape = (3, 224, 224)
    else:
        raise NotImplementedError(args.dataset)
    
    val_dataloader = None
    if 0 < args.split and args.split < 1:
        n_samples = len(train_dataset)
        train_size = int(n_samples * args.split)
        val_size = int(n_samples - train_size)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_dataset, 
                                                                   lengths=(train_size, val_size))

    mixup_fn = None
    if args.mixup > 0:
        mixup_arg = dict(mixup_alpha=args.mixup,
                         cutmix_alpha=1.0, 
                         cutmix_minmax=None, 
                         prob=1.0, 
                         switch_prob=0.5, 
                         mode='batch', 
                         label_smoothing=0.1,
                         num_classes=num_classes)
        mixup_fn = Mixup(**mixup_arg)

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True, 
                                  batch_size=args.batch_size, 
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch,
                                  pin_memory=args.pin_memory,
                                  drop_last=True)
    if val_dataset is not None:
        val_dataloader = DataLoader(val_dataset,
                                    shuffle=True, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers, 
                                    prefetch_factor=args.prefetch,
                                    pin_memory=args.pin_memory)
        
    test_dataloader = DataLoader(test_dataset,
                                 shuffle=False,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 prefetch_factor=args.prefetch,
                                 pin_memory=args.pin_memory)
    
    return [train_dataloader, val_dataloader, test_dataloader, num_classes, input_shape, mixup_fn]

def get_mnist_dataset(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5), (0.5)),
    ])

    train = torchvision.datasets.MNIST(root=args.data_path,
                                       train=True,
                                       download=False,
                                       transform=transform)
    test = torchvision.datasets.MNIST(root=args.data_path,
                                      train=False,
                                      download=False,
                                      transform=transform)
    
    return train, test


def get_cifar10_dataset(args):
    if args.snn:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train = torchvision.datasets.CIFAR10(root=args.data_path,
                                         train=True,
                                         download=False,
                                         transform=train_transform)
    test = torchvision.datasets.CIFAR10(root=args.data_path,
                                        train=False,
                                        download=False,
                                        transform=test_transform)
    
    return train, test


def get_cifar100_dataset(args):
    if args.snn:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train = torchvision.datasets.CIFAR100(root=args.data_path,
                                          train=True,
                                          download=False,
                                          transform=train_transform)
    test = torchvision.datasets.CIFAR100(root=args.data_path,
                                         train=False,
                                         download=False,
                                         transform=test_transform)
    
    return train, test

def get_tinyimagenet_dataset(args):
    if args.snn:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train = torchvision.datasets.ImageFolder(root=os.path.join(args.data_path, 'tiny-imagenet-200/train'),
                                             transform=train_transform)
    
    test = torchvision.datasets.ImageFolder(root=os.path.join(args.data_path, 'tiny-imagenet-200/val'),
                                            transform=test_transform)
    
    return train, test                      

class Imagenet_A_dataset(torchvision.datasets.ImageNet):
    def __init__(self, root, split="train", **kwargs):
        super().__init__(root, split, **kwargs)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # start_ = time.time()
        path, target = self.samples[index]
        # sample_time = time.time()
        sample = self.loader(path)
        # load_time = time.time()
        if self.transform is not None:
            sample_np = np.array(sample)
            # sample = self.transform(sample)
            sample_np = self.transform(image=sample_np)
            sample = sample_np['image']
        # transform_time = time.time()
        # print(sample_time-start_, load_time-sample_time, transform_time-load_time)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

# def get_imagenet_dataset(args):
#     train = torchvision.datasets.ImageNet(root=os.path.join(args.data_path, 'imagenet'), split='train')

#     test_transform = transforms.Compose([
#         transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#     test = torchvision.datasets.ImageNet(root=os.path.join(args.data_path, 'imagenet'), split='val', transform=test_transform)

#     return train, test
def get_imagenet_dataset(args):
    if args.snn:
        train_transform = A.Compose([
            A.RandomResizedCrop(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2()
        ])
    else:
        train_transform = A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(),
            A.HueSaturationValue(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2()
        ])
    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    train = Imagenet_A_dataset(root=os.path.join(args.data_path, 'imagenet'),
                               split='train',
                               transform=train_transform)
    
    test = torchvision.datasets.ImageNet(root=os.path.join(args.data_path, 'imagenet'),
                                         split='val',
                                         transform=test_transform)
    
    return train, test

# def get_imagenet_dataset(args):
#     if args.snn:
#         train_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#         ])
#     else:
#         train_transform = create_transform(
#             input_size=224,
#             is_training=True,
#             color_jitter=0.4,
#             auto_augment="rand-m9-mstd0.5-inc1",
#             re_prob=0.25,
#             re_mode="pixel",
#             re_count=1,
#             interpolation="bicubic"
#         )
#     val_transform = transforms.Compose([
#         transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])

#     # train = CachedImageFolder(os.path.join(args.data_path, 'imagenet'),
#     #                           "train_map.txt", "train.zip@/", 
#     #                           train_transform)
#     # test = CachedImageFolder(os.path.join(args.data_path, 'imagenet'),
#     #                          "validation_map.txt", "val.zip@/",
#     #                          val_transform)
    
#     train = create_dataset(
#         "imagenet",
#         root=os.path.join(args.data_path, 'imagenet'), split="train", is_training=True,
#         batch_size=args.batch_size, repeats=1, transform=train_transform)
#     test = create_dataset(
#         "imagenet",
#         root=os.path.join(args.data_path, 'imagenet'), split="val", is_training=False,
#         batch_size=args.batch_size, repeats=1, transform=val_transform)

#     return train, test
                              

if __name__ == "__main__":
    from argument import get_args
    args = get_args()
    train_dataloader, val_dataloader, test_dataloader, num_classes, in_shapes, mixup_fn = dataloader_factory(args)
    print(vars(train_dataloader))
    print(vars(train_dataloader.dataset))
    print(train_dataloader.dataset.transform)
    print(type(train_dataloader.dataset))
    print(train_dataloader.dataset[0])

    # dataset_train = create_dataset(
    #     args.dataset,
    #     root=args.data_dir, split=args.train_split, is_training=True,
    #     batch_size=args.batch_size, repeats=args.epoch_repeats)

    # # setup mixup / cutmix
    # collate_fn = None
    # mixup_args = dict(mixup_alpha=args.mixup,
    #                  cutmix_alpha=1.0, 
    #                  cutmix_minmax=None, 
    #                  prob=1.0, 
    #                  switch_prob=0.5, 
    #                  mode='batch', 
    #                  label_smoothing=0.1,
    #                  num_classes=num_classes)
    # collate_fn = FastCollateMixup(**mixup_args)

    # # create data loaders w/ augmentation pipeiine
    # train_interpolation = 'bicubic'
    # loader_train = create_loader(
    #     dataset_train,
    #     input_size=224,
    #     batch_size=args.batch_size,
    #     is_training=True,
    #     use_prefetcher=True,
    #     no_aug=args.no_aug,
    #     re_prob=args.reprob,
    #     re_mode=args.remode,
    #     re_count=args.recount,
    #     re_split=args.resplit,
    #     scale=args.scale,
    #     ratio=args.ratio,
    #     hflip=args.hflip,
    #     vflip=args.vflip,
    #     color_jitter=args.color_jitter,
    #     auto_augment='rand-m9-mstd0.5-inc1',
    #     num_aug_splits=num_aug_splits,
    #     interpolation=train_interpolation,
    #     num_workers=8,
    #     collate_fn=collate_fn,
    #     pin_memory=True,
    # )

    # print(train_dataloader.dataset.transform)
    # print(type(train_dataloader.dataset))
    # print(type(train_dataloader.dataset[0]))

    print("dataloaded")
    for i, (images, labels) in enumerate(train_dataloader):
        print(i)
        print(images.size(), labels.size())
        # images = images.to(device, non_blocking=True)
        # labels = labels.to(device, non_blocking=True)