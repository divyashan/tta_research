
import torch
import torchvision
from torchvision import transforms

TRANSFORM_MAP = {'cropll': lambda x: transforms.functional.crop(x, 32, 0, 224, 224),
                 'croplr':  lambda x: transforms.functional.crop(x, 32, 32, 224, 224),
                 'cropur':  lambda x: transforms.functional.crop(x, 0, 32, 224, 224),
                 'cropul':  lambda x: transforms.functional.crop(x, 0, 0, 224, 224),
                 'hflip': transforms.RandomHorizontalFlip(p=1)}

def get_flowers_dataloader(batch_size=64):
    image_size = 256
    crop_size = 224
    normalize = transforms.Normalize(mean=[0.5208, 0.4205, 0.3441],
                                     std=[0.2944, 0.2465, 0.2735])

    shuffle=False
    data_path= './datasets/flowers102/test'
    
    d_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize
    ])
    
    
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=d_transforms)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle)
    return dataloader


def get_flowers_dataloader_with_aug(aug_name, batch_size=64):
    image_size = 256
    crop_size = 224
    normalize = transforms.Normalize(mean=[0.5208, 0.4205, 0.3441],
                                     std=[0.2944, 0.2465, 0.2735])

    shuffle=False
    data_path= './datasets/flowers102/test'
    
    crop_transform = transforms.CenterCrop(image_size)
    if aug_name.startswith('crop'):
        crop_transform = TRANSFORM_MAP[aug_name]
    
    if aug_name == 'none':
        d_transforms = transforms.Compose([
            transforms.Resize(image_size),
            crop_transform,
            transforms.ToTensor(),
            normalize
        ])
    else:
        d_transforms = transforms.Compose([
            transforms.Resize(image_size),
            TRANSFORM_MAP[aug_name],
            crop_transform,
            transforms.ToTensor(),
            normalize
        ])
    
    
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=d_transforms)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle)
    return dataloader
