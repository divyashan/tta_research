import torch
import os
from torchvision import datasets
from torchvision import transforms


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)    
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res
    
    
def get_cifar100_dataloader(train=False, batch_size=32):
    data_root='/tmp/public_dataset/pytorch'
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    dataset = datasets.CIFAR100(
                root=data_root, train=train, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                    transforms.Pad(4)),
                ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
    return dataloader

def get_cifar100_dataloader_hflip(train=False, batch_size=32):
    data_root='/tmp/public_dataset/pytorch'
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    dataset = datasets.CIFAR100(
                root=data_root, train=train, download=True,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(1.0),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                    transforms.Pad(4)),
                ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
    return dataloader

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
