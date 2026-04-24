import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from configs import mean, std


def get_loaders(batch_size, num_workers):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.5,0.5,0.1,0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    train_full_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=test_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    indices = list(range(len(train_full_aug)))
    split = int(0.8*len(indices))

    train_idx = indices[:split]
    val_idx = indices[split:]

    train_set = Subset(train_full_aug, train_idx)
    val_set = Subset(train_full, val_idx)

    train_loader = DataLoader(train_set,batch_size,shuffle=True,num_workers=num_workers)
    val_loader = DataLoader(val_set,batch_size,shuffle=False,num_workers=num_workers)
    test_loader = DataLoader(test_set,batch_size,shuffle=False,num_workers=num_workers)

    return train_loader, val_loader, test_loader