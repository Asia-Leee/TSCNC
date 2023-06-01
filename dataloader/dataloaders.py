import os
import shutil

import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from timm.data.constants import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from timm.data import create_transform
def mnist(batch_size=100, pm=False):
    transf = [transforms.ToTensor()]
    if pm:
        transf.append(transforms.Lambda(lambda x: x.view(-1, 784)))
    transform_data = transforms.Compose(transf)

    kwargs = {'num_workers': 4, 'pin_memory': torch.cuda.is_available()}  # todo
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transform_data),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transform_data),
        batch_size=batch_size, shuffle=True, **kwargs)
    num_classes = 10

    return train_loader, val_loader, num_classes


def cifar10(root,augment=True, batch_size=128,):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    logging = 'Using'
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize
            ])
        logging += ' augmented'
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # normalize
        ])
    # if isnormalize:
    #     transform_train
    print(logging + ' CIFAR 10.')
    kwargs = {'num_workers': 8, 'pin_memory': torch.cuda.is_available()}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root, train=True, download=True,
                         transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root, train=False, transform=transform_test),
        batch_size=batch_size, shuffle=True, **kwargs)
    num_classes = 10

    return train_loader, val_loader, num_classes


def cifar100(root,augment=True, batch_size=128):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
                                     std=[x / 255.0 for x in [68.2, 65.4, 70.4]])
    logging = 'Using'
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
            ])
        logging += ' augmented'
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # normalize
        ])

    print(logging + ' CIFAR 100.')
    kwargs = {'num_workers': 8, 'pin_memory': torch.cuda.is_available()}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root, train=True, download=True,
                         transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root, train=False, transform=transform_test),
        batch_size=batch_size, shuffle=True, **kwargs)
    num_classes = 100

    return train_loader, val_loader, num_classes


def imagnenet_1k_torch(data_path,batch_size,seed):
    mean=IMAGENET_DEFAULT_MEAN
    std=IMAGENET_DEFAULT_STD
    num_class=1000
    input_size=224
    # train_dataloader
    transform_train= create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=mean,
            std=std,
        )
    print('transform_train:',transform_train)

    root_train = os.path.join(data_path,'train')
    datasets_train = torchvision.datasets.ImageFolder(root_train, transform=transform_train)
    sampler_train=torch.utils.data.DistributedSampler(datasets_train,num_replicas=1,rank=0,shuffle=True,seed=seed,)
    dataloader_train=torch.utils.data.DataLoader(
        datasets_train,sampler=sampler_train,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    #test_dataloader
    t=[]
    crop_pct=224/256
    size=256
    t.append(transforms.Resize(size,interpolation=3),)
    t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean,std))
    transform_test=transforms.Compose(t)
    print('transform_test:',transform_test)

    root_test = os.path.join(data_path,'ILSVRC2012_img_val')
    datasets_test = torchvision.datasets.ImageFolder(root_test, transform=transform_test)
    sampler_test=torch.utils.data.DistributedSampler(
        datasets_test, num_replicas=1, rank=0, shuffle=False)
    dataloader_test=torch.utils.data.DataLoader(
        datasets_test,sampler=sampler_test,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False)
    return dataloader_train,dataloader_test,num_class



def dataset_partition(list_path):
    root='../../../dataset/tiny-imagenet-200/val/'
    with open(list_path, 'r') as infile:
        for line in infile:
            img_filename = line.strip().split()[0]
            label_filename = line.strip().split()[1]
            directory=root+label_filename
            img=root+'images/'+img_filename
            img_copy=root+label_filename+'/'+img_filename
            if not os.path.exists(directory):
                os.makedirs(directory)
            shutil.copy(img,img_copy)

def tiny_imagenet_base(data_path,batch_size,seed):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    num_class = 200
    input_size = 64
    # train_dataloader
    transform_train = create_transform(
        input_size=input_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        mean=mean,
        std=std,
    )
    print('transform_train:', transform_train)

    root_train = os.path.join(data_path, 'train')
    datasets_train = torchvision.datasets.ImageFolder(root_train, transform=transform_train)
    sampler_train = torch.utils.data.DistributedSampler(datasets_train, num_replicas=1, rank=0, shuffle=True,
                                                        seed=seed, )
    dataloader_train = torch.utils.data.DataLoader(
        datasets_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # test_dataloader
    t = []
    crop_pct = 224 / 256
    size = int(input_size/crop_pct)
    t.append(transforms.Resize(size, interpolation=3), )
    t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    transform_test = transforms.Compose(t)
    print('transform_test:', transform_test)

    root_test = os.path.join(data_path, 'val')
    datasets_test = torchvision.datasets.ImageFolder(root_test, transform=transform_test)
    sampler_test = torch.utils.data.DistributedSampler(
        datasets_test, num_replicas=1, rank=0, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(
        datasets_test, sampler=sampler_test,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False)
    return dataloader_train, dataloader_test, num_class

def tiny_imagenet_adv(data_path,batch_size,seed):
    num_class = 200
    input_size = 64
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    normalize = transforms.Normalize(mean=mean,std=std)
    # train_dataloader
    transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    root_train = os.path.join(data_path, 'train')
    datasets_train = torchvision.datasets.ImageFolder(root_train, transform=transform_train)
    sampler_train = torch.utils.data.DistributedSampler(datasets_train, num_replicas=1, rank=0, shuffle=True,
                                                        seed=seed, )
    dataloader_train = torch.utils.data.DataLoader(
        datasets_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # test_dataloader
    t = []
    # crop_pct = 224 / 256
    # size = int(input_size/crop_pct)
    # t.append(transforms.Resize(size, interpolation=3), )
    # t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    transform_test = transforms.Compose(t)
    print('transform_test:', transform_test)

    root_test = os.path.join(data_path, 'val')
    datasets_test = torchvision.datasets.ImageFolder(root_test, transform=transform_test)
    sampler_test = torch.utils.data.DistributedSampler(
        datasets_test, num_replicas=1, rank=0, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(
        datasets_test, sampler=sampler_test,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False)
    return dataloader_train, dataloader_test, num_class






