import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


def load_MNIST(bs, seed=1234, noise_factor=None, binarize=False):
    print('==> Preparing data..')
    np.random.seed(seed)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)

    if noise_factor is not None:
        num_random = int(trainset.data.shape[0] * noise_factor)
        copy = trainset.targets
        random_indices = np.random.choice(range(trainset.data.shape[0]), size=num_random, replace=False)
        random_labels = np.random.choice(range(10), size=num_random)
        for i in range(num_random):
            copy[random_indices[i]] = torch.tensor(random_labels[i]).double()
        trainset.targets = copy

    if binarize:
        trainset.targets[trainset.targets <= 4] = 1
        trainset.targets[trainset.targets > 4] = 0
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    if binarize:
        testset.targets[testset.targets <= 4] = 1
        testset.targets[testset.targets > 4] = 0
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)

    return trainset, trainloader, testset, testloader


def load_cifar(bs, random=False, seed=1234, noise_factor=None):
    print('==> Preparing data..')
    np.random.seed(seed)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    if random:
        num_random = int(trainset.data.shape[0] * noise_factor)
        copy = trainset.targets.copy()
        random_indices = np.random.choice(range(trainset.data.shape[0]), size=num_random, replace=False)
        random_labels = np.random.choice(range(10), size=num_random)
        for i in range(num_random):
            copy[random_indices[i]] = random_labels[i]
        trainset.targets = copy

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)

    return trainset, trainloader, testset, testloader