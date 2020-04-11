import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt

from models import ResNet
from datasets import load_MNIST
from train_model import train, val_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

binarize = True             # Binarize labels or not
noise_factor = None         # Probability/Percentage of noise corruption in the labels
lr = 0.05
epochs = 10
bs = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
trainset, trainloader, testset, testloader = load_MNIST(bs, noise_factor=noise_factor, binarize=binarize)

# Model
print('==> Building model..')

net = ResNet(input_dim=784, inter_dim=200, output_dim=100, depth=4, theta=1)
net = net.to(device)

writer = SummaryWriter()
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.functional.binary_cross_entropy_with_logits
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training

for epoch in range(epochs):
    _, _ = train(epoch, net, optimizer, device, trainloader, criterion, writer)
    _, _ = val_score(epoch, net, device, testloader, criterion)

writer.close()