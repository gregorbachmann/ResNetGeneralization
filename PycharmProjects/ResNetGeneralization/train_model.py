import torch

from utils import surrogate_error


def train(epoch, net, optimizer, device, trainloader, criterion, writer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_surr = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets.double())
        surr = surrogate_error(outputs, targets.double())
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        train_surr += surr.item() * inputs.size(0)
        predicted = (torch.sign(outputs) + 1) / 2
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        net.weight_norms(writer, epoch * len(trainloader) + batch_idx)
    epoch_loss = train_loss / total
    epoch_acc = correct / total
    epoch_surr = train_surr / total

    print('{} Loss: {:.4f} Acc: {:.4f} Surr: {:.4f}'.format(
        'Training', epoch_loss, epoch_acc, epoch_surr))

    return epoch_loss, epoch_acc


def val_score(epoch, net, device, testloader, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets.double())
            test_loss += loss.item() * inputs.size(0)
            predicted = (torch.sign(outputs) + 1) / 2
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        epoch_loss = test_loss / total
        epoch_acc = correct / total

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'Test', epoch_loss, epoch_acc))

    return epoch_loss, epoch_acc