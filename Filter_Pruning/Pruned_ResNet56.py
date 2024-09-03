import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self.make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet56():
    return ResNet(BasicBlock, [9, 9, 9])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_set = torchvision.datasets.CIFAR10("./data/train", train=True, transform=transform_train, download=True)
test_set = torchvision.datasets.CIFAR10("./data/test", train=False, transform=transform_test, download=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.01

train_dataLoader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_dataLoader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

net = ResNet56().to(DEVICE)

print("Original ResNet56:")
print(net)

def prune_conv_layer(layer, amount):
    # 提取第一个子层（卷积层）
    conv_layer = list(layer.children())[0]
    weight = conv_layer.weight.data.cpu().numpy()

    # 计算filter的L1范数
    magnitudes = np.sum(np.abs(weight), axis=(1, 2, 3))
    prune_num = int(amount * len(magnitudes))
    prune_indices = np.argsort(magnitudes)[:prune_num]

    new_conv = nn.Conv2d(
        in_channels=conv_layer.in_channels,
        out_channels=conv_layer.out_channels - prune_num,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None
    ).to(DEVICE)

    new_conv.weight.data = torch.from_numpy(
        np.delete(weight, prune_indices, axis=0) # weight的形状是[输入通道数, 输出通道数, 卷积核高, 卷积核宽]
    ).to(DEVICE)

    # 调整第一个BN层的参数
    bn_layer = list(layer.children())[1]

    new_bn = nn.BatchNorm2d(
        num_features=bn_layer.num_features - prune_num
    ).to(DEVICE)

    new_bn.weight.data = torch.from_numpy(
        np.delete(bn_layer.weight.data.cpu().numpy(), prune_indices)
    ).to(DEVICE)

    # 调整第二个卷积层的输入通道数
    conv_layer_2 = list(layer.children())[2]

    new_conv_2 = nn.Conv2d(
        in_channels=conv_layer_2.in_channels - prune_num,
        out_channels=conv_layer_2.out_channels,
        kernel_size=conv_layer_2.kernel_size,
        stride=conv_layer_2.stride,
        padding=conv_layer_2.padding,
        bias=conv_layer_2.bias is not None
    ).to(DEVICE)

    new_conv_2.weight.data = torch.from_numpy(
        np.delete(conv_layer_2.weight.data.cpu().numpy(), prune_indices, axis=1)
    ).to(DEVICE)

    return new_conv, new_bn, new_conv_2


def prune_resnet56(net, skip_layers, prune_ratios):
    layer_idx = 2
    for i, layer in enumerate([net.layer1, net.layer2, net.layer3]):
        for j in range(len(layer)):
            if layer_idx not in skip_layers:
                print(f"Pruning layer {layer_idx} with ratio {prune_ratios[i]}")
                layer[j].conv1, layer[j].bn1, layer[j].conv2 = prune_conv_layer(layer[j], prune_ratios[i])
            layer_idx += 2


skip_layers = [16, 18, 20, 34, 38, 54]
prune_ratios = [0.6, 0.3, 0.1]  # p1=60%, p2=30%, p3=10%


prune_resnet56(net, skip_layers, prune_ratios)
print("-" * 50)
print("Pruned ResNet56:")
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)


def train(epoch, model, train_dataLoader, criterion, optimizer, device):
    train_loss = 0
    correct = 0
    total = 0

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_dataLoader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print("Epoch: {} - iter: {} | Loss: {:.3f} | Acc: {:.3f}%".format(
            epoch, batch_idx, train_loss / (batch_idx + 1), correct * 100. / total))

def test(epoch, model, test_dataLoader, device):
    total = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for data in test_dataLoader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += labels.eq(predicted).sum().item()

    print("Epoch: {}, Accuracy: {:.2f}%".format(epoch, correct * 100. / total))

# 微调20个epoch
for epoch in range(EPOCHS):
    train(epoch, net, train_dataLoader, criterion, optimizer, DEVICE)
    test(epoch, net, test_dataLoader, DEVICE)