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
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
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
        np.delete(weight, prune_indices, axis=0)
    ).to(DEVICE)

    if conv_layer.bias is not None:
        new_conv.bias.data = torch.from_numpy(
            np.delete(conv_layer.bias.data.cpu().numpy(), prune_indices)
        ).to(DEVICE)

    return new_conv


def prune_resnet56(net, skip_layers, prune_ratios):
    layer_idx = 2
    for i, layer in enumerate([net.layer1, net.layer2, net.layer3]):
        for j in range(len(layer)):
            if layer_idx not in skip_layers:
                print(f"Pruning layer {layer_idx} with ratio {prune_ratios[i]}")
                layer[j].conv1 = prune_conv_layer(layer[j], prune_ratios[i])
            layer_idx += 2


skip_layers = [16, 18, 20, 34, 38, 54]
prune_ratios = [0.6, 0.3, 0.1]  # p1=60%, p2=30%, p3=10%


prune_resnet56(net, skip_layers, prune_ratios)
print("-" * 50)
print("Pruned ResNet56:")
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)


def train():
    net.train()
    for batch_idx, (inputs, targets) in enumerate(train_dataLoader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


# 微调20个epoch
# for epoch in range(EPOCHS):
#     train()
