import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader


class ResNetBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResNetBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or ch_in != ch_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BranchyResNet(nn.Module):
    def __init__(self, ResNetBlock, num_classes=10):
        super(BranchyResNet, self).__init__()
        self.ch_in = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.ResBlock1 = self.make_layer(ResNetBlock, 64, 2, stride=1)
        self.ResBlock2 = self.make_layer(ResNetBlock, 128, 2, stride=2)
        self.ResBlock3 = self.make_layer(ResNetBlock, 256, 2, stride=2)
        self.ResBlock4 = self.make_layer(ResNetBlock, 512, 2, stride=2)

        self.exit1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

        self.exit2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

        self.exit3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.ch_in, channels, stride))
            self.ch_in = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.ResBlock1(out)
        out1 = self.exit1(out)

        out = self.ResBlock2(out)
        out2 = self.exit2(out)

        out = self.ResBlock3(out)
        out3 = self.exit3(out)

        out = self.ResBlock4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out1, out2, out3, out


def BranchyNet():
    return BranchyResNet(ResNetBlock)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 135
BATCH_SIZE = 128
LR = 0.1

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

train_dataLoader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_dataLoader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

net = BranchyNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)


def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    sum_loss1 = 0.0
    sum_loss2 = 0.0
    sum_loss3 = 0.0
    sum_loss4 = 0.0
    total = 0

    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs1, outputs2, outputs3, outputs4 = model(inputs)

        loss1 = criterion(outputs1, labels)
        loss2 = criterion(outputs2, labels)
        loss3 = criterion(outputs3, labels)
        loss4 = criterion(outputs4, labels)

        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        optimizer.step()

        sum_loss1 += loss1.item()
        sum_loss2 += loss2.item()
        sum_loss3 += loss3.item()
        sum_loss4 += loss4.item()
        total += labels.size(0)
        print("Epoch: {}, iter: {}, Combined Loss: {:.6f}, Loss1: {:.6f}, Loss2: {:.6f}, Loss3: {:.6f}, Loss4: {:.6f}"
              .format(epoch, batch_idx, (sum_loss1 + sum_loss2 + sum_loss3 + sum_loss4) / total,
                      sum_loss1 / total, sum_loss2 / total, sum_loss3 / total, sum_loss4 / total))

def test(model, test_loader, device):
    model.eval()
    total = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs1, outputs2, outputs3, outputs4 = model(inputs)

            _, predicted1 = outputs1.max(1)
            _, predicted2 = outputs2.max(1)
            _, predicted3 = outputs3.max(1)
            _, predicted4 = outputs4.max(1)

            total += labels.size(0)
            correct1 += labels.eq(predicted1).sum().item()
            correct2 += labels.eq(predicted2).sum().item()
            correct3 += labels.eq(predicted3).sum().item()
            correct4 += labels.eq(predicted4).sum().item()

    print("Exit 1 Accuracy: {:.2f}%".format(correct1 * 100. / total))
    print("Exit 2 Accuracy: {:.2f}%".format(correct2 * 100. / total))
    print("Exit 3 Accuracy: {:.2f}%".format(correct3 * 100. / total))
    print("Final Output Accuracy: {:.2f}%".format(correct4 * 100. / total))

if __name__ == '__main__':
    for epoch in range(1, EPOCHS + 1):
        print("----- Epoch {} -----".format(epoch))
        train(epoch, net, train_dataLoader, criterion, optimizer, DEVICE)
        test(net, test_dataLoader, DEVICE)