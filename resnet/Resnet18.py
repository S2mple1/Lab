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


class ResNet(nn.Module):
    def __init__(self, ResNetBlock, num_classes=10):
        super(ResNet, self).__init__()
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
        out = self.ResBlock2(out)
        out = self.ResBlock3(out)
        out = self.ResBlock4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResNetBlock)


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

# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = ResNet18().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

best_acc = 85.0

if __name__ == '__main__':
    with open("train.txt", "w") as f:
        for epoch in range(1, EPOCHS + 1):
            print(epoch)
            net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, data in enumerate(train_dataLoader, 0):
                length = len(train_dataLoader)
                inputs, targets = data
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, dim=1)
                total += targets.size(0)
                correct += torch.eq(predicted, targets).cpu().sum().item()
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch, (i + 1 + (epoch - 1) * length), sum_loss / (i + 1), 100. * correct / total))
                f.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                        % (epoch, (i + 1 + (epoch - 1) * length), sum_loss / (i + 1), 100. * correct / total))
                f.write('\n')
                f.flush()

        with torch.no_grad():
            correct = 0.0
            total = 0.0

            for data in test_dataLoader:
                net.eval()
                inputs, targets = data
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, dim=1)
                total += inputs.size(0)
                correct += torch.eq(predicted, targets).sum().item()

            acc = 100. * correct / total
            print("Test Accuracy:{:.3f}".format(acc))
            if acc > best_acc:
                torch.save(net.state_dict(), "{}_{:.3f}.pth".format(epoch, acc))
                best_acc = acc
