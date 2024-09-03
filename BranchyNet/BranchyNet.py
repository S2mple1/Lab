import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


class BranchyNet(nn.Module):
    def __init__(self, num_classes=10):
        super(BranchyNet ,self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        self.exit1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.exit2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.exit3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        out1 = self.exit1(x)

        x = self.block2(x)
        out2 = self.exit2(x)

        x = self.block3(x)
        out3 = self.exit3(x)

        return out1, out2, out3


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 10

model = BranchyNet().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
sum_loss1 = 0.0
sum_loss2 = 0.0
sum_loss3 = 0.0
total = 0

def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()

    global sum_loss1
    global sum_loss2
    global sum_loss3
    global total

    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs1, outputs2, outputs3 = model(inputs)

        loss1 = criterion(outputs1, labels)
        loss2 = criterion(outputs2, labels)
        loss3 = criterion(outputs3, labels)

        loss = loss1 + loss2 + loss3
        loss.backward()
        optimizer.step()

        sum_loss1 += loss1.item()
        sum_loss2 += loss2.item()
        sum_loss3 += loss3.item()
        total += labels.size(0)
        print("Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\tLoss3: {:.6f}".format(
            epoch, batch_idx * len(inputs), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), sum_loss1 / total, sum_loss2 / total, sum_loss3 / total))

def test(model, test_loader, device):
    model.eval()

    total = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs1, outputs2, outputs3 = model(inputs)

            _, predicted1 = outputs1.max(1)
            _, predicted2 = outputs2.max(1)
            _, predicted3 = outputs3.max(1)

            total += labels.size(0)
            correct1 += labels.eq(predicted1).sum().item()
            correct2 += labels.eq(predicted2).sum().item()
            correct3 += labels.eq(predicted3).sum().item()

    print("Exit 1 Accuracy: {:.2f}%".format(correct1 * 100. / total))
    print("Exit 2 Accuracy: {:.2f}%".format(correct2 * 100. / total))
    print("Final Output Accuracy: {:.2f}%".format(correct3 * 100. / total))

if __name__ == '__main__':
    for epoch in range(1, EPOCHS + 1):
        print("Epoch: {}".format(epoch))
        train(epoch, model, train_loader, criterion, optimizer, DEVICE)
        test(model, test_loader, DEVICE)





