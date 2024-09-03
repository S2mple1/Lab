import torch

from Pruned_ResNet56 import ResNet56, test_dataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = ResNet56().to(DEVICE)

model.load_state_dict(torch.load("fine_tuned_pruned_model.pth"))


total = 0
correct = 0

model.eval()
with torch.no_grad():
    for data in test_dataLoader:
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += labels.eq(predicted).sum().item()

print("Test Accuracy: {:.2f}%".format(correct * 100. / total))