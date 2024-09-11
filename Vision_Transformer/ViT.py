import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Shape x: [batch_size, channels, height, width]
        x = self.proj(x).flatten(2).transpose(1, 2)
        # [batch_size, emb_size, n_patches ** 0.5, n_patches ** 0.5]
        # Flatten into patches: [batch_size, emb_size, n_patches]
        # [batch_size, n_patches, emb_size]
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, ff_hidden_mult=4, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, ff_hidden_mult * emb_size),
            nn.GELU(),
            nn.Linear(ff_hidden_mult * emb_size, emb_size)
        )
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, emb_size=768, num_layers=12, num_heads=12,
                 num_classes=100, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, emb_size))

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        # Get patches and embed them
        x = self.patch_embed(x)  # Shape: [batch_size, n_patches, emb_size]

        # Add class token
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, emb_size]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, n_patches+1, emb_size]

        # Add position embedding
        x = x + self.pos_embedding

        # Pass through Transformer encoder layers
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        # Take class token for classification
        cls_output = x[:, 0]  # [batch_size, emb_size]
        x = self.mlp_head(cls_output)  # [batch_size, num_classes]

        return x


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR-100 均值和标准差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
EPOCHS = 10

train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = VisionTransformer()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Step: {batch_idx}, Loss: {running_loss/(batch_idx+1):.3f}, Acc: {100.*correct/total:.3f}%')

# 测试过程
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(EPOCHS), targets.to(EPOCHS)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f'Epoch: {epoch}, Test Loss: {test_loss/(batch_idx+1):.3f}, Test Acc: {100.*correct/total:.3f}%')


# 开始训练和测试
for epoch in range(1, EPOCHS + 1):
    train(epoch)
    test(epoch)