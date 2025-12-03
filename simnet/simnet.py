import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class SimilarityNet(nn.Module):
    # 注意：input_dim 是 (H, W)，即 (32, 256)
    def __init__(self, input_dim=(32, 256), num_classes=1):
        super(SimilarityNet, self).__init__()
        m, k = input_dim  # input shape: (batch_size, m, k)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)    # -> [B, 128, 16, 128]
        self.layer3 = self._make_layer(128, 256, stride=2)   # -> [B, 256, 8, 64]
        self.layer4 = self._make_layer(256, 512, stride=2)   # -> [B, 512, 4, 32]
        self.layer5 = self._make_layer(512, 512, stride=1)   # -> [B, 512, 4, 32]
        
        # 自适应平均池化，将 [B, 512, 4, 32] 池化为 [B, 512, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(ResidualBlock2D(in_channels, out_channels, stride))
        # 您原来的代码中每层包含两个块，这里保持一致
        layers.append(ResidualBlock2D(out_channels, out_channels, stride=1)) 
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, 32, 256)
        x = x.unsqueeze(1)  # (batch_size, 1, 32, 256)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)  # (batch_size, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (batch_size, 512)
        
        # 使用 sigmoid 激活函数将输出映射到 0-1 范围
        x = torch.sigmoid(self.fc(x))  # (batch_size, 1)
        
        # 如果希望输出是 [BATCH_SIZE] 而不是 [BATCH_SIZE, 1]，则使用 squeeze
        return x.squeeze(1)  # (batch_size,)