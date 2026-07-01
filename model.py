import torch
import torch.nn as nn
import torch.nn.functional as F

class GestureCNN(nn.Module):
    def __init__(self, num_classes = 10):
      super(GestureCNN, self).__init__()

      # Layer 1: 1 input -> 8 output channels
      self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 1, padding = 1)
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

      # Layer 2: 8 input -> 16 output channels
      self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)

      #FLATTENing: 16 feature maps * 14 * 14 pixels = 3136 features
      self.fc1 = nn.Linear(16 * 14 * 14, 64)
      self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 14 * 14)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
