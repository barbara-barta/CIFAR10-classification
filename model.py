import torch
import torch.nn as nn
import torch.nn.functional as F

class SixLayerPoolCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,32,3,1,1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,32,3,1,1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32,64,3,1,1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,64,3,1,1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64,128,3,1,1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128,128,3,1,1)
        self.bn6 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(128*8*8,10)

    def forward(self,x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.bn2(self.conv2(x1))
        x = F.relu(x1 + x2)
        x = self.pool(x)

        x3 = F.relu(self.bn3(self.conv3(x)))
        x4 = self.bn4(self.conv4(x3))
        x = F.relu(x3 + x4)
        x = self.pool(x)

        x5 = F.relu(self.bn5(self.conv5(x)))
        x6 = self.bn6(self.conv6(x5))
        x = F.relu(x5 + x6)

        x = x.view(x.size(0), -1)
        return self.fc(x)