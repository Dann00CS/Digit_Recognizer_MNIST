import torch.nn as nn
import torch
import torch.nn.functional as F

class MNIST_model(nn.Module):
    def __init__(self):
        super(MNIST_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding="same"),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding="same"),
            nn.ReLU(),
            nn.Dropout2d(p=0.1)
            # nn.MaxPool2d(kernel_size=2)
        )
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Sequential(
            nn.Linear(32, 10),
            # nn.ReLU(),
            # nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.GAP(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
    
    # def __init__(self):
    #     super(MNIST_model, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    #     self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    #     self.conv2_drop = nn.Dropout2d()
    #     self.fc1 = nn.Linear(320, 50)
    #     self.fc2 = nn.Linear(50, 10)
    #     self.max_pool2d = nn.MaxPool2d(2)

    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #     x = x.view(-1, 320)
    #     x = F.relu(self.fc1(x))
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)
    #     return x #F.softmax(x, dim=1)
    
