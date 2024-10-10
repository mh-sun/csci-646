import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, dropout=0.1):
        super(CNN, self).__init__()
        
        # Convolution Layers
        self.conv_layers = nn.Sequential(
            # Convolution Layer 1
            nn.Conv2d(3, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2, 2),
            
            # Convolution Layer 2
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2, 2),
            
            # Convolution Layer 3
            nn.Conv2d(128, 256, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2, 2),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            # Layer 1
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
