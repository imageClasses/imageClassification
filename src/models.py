import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class MultiLabelResnet(nn.Module):
    "This model uses Resnet base model with best current weights"
    
    def __init__(self, num_classes=14):
        super(MultiLabelResnet, self).__init__()
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.base_model(x)
        out = self.classifier(features)
        return out
    
class MultiLabelCNN(nn.Module):
    "Simple home-made CNN for comparing to resnet"
    
    def __init__(self, num_classes=14, dropout_rate=0.1):
        super(MultiLabelCNN, self).__init__()
        self.dropout = nn.Dropout(0.05)
        self.features = nn.Sequential(
          nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.BatchNorm2d(16),
          nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.BatchNorm2d(32)
        )

        self.classify = nn.Sequential(
          nn.Linear(32 * 32 * 32, 60),
          nn.ReLU(),
          nn.Dropout(p=dropout_rate),
          nn.Linear(60, num_classes),
          nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32 * 32 * 32)
        x = self.classify(x)
        return x