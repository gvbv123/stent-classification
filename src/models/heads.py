import torch.nn as nn

class ClassificationHead(nn.Module):
    
    def __init__(self, in_features, num_classes=2, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x
