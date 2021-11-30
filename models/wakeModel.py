import torch.nn as nn

class wakeModel(nn.Module):
    def __init__(
        self,
        num_classes=2,
        num_channels=3,
        dimensions=(320,256),
        bias=False,
        **kwargs
    ):
        super().__init__()
        
        self.conv1_1 = nn.Conv2d(num_channels, 16, 3, stride=1, padding=1, bias=bias)
        self.relu = nn.ReLU();
        self.conv1_2 = nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=bias)
        self.conv1_3 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=bias)
        self.maxpool = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=bias)
        self.conv2_2 = nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=bias)
        self.conv2_3 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=bias)

        self.lastPool = nn.MaxPool2d(4)

        self.fc1 = nn.Linear(10*8*64, 40, bias=bias)
        self.fc2 = nn.Linear(40, num_classes, bias=bias)

        self.dropout = nn.Dropout(.25)
        
    def forward(self,x):
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv1_3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.conv2_3(x)
        x = self.relu(x)
        x = self.lastPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def wakemodel(pretrained=False, **kwargs):
    assert not pretrained
    return wakeModel(**kwargs)

models = [
    {
        'name': 'wakeModel',
        'min_input': 1,
        'dim': 2,
    },
]

class wakeModel2(nn.Module):
    def __init__(
        self,
        num_classes=2,
        num_channels=3,
        dimensions=(320,256),
        bias=False,
        **kwargs
    ):
        super().__init__()
        
        self.conv1_1 = nn.Conv2d(num_channels, 16, 3, stride=1, padding=1, bias=bias)
        self.relu = nn.ReLU();
        self.conv1_2 = nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=bias)
        self.conv1_3 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=bias)
        self.maxpool = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=bias)
        self.conv2_2 = nn.Conv2d(128, 192, 3, stride=1, padding=1, bias=bias)
        self.conv2_3 = nn.Conv2d(192, 64, 3, stride=1, padding=1, bias=bias)

        self.fc1 = nn.Linear(5*4*64, 48, bias=bias)
        self.fc2 = nn.Linear(48, num_classes, bias=bias)

        self.dropout = nn.Dropout(.25)
        
    def forward(self,x):
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv1_3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2_3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


