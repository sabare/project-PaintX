import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.bn1=nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.bn2=nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.bn3=nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        #self.drp1=nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        #self.drp2=nn.Dropout(0.1)
        self.fc5 = nn.Linear(64, 7)

    def forward(self, x):
        x = (self.bn1(F.leaky_relu(self.conv1(x))))
        x = self.pool(self.bn2(F.leaky_relu(self.conv2(x))))
        x = self.pool(self.bn3(F.leaky_relu(self.conv3(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.leaky_relu((self.fc1(x)))
        x = F.leaky_relu((self.fc2(x)))
        x = self.fc5(x)
        return x


model = Net()
model.load_state_dict(torch.load("Header/model_val.pth"))
model.eval()
#model = torch.load("Header/draw_model.pth")
#model.eval()

def pred(image):
    predi = model(image.float())
    print(predi)
    result = torch.argmax(predi[0])

    return result
