import cv2 as cv
import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet
import numpy as np
class ResBlock(nn.Module):
    def __init__(self,channels):
        super(ResBlock, self).__init__()
        self.conv2dRelu = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(channels),
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(channels)
        )
        self.relu = nn.ReLU(channels)
    def forward(self,x):
        return self.relu(x + self.conv2dRelu(x))
class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet,self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=2,padding=0),
            nn.ReLU(64)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=2,padding=0),
            nn.ReLU(64)
        )
        self.tinypart1 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64)
        )
    def forward(self,x):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c8 = self.tinypart1(c2)
        return c8
if __name__ == "__main__":
    model = TinyNet()
    for module in model.modules():
        try:
            nn.init.constant_(module.weight, 0.05)
            nn.init.zeros_(module.bias)
            nn.init.zeros_(module.running_mean)
            nn.init.ones_(module.running_var)
        except Exception as e:
            pass
        if type(module) is nn.BatchNorm2d:
            module.eval()
    x = torch.ones(1,3,640,640,requires_grad= True)
    pred = model(x)
    grad = torch.zeros_like(pred, requires_grad= True)
    grad[0, 0, 64, 64] = 1
    pred.backward(gradient = grad)
    grad_input = x.grad[0,0,...].data.numpy()
    grad_input = grad_input / np.max(grad_input)
    # 有效感受野 0.75 - 0.85
    #grad_input = np.where(grad_input>0.85,1,0)
    grad_input = np.where(grad_input>0.75,1,0)
    # 注释掉即为感受野
    grad_input = (grad_input * 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    grad_input = cv.dilate(grad_input, kernel=kernel)

    contours, _ = cv.findContours(grad_input, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    rect = cv.boundingRect(contours[0])
    print(rect[-2:])
    cv.imshow( "a",grad_input)
    cv.waitKey(0)