'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-09-09 23:13:46
@LastEditors: Aoru Xue
@LastEditTime: 2019-09-28 16:24:56
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from priors import Priors
from basket_utils import *
class ResBlock(nn.Module):
    def __init__(self,channels):
        super(ResBlock, self).__init__()
        self.conv2dRelu = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU6(channels),
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU6(channels)
        )
        self.relu = nn.ReLU6(channels)
    def forward(self,x):
        return self.relu(x + self.conv2dRelu(x))
class BasketLossBranch(nn.Module):
    def __init__(self,in_channels,out_channels=64,num_classes=3):
        super(BasketLossBranch, self).__init__()
        self.conv1x1relu = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0),
            nn.ReLU6(out_channels)
        )
        self.score =nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=0),
            nn.ReLU6(out_channels),
            nn.Conv2d(out_channels,num_classes,kernel_size=1,stride=1)
        )
        self.locations =nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=0),
            nn.ReLU6(out_channels),
            nn.Conv2d(out_channels,4,kernel_size=1,stride=1)
        )
    def forward(self,x):
        score = self.score(self.conv1x1relu(x))
        locations = self.locations(self.conv1x1relu(x))
        return score,locations
class BasketNet(nn.Module):
    def __init__(self,num_classes = 3):
        super(BasketNet, self).__init__()
        self.num_classes = num_classes 
        self.priors = None
        self.c1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=2,padding=0),
            nn.ReLU6(64)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=2,padding=0),
            nn.ReLU6(64)
        )
        self.tinypart1 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64)
        )
        self.tinypart2 = ResBlock(64)
        self.c11 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=2,padding=0),
            nn.ReLU6(64)
        )
        self.smallpart1 = ResBlock(64)
        self.smallpart2 = ResBlock(64)
        self.c16 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=0),
            nn.ReLU6(128)
        )
        self.mediumpart = ResBlock(128)
        self.c19 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),
            nn.ReLU6(128)
        )
        self.largepart1 = ResBlock(128)
        self.largepart2 = ResBlock(128)
        self.largepart3 = ResBlock(128)


        self.lossbranch1 = BasketLossBranch(64,num_classes = self.num_classes)
        self.lossbranch2 = BasketLossBranch(64,num_classes = self.num_classes)
        self.lossbranch3 = BasketLossBranch(64,num_classes = self.num_classes)
        self.lossbranch4 = BasketLossBranch(64,num_classes = self.num_classes)
        self.lossbranch5 = BasketLossBranch(128,num_classes = self.num_classes)
        self.lossbranch6 = BasketLossBranch(128,num_classes = self.num_classes)
        self.lossbranch7 = BasketLossBranch(128,num_classes = self.num_classes)
        self.lossbranch8 = BasketLossBranch(128,num_classes = self.num_classes)
    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)

        c8 = self.tinypart1(c2)
        c10 = self.tinypart2(c8)

        c11 = self.c11(c10)
        c13 = self.smallpart1(c11)
        c15 = self.smallpart2(c13)

        c16 = self.c16(c15)
        c18 = self.mediumpart(c16)

        c19 = self.c19(c18)
        c21 = self.largepart1(c19)
        c23 = self.largepart2(c21)
        c25 = self.largepart3(c23)

        score1,loc1 = self.lossbranch1(c8)
        score2,loc2 = self.lossbranch2(c10)
        score3,loc3 = self.lossbranch3(c13)
        score4,loc4 = self.lossbranch4(c15)
        score5,loc5 = self.lossbranch5(c18)
        score6,loc6 = self.lossbranch6(c21)
        score7,loc7 = self.lossbranch7(c23)
        score8,loc8 = self.lossbranch8(c25)

        if not self.training:
            pred_score1 = torch.softmax(score1, dim=1)[:, 0, :, :]
            pred_bbox1 = loc1

            pred_score2 = torch.softmax(score2, dim=1)[:, 0, :, :]
            pred_bbox2 = loc2

            pred_score3 = torch.softmax(score3, dim=1)[:, 0, :, :]
            pred_bbox3 = loc3

            pred_score4 = torch.softmax(score4, dim=1)[:, 0, :, :]
            pred_bbox4 = loc4

            pred_score5 = torch.softmax(score5, dim=1)[:, 0, :, :]
            pred_bbox5 = loc5

            pred_score6 = torch.softmax(score6, dim=1)[:, 0, :, :]
            pred_bbox6 = loc6

            pred_score7 = torch.softmax(score7, dim=1)[:, 0, :, :]
            pred_bbox7 = loc7

            pred_score8 = torch.softmax(score8, dim=1)[:, 0, :, :]
            pred_bbox8 = loc8
            return pred_score1, pred_bbox1, pred_score2, pred_bbox2, pred_score3, pred_bbox3, pred_score4, pred_bbox4, pred_score5, pred_bbox5, pred_score6, pred_bbox6, pred_score7, pred_bbox7, pred_score8, pred_bbox8

        else:
            return score1, loc1, score2, loc2, score3, loc3, score4, loc4, score5, loc5, score6, loc6, score7, loc7, score8, loc8



# from torchsummary import summary
if __name__ == '__main__':
    model = BasketNet(num_classes=2).cuda()
    input = torch.randn([4, 3, 640, 640]).cuda()
    score1, loc1, score2, loc2, score3, loc3, score4, loc4, score5, loc5, score6, loc6, score7, loc7, score8, loc8 = model(input)
    print(score1.shape, loc8.shape)

