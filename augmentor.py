'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-09-17 00:45:53
@LastEditors: Aoru Xue
@LastEditTime: 2019-09-28 17:08:10
'''
import cv2 as cv
import numpy as np
from PIL import Image
from albumentations import (
    
    HorizontalFlip,
    Resize,
    Compose,
    RandomSunFlare,
    RandomShadow,
    RandomBrightness,
    RandomContrast,
    RandomCrop,
    GaussianBlur,
    ToTensor
)

class BasketAug(object):
    def __init__(self,):
        self.transform = Compose([
            RandomCrop(height = 512,width = 512,p=0.5),
            RandomBrightness(p=0.5),
            RandomContrast(p=0.5),
            
            RandomSunFlare(p=0.5, flare_roi=(0, 0, 1, 0.5), angle_lower=0.5,src_radius= 150),
            RandomShadow(p=0.5, num_shadows_lower=1, num_shadows_upper=1, 
                        shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1)),
            HorizontalFlip(p=0.5), 
            
            
            GaussianBlur(p=0.5),
            ToTensor(normalize = {"mean":[0.485, 0.456, 0.406],"std":[0.229, 0.224, 0.225]}),
        ],
        bbox_params={"format" : "albumentations","min_area": 0,"min_visibility": 0,'label_fields': ['category_id']}
        )
        
    def __call__(self,cv_img, boxes=None, labels=None):
        auged = self.transform(image = cv_img,bboxes = boxes, category_id = labels)
        
        return auged["image"],auged["bboxes"],auged["category_id"]
if __name__ == "__main__":
    basket_aug = BasketAug()
    img = cv.imread("./a.jpg")
    boxes = [[0.2,0.2,0.5,0.5],[0,0,0.1,0.1]]
    labels = [0,1]
    img,boxes,labels = basket_aug(img,boxes,labels)
    
    for box in boxes:
        xmin,ymin,xmax,ymax = box
        xmin*=512
        ymin*=512
        xmax*=512
        ymax*=512
        cv.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255,0,0),2)
    cv.imshow("img",img)
    cv.waitKey(0)

