'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-09-10 12:51:20
@LastEditors: Aoru Xue
@LastEditTime: 2019-09-13 20:59:57
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np  
from priors import Priors
from basket_utils import *
np.set_printoptions(threshold=np.inf)  
class BasketDataset(Dataset):
    def __init__(self,img_path,transform = None,center_variance = 0.1,size_variance = 0.2):
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.img_paths = glob.glob(img_path + "/images/*.jpg")
        self.labels = [label.replace(".jpg",".xml").replace("images","labels") for label in self.img_paths]
        self.class_names = ("__background__","basketball","volleyball")
        prior = Priors() 
        self.center_form_priors = prior() # center form
        self.imgW,self.imgH = 512,512
        self.corner_form_priors = center_form_to_corner_form(self.center_form_priors)
        self.transform = transform  
    def __getitem__(self,idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label_file = self.labels[idx]
        gt_bboxes,gt_classes = self._get_annotation(idx)
        
        if self.transform:
            img = self.transform(img)
      

        gt_bboxes,gt_classes = assign_priors(gt_bboxes,gt_classes,self.corner_form_priors,0.5) # corner form
        #imH,imW = cv_img.shape[:2]
        
        gt_bboxes = corner_form_to_center_form(gt_bboxes) # (1524, 4) center form
        locations = convert_boxes_to_locations(gt_bboxes, self.center_form_priors, self.center_variance, self.size_variance) # 相当于归一化
        # 拟合距离而不是直接拟合，这样更容易拟合。
        
        return [img,locations,gt_classes]
    def _get_annotation(self,idx):
        annotation_file = self.labels[idx]
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        #is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            
            boxes.append([x1/self.imgW,y1/self.imgH,x2/self.imgW,y2/self.imgH])
            labels.append(self.class_names.index(class_name))
        return (torch.tensor(boxes, dtype=torch.float),
                torch.tensor(labels, dtype=torch.long))
    def __len__(self):
        return len(self.img_paths)
if __name__ == '__main__':
    datset = BasketDataset("./datasets")
    import cv2 as cv
    img,gt_loc,gt_labels = datset[0]
    cv_img = np.array(img)
    cv_img = cv.cvtColor(cv_img,cv.COLOR_RGB2BGR)
    idx = gt_labels > 0
    #print(gt_loc.size(),dataset.priors.size())
    loc = convert_locations_to_boxes(gt_loc,datset.center_form_priors,0.1,0.2)
    loc = loc[idx]
    label = gt_labels[idx]
    for i in range(loc.size(0)):
        print(loc.size())
        x1,y1,w,h = loc[i,:]
        #print(x,y,r)
        x1 = x1.item() * 512.
        y1 = y1.item() * 512.
        w= w.item() * 512.
        h = h.item() * 512.        
        #cv.circle(cv_img,(int(x),int(y)),int(r),(255,0,0),2)
        cv.rectangle(cv_img,(int(x1 - w/2),int(y1-h/2)),(int(x1 + w/2),int(y1 + h/2)),(255,0,0),2)
    cv.imshow("cv",cv_img)
    cv.waitKey(0)