'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-09-10 11:29:26
@LastEditors: Aoru Xue
@LastEditTime: 2019-09-13 16:44:14
'''
import torch
import torch.nn as nn
import numpy as np
def within_radius(prior_points,gt_positions):
    dxy = torch.abs(prior_points[...,:2] - gt_positions[...,:2])
    
    return (dxy[:,:,0] < gt_positions[:,:,2]) & (dxy[:,:,1] < gt_positions[:,:,2])

def assign_priors(priors,gt_positions,gt_labels = None):
    # priors (n,3) x,y,r
    # gt_positions (m,3) x,y,r
    # gt_labels (m)
    # labels_priors (n)
    # (n,m,2)
    dexy = within_radius(priors.unsqueeze(1),gt_positions.unsqueeze(0)) #(num_priors,num_targets)
    tf,idx = torch.max(dexy,dim = 1) # 最大的那个值要么是true要么是false，tf。
    # idx 表示跟第几个框子比较近
    labels = gt_labels[idx]
    labels[tf == False] = 0
    labels[torch.sum(dexy,dim = 1) > 1] = 0
    prs = gt_positions[idx]
    return prs,labels
if __name__ == "__main__":
    priors = torch.randn(20,3)
    gt_positions = torch.randn(10,3)
    gt_labels = torch.LongTensor(range(1,11,1))
    prs,labels = assign_priors(priors,gt_positions,gt_labels)
    print(prs,labels)