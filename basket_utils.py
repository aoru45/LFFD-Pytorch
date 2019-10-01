'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-09-02 01:32:25
@LastEditors: Aoru Xue
@LastEditTime: 2019-10-01 10:36:00
'''
import torch
import math


def convert_locations_to_boxes(locations, center_form_priors, variance = 2):

    # priors can have one dimension less.
    if center_form_priors.dim() + 1 == locations.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        center_form_priors[..., :2] - locations[..., :2] * variance * center_form_priors[..., 2:],
        center_form_priors[..., :2] - locations[..., 2:] * variance * center_form_priors[..., 2:]
    ], dim=locations.dim() - 1)


def convert_boxes_to_locations(corner_form_boxes, center_form_priors,variance = 2):
    
    if center_form_priors.dim() + 1 == corner_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_priors[..., :2] - corner_form_boxes[..., :2]) / center_form_priors[..., 2:] / variance,
        (center_form_priors[..., :2] - corner_form_boxes[..., 2:]) / center_form_priors[..., 2:] / variance
    ], dim=corner_form_boxes.dim() - 1)


def area_of(left_top, right_bottom):
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2]) # 每个predbox与gt左上角坐标比一下，取最大的，因为是两个坐标之间的比较
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def in_center(priors,gt):
    # 1,n,4    m,1,4
    s1 = (priors[...,0] > gt[...,0]) & (priors[...,1] > gt[...,1]) & (priors[...,0] < gt[...,2]) & (priors[...,1] < gt[...,3]) 
    return s1
def assign_priors(gt_boxes, gt_labels, center_form_priors,
                  iou_threshold):
    
    #print(gt_boxes,corner_form_priors)
    # size: num_priors x num_targets
    s1 = in_center(center_form_priors.unsqueeze(1),gt_boxes.unsqueeze(0)) # 直接匹配在中间的

    s1[torch.sum(s1,dim = 1) > 1] = False # 同时有多个匹配的
    best_target_per_prior, best_target_per_prior_index = s1.max(1)
    
    
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior == False] = 0  # the backgournd id
    
    boxes = gt_boxes[best_target_per_prior_index]
    t = 0 
    center_form_gt_boxes = corner_form_to_center_form(boxes)
    for f,scale in zip([127,127,63,63,31,15,15,15],[(10,15),(15,20),(20,40),(40,70),(70,110),(110,250),(250,400),(400,560)]):
        # s2 = (center_form_gt_boxes[t:t+f*f,2] > scale[0]/512. * 0.9) & \
        # (center_form_gt_boxes[t:t+f*f,2] < scale[0]/512.) & \
        # (center_form_gt_boxes[t:t+f*f,2] > scale[1]/512.) & \
        # (center_form_gt_boxes[t:t+f*f,2] < scale[1]/512. * 1.1)
        
        # s3 = (center_form_gt_boxes[t:t+f*f,3] > scale[0]/512. * 0.9) & \
        # (center_form_gt_boxes[t:t+f*f,3] < scale[0]/512.) & \
        # (center_form_gt_boxes[t:t+f*f,3] > scale[1]/512.) & \
        # (center_form_gt_boxes[t:t+f*f,3] < scale[1]/512. * 1.1)
        s2 = (center_form_gt_boxes[t:t+f*f, 2] < scale[0]/512.) | \
        (center_form_gt_boxes[t:t+f*f, 2] > scale[1]/512.) 

        s3 = (center_form_gt_boxes[t:t+f*f, 3] < scale[0]/512.) | \
           (center_form_gt_boxes[t:t+f*f, 3] > scale[1]/512.)
        labels[t:t+f*f][s2 & s3] = 0 
        # labels[t:t+f*f][s3==True] = 0
        #labels[t:t+f*f][s4==True] = 0 
        t += f*f
    
    
    # # size: num_priors
    # best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # # size: num_targets
    # best_prior_per_target, best_prior_per_target_index = ious.max(0)

    # for target_index, prior_index in enumerate(best_prior_per_target_index):
    #     best_target_per_prior_index[prior_index] = target_index
    # best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # # size: num_priors
    # labels = gt_labels[best_target_per_prior_index]
    # labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    # boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels


def hard_negative_mining(loss, labels, neg_pos_ratio):
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:] / 2,
                      locations[..., :2] + locations[..., 2:] / 2], locations.dim() - 1)


def corner_form_to_center_form(boxes):
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)
