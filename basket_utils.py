'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-09-02 01:32:25
@LastEditors: Aoru Xue
@LastEditTime: 2019-10-02 18:37:41
'''
import torch
import math
from config import *

def convert_locations_to_boxes(locations, priors, variance = 2):

    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        priors[..., :2] - locations[..., :2]  * priors[..., 2:],
        priors[..., :2] - locations[..., 2:]  * priors[..., 2:]
    ], dim=locations.dim() - 1)


def convert_boxes_to_locations(corner_form_boxes, priors,variance = 2):
    
    if priors.dim() + 1 == corner_form_boxes.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        (priors[..., :2] - corner_form_boxes[..., :2]) / priors[..., 2:],
        (priors[..., :2] - corner_form_boxes[..., 2:]) / priors[..., 2:]
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
'''
def assign_priors(gt_boxes, gt_labels, priors,
                  iou_threshold):
    
    #print(gt_boxes,corner_form_priors)
    # size: num_priors x num_targets
    # prior 包含 gt
    s1 = in_center(priors.unsqueeze(1),gt_boxes.unsqueeze(0)) # 直接匹配在中间的

    #s1[torch.sum(s1,dim = 1) > 1] = False # 同时有多个匹配的
    not_ignored = torch.ones(s1.size(0),dtype = torch.uint8)
    not_ignored[torch.sum(s1,dim = 1) > 1] = 0
    best_target_per_prior, best_target_per_prior_index = s1.max(1)
    
    
    labels = gt_labels[best_target_per_prior_index] # (num_priors,1)
    labels[best_target_per_prior == False] = 0  # the backgournd id,没有匹配的给背景
    
    boxes = gt_boxes[best_target_per_prior_index] #num_priors，４
    t = 0 
    center_form_gt_boxes = corner_form_to_center_form(boxes)
    for f,scale in zip(feature_maps,scales):
        d = torch.min(center_form_gt_boxes[t:t+f*f, 2],center_form_gt_boxes[t:t+f*f, 3])
        condition = (d < scale[0]/image_size) | (d > scale[1]/image_size) 
        labels[t:t+f*f][condition] = 0 
        
        left_gray_scale = [0.9 *scale[0], scale[0]]
        right_gray_scale = [scale[1], 1.1 * scale[1]]
        
        not_ignored[t:t+f*f][(d > left_gray_scale[0]) & (d < left_gray_scale[1])] = 0
        not_ignored[t:t+f*f][(d > right_gray_scale[0]) & (d < right_gray_scale[1])] = 0

        t += f*f
    return boxes, labels, not_ignored
'''
def assign_priors(gt_boxes, gt_labels, priors,
                  iou_threshold):
    
    #print(gt_boxes,corner_form_priors)
    # size: num_priors x num_targets
    # prior 包含 gt
    s1 = in_center(priors.unsqueeze(1),gt_boxes.unsqueeze(0)) # 直接匹配在中间的
    corner_form_priors = center_form_to_corner_form(priors)
    ious = iou_of(corner_form_priors.unsqueeze(1),gt_boxes.unsqueeze(0))
    #s1[torch.sum(s1,dim = 1) > 1] = False # 同时有多个匹配的
    not_ignored = torch.ones(s1.size(0),dtype = torch.uint8)
    not_ignored[torch.sum(s1,dim = 1) > 1] = 0
    best_target_per_prior, best_target_per_prior_index = (ious*s1).max(1)
    best_prior_per_target, best_prior_per_target_index = ious.max(0)
    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    labels = gt_labels[best_target_per_prior_index] # (num_priors,1)
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id,没有匹配的给背景
    
    boxes = gt_boxes[best_target_per_prior_index] #num_priors，４
    t = 0 
    center_form_gt_boxes = corner_form_to_center_form(boxes)
    for f,scale in zip(feature_maps,scales):
        d = torch.min(center_form_gt_boxes[t:t+f*f, 2],center_form_gt_boxes[t:t+f*f, 3])
        #condition = (d < scale[0]/image_size) | (d > scale[1]/image_size) 
        #labels[t:t+f*f][condition] = 0 
        
        left_gray_scale = [0.9 *scale[0], scale[0]]
        right_gray_scale = [scale[1], 1.1 * scale[1]]
        
        not_ignored[t:t+f*f][(d > left_gray_scale[0]) & (d < left_gray_scale[1])] = 0
        not_ignored[t:t+f*f][(d > right_gray_scale[0]) & (d < right_gray_scale[1])] = 0

        t += f*f
    return boxes, labels, not_ignored

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
