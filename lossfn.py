'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-09-10 00:03:42
@LastEditors: Aoru Xue
@LastEditTime: 2019-09-28 16:34:11
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import basket_utils


class BasketLoss(nn.Module):
    def __init__(self):
        super(BasketLoss, self).__init__()
        self.neg_pos_ratio = 10
    def forward(self,scores, predicted_locations, labels, gt_locations):
        num_classes = scores.size(2)
        with torch.no_grad():
            loss = -F.log_softmax(scores, dim=2)[:, :, 0]
            mask = basket_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = scores[mask, :]
        classification_loss = F.cross_entropy(confidence.view(-1, num_classes), labels[mask], reduction='sum')
        print(classification_loss.numpy().sum(), " cls loos")

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        mse_loss = F.mse_loss(predicted_locations, gt_locations, reduction='sum')
        print(mse_loss.numpy().sum(), " reg loss")
        #smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return mse_loss / num_pos, classification_loss / num_pos


class LFFDLoss(nn.Module):

    def __init__(self, hnm_ratio):

        super(LFFDLoss, self).__init__()
        self.hnm_ratio = int(hnm_ratio)

    def forward(self, pred_score, preo_bbox, gt_label, gt_mask):
        pred_score_softmax = torch.softmax(pred_score, dim=1)
        loss_mask = torch.ones(pred_score_softmax.shape)
        #hnm is only performed on the classification loss
        if self.hnm_ratio > 0:
            pos_flag = (gt_label[:, 0, :, :] > 0.5)
            pos_num = torch.sum(pos_flag)

            if pos_num > 0:
                neg_flag = (gt_label[:, 1, :, :] > 0.5)
                neg_num = torch.sum(neg_flag)
                neg_num_selected = min(int(self.hnm_ratio * pos_num), int(neg_num))
                neg_prob = torch.where(neg_flag, pred_score_softmax[:, 1, :, :], torch.zeros_like(pred_score_softmax[:, 1, :, :]))
                neg_prob_sort, _ = torch.sort(neg_prob.reshape(1, -1), descending=False)

                prob_threshold = neg_prob_sort[0][neg_num_selected-1]
                neg_grad_flag = (neg_prob <= prob_threshold)
                loss_mask = torch.cat([pos_flag.unsqueeze(1), neg_grad_flag.unsqueeze(1)],dim=1)
            else:
                neg_choice_ratio = 0.1
                neg_num_selected = int(pred_score_softmax[:, 1, :, :].numel() * neg_choice_ratio)
                neg_prob = pred_score_softmax[:, 1, :, :]
                neg_prob_sort, _ = torch.sort(neg_prob.reshape(1, -1), descending=False)
                prob_threshold = neg_prob_sort[0][neg_num_selected-1]
                neg_grad_flag = (neg_prob <= prob_threshold)
                loss_mask = torch.cat([pos_flag.unsqueeze(1), neg_grad_flag.unsqueeze(1)],dim=1)
        pred_score_softmax_masked = pred_score_softmax[loss_mask]
        pred_score_log = torch.log(pred_score_softmax_masked)
        score_cross_entropy = -gt_label[:, :2, :, :][loss_mask] * pred_score_log
        loss_score = torch.sum(score_cross_entropy) / score_cross_entropy.numel()

        mask_bbox = gt_mask[:, 2:6, :, :]
        if torch.sum(mask_bbox) == 0:
            loss_bbox = torch.zeros_like(loss_score)
        else:
            predict_bbox = preo_bbox * mask_bbox
            label_bbox = gt_label[:, 2:6, :, :] * mask_bbox
            loss_bbox = F.mse_loss(predict_bbox, label_bbox, reduction='sum') / torch.sum(mask_bbox)
        return loss_score, loss_bbox


if __name__ == "__main__":
    loss_fn = LFFDLoss(5)
    scores = torch.randn(4, 2, 159, 159)
    locations = torch.randn(4, 4, 159, 159)

    gt_label = torch.randn(4, 6, 159, 159)
    gt_mask = torch.randn(4,6,159, 159)
    loss_score, loss_bbox = loss_fn(scores, locations, gt_label, gt_mask)
    print(loss_score, loss_bbox)
