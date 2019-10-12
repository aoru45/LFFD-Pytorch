'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-09-10 13:56:50
@LastEditors: Aoru Xue
@LastEditTime: 2019-10-01 10:37:04
'''
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import BasketNet
from torchvision import transforms
from dataset import *
from lossfn import BasketLoss, LFFDLoss
from tqdm import tqdm
# from augmentor import BasketAug


def train():
    pass


def org():
    # transform = BasketAug()
    # # transform = transforms.Compose([
    # #    transforms.Resize(512),
    # #    transforms.ToTensor(),
    # #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # # ])
    # dataset = BasketDataset("/media/xueaoru/Ubuntu/basketnet", transform=transform)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=6)
    #
    # net = BasketNet(num_classes=3)
    # net.cuda()
    # # net.load_state_dict(torch.load("./ckpt/502.pth"))
    # optimizer = optim.Adam(net.parameters(), lr=1e-4)
    # # optimizer = optim.SGD(net.parameters(), lr=1e-1,momentum=0.9,weight_decay=0.)
    # # scheduler = optim.lr_scheduler.StepLR(optimizer,100,0.1)
    # loss_fn = BasketLoss()
    # num_batches = len(dataloader)
    # min_loss = float("inf")
    # for epoch in range(10000):
    #     epoch_loss_cls = 0.
    #     epoch_loss_reg = 0.
    #
    #     for img, gt_pos, gt_labels in tqdm(dataloader):
    #         img = img.cuda()
    #         gt_pos = gt_pos.cuda()
    #         gt_labels = gt_labels.cuda()
    #         cls, loc = net(img)
    #         reg_loss, cls_loss = loss_fn(cls, loc, gt_labels, gt_pos)
    #         epoch_loss_cls += cls_loss.item()
    #         epoch_loss_reg += reg_loss.item()
    #         loss = reg_loss + cls_loss
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     # scheduler.step()
    #     print("cls_loss:{}---reg_loss:{}".format(epoch_loss_cls / num_batches, epoch_loss_reg / num_batches))
    #     if (epoch_loss_cls / num_batches + epoch_loss_reg / num_batches) < min_loss:
    #         min_loss = epoch_loss_cls / num_batches + epoch_loss_reg / num_batches
    #         torch.save(net.state_dict(), "./ckpt/{}.pth".format(
    #             int(epoch_loss_cls / num_batches * 1000 + epoch_loss_reg / num_batches * 1000)))
    pass

def lffd_train():
    # trigger for horizon flip
    param_enable_horizon_flip = True

    # trigger for vertical flip
    param_enable_vertical_flip = False

    # trigger for brightness
    param_enable_random_brightness = True
    param_brightness_factors = {'min_factor': 0.5, 'max_factor': 1.5}

    # trigger for saturation
    param_enable_random_saturation = True
    param_saturation_factors = {'min_factor': 0.5, 'max_factor': 1.5}

    # trigger for contrast
    param_enable_random_contrast = True
    param_contrast_factors = {'min_factor': 0.5, 'max_factor': 1.5}

    # trigger for blur
    param_enable_blur = False
    param_blur_factors = {'mode': 'random', 'sigma': 1}
    param_blur_kernel_size_list = [3]

    # negative image resize interval
    param_neg_image_resize_factor_interval = [0.5, 3.5]

    '''
        algorithm
    '''
    # the number of image channels
    param_num_image_channel = 3

    # the number of output scales (loss branches)
    param_num_output_scales = 8

    # feature map size for each scale
    param_feature_map_size_list = [159, 159, 79, 79, 39, 19, 19, 19]

    # bbox lower bound for each scale
    param_bbox_small_list = [10, 15, 20, 40, 70, 110, 250, 400]
    assert len(param_bbox_small_list) == param_num_output_scales

    # bbox upper bound for each scale
    param_bbox_large_list = [15, 20, 40, 70, 110, 250, 400, 560]
    assert len(param_bbox_large_list) == param_num_output_scales

    # bbox gray lower bound for each scale
    param_bbox_small_gray_list = [math.floor(v * 0.9) for v in param_bbox_small_list]
    # bbox gray upper bound for each scale
    param_bbox_large_gray_list = [math.ceil(v * 1.1) for v in param_bbox_large_list]

    # the RF size of each scale used for normalization, here we use param_bbox_large_list for better regression
    param_receptive_field_list = param_bbox_large_list
    # RF stride for each scale
    param_receptive_field_stride = [4, 4, 8, 8, 16, 32, 32, 32]
    # the start location of the first RF of each scale
    param_receptive_field_center_start = [3, 3, 7, 7, 15, 31, 31, 31]

    # the sum of the number of output channels, 2 channels for classification and 4 for bbox regression
    param_num_output_channels = 6

    # the ratio of neg image in a batch
    param_neg_image_ratio = 0.1

    # input height for network
    param_net_input_height = 640

    # input width for network
    param_net_input_width = 640

    db = LFFDDatasetPKL(pickle_file_path="./datasets/widerface_train_data_gt_8.pkl",
                           enable_horizon_flip=param_enable_horizon_flip,
                           enable_vertical_flip=param_enable_vertical_flip,
                           enable_random_brightness=param_enable_random_brightness,
                           brightness_params=param_brightness_factors,
                           enable_random_saturation=param_enable_random_saturation,
                           saturation_params=param_saturation_factors,
                           enable_random_contrast=param_enable_random_contrast,
                           contrast_params=param_contrast_factors,
                           enable_blur=param_enable_blur,
                           blur_params=param_blur_factors,
                           blur_kernel_size_list=param_blur_kernel_size_list,
                           num_image_channels=param_num_image_channel,
                           net_input_height=param_net_input_height,
                           net_input_width=param_net_input_width,
                           num_output_scales=param_num_output_scales,
                           receptive_field_list=param_receptive_field_list,
                           receptive_field_stride=param_receptive_field_stride,
                           feature_map_size_list=param_feature_map_size_list,
                           receptive_field_center_start=param_receptive_field_center_start,
                           bbox_small_list=param_bbox_small_list,
                           bbox_large_list=param_bbox_large_list,
                           bbox_small_gray_list=param_bbox_small_gray_list,
                           bbox_large_gray_list=param_bbox_large_gray_list,
                           num_output_channels=param_num_output_channels,
                           neg_image_resize_factor_interval=param_neg_image_resize_factor_interval)

    param_model_save_interval = 100000

    param_num_train_loops = 2000000

    batchsampler = LFFDBatchSampler(dataset=db,
                                    batch_size=16,
                                    num_neg_images_per_batch=4,
                                    max_iter=param_num_train_loops)

    dataloader = DataLoader(dataset=db,
                                   batch_sampler=batchsampler, timeout=30,
                                   collate_fn=lffd_collate, num_workers=4)

    net = BasketNet(num_classes=2)
    net.cuda()
    # init learning rate
    param_learning_rate = 0.01
    # weight decay
    param_weight_decay = 0.0001
    # momentum
    param_momentum = 0.9

    # learning rate scheduler -- MultiFactorScheduler
    scheduler_step_list = [500000, 1000000, 1500000]
    # multiply factor of scheduler
    scheduler_factor = 0.1

    lffd_optimer = optim.SGD(net.parameters(), lr=param_learning_rate, momentum=param_momentum, weight_decay=param_weight_decay)
    lffd_lr_scheduler = optim.lr_scheduler.MultiStepLR(lffd_optimer, scheduler_step_list, gamma=scheduler_factor)
    loss_fn = LFFDLoss(hnm_ratio=1)
    num_batches = len(dataloader)
    min_loss = 100000.0
    epoch_loss_cls = 0
    epoch_loss_reg = 0
    for imagbatch, branch1_label, branch2_label, branch3_label, branch4_label, branch5_label, branch6_label, branch7_label, branch8_label, branch1_mask, branch2_mask, branch3_mask, branch4_mask, branch5_mask, branch6_mask, branch7_mask, branch8_mask in tqdm(dataloader):
        # print(imagbatch[0])
        # print(imagbatch[1])
        # print("=======")
        # print(branch1_label[0])
        # print(branch1_label[1])
        # print("+++++++++")
        imagbatch = imagbatch.cuda()
        branch1_label = branch1_label.cuda()
        branch1_mask = branch1_mask.cuda()
        branch2_label = branch2_label.cuda()
        branch2_mask = branch2_mask.cuda()
        branch3_label = branch3_label.cuda()
        branch3_mask = branch3_mask.cuda()
        branch4_label = branch4_label.cuda()
        branch4_mask = branch4_mask.cuda()
        branch5_label = branch5_label.cuda()
        branch5_mask = branch5_mask.cuda()
        branch6_label = branch6_label.cuda()
        branch6_mask = branch6_mask.cuda()
        branch7_label = branch7_label.cuda()
        branch7_mask = branch7_mask.cuda()
        branch8_label = branch8_label.cuda()
        branch8_mask = branch8_mask.cuda()
        score1, loc1, score2, loc2, score3, loc3, score4, loc4, score5, loc5, score6, loc6, score7, loc7, score8, loc8 = net(imagbatch)
        reg_loss1, cls_loss1 = loss_fn(score1, loc1, branch1_label, branch1_mask)
        reg_loss2, cls_loss2 = loss_fn(score2, loc2, branch2_label, branch2_mask)
        reg_loss3, cls_loss3 = loss_fn(score3, loc3, branch3_label, branch3_mask)
        reg_loss4, cls_loss4 = loss_fn(score4, loc4, branch4_label, branch4_mask)
        reg_loss5, cls_loss5 = loss_fn(score5, loc5, branch5_label, branch5_mask)
        reg_loss6, cls_loss6 = loss_fn(score6, loc6, branch6_label, branch6_mask)
        reg_loss7, cls_loss7 = loss_fn(score7, loc7, branch7_label, branch7_mask)
        reg_loss8, cls_loss8 = loss_fn(score8, loc8, branch8_label, branch8_mask)
        epoch_loss_cls += (cls_loss1.item() + cls_loss2.item() + cls_loss3.item() + cls_loss4.item() + cls_loss5.item() + cls_loss6.item() + cls_loss7.item() + cls_loss8.item())
        epoch_loss_reg += (reg_loss1.item() + reg_loss2.item() + reg_loss3.item() + reg_loss4.item() + reg_loss5.item() + reg_loss6.item() + reg_loss7.item() + reg_loss8.item())
        loss = cls_loss1 + cls_loss2 + cls_loss3 + cls_loss4 + cls_loss5 + cls_loss6 + cls_loss7 + cls_loss8 + \
                   reg_loss1 + reg_loss2 + reg_loss3 + reg_loss4 + reg_loss5 + reg_loss6 + reg_loss7 + reg_loss8
        # loss = loss / 8
        print(loss)
        lffd_optimer.zero_grad()
        loss.backward()
        # if reg_loss1 > 0:
        #     reg_loss1.backward(retain_graph=True)
        # if reg_loss2 > 0:
        #     reg_loss2.backward(retain_graph=True)
        # if reg_loss3 > 0:
        #     reg_loss3.backward(retain_graph=True)
        # if reg_loss4 > 0:
        #     reg_loss4.backward(retain_graph=True)
        # if reg_loss5 > 0:
        #     reg_loss5.backward(retain_graph=True)
        # if reg_loss6 > 0:
        #     reg_loss6.backward(retain_graph=True)
        # if reg_loss7 > 0:
        #     reg_loss7.backward(retain_graph=True)
        # if reg_loss8 > 0:
        #     reg_loss8.backward(retain_graph=True)
        #
        # if cls_loss1 > 0:
        #     cls_loss1.backward(retain_graph=True)
        # if cls_loss2 > 0:
        #     cls_loss2.backward(retain_graph=True)
        # if cls_loss3 > 0:
        #     cls_loss3.backward(retain_graph=True)
        # if cls_loss4 > 0:
        #     cls_loss4.backward(retain_graph=True)
        # if cls_loss5 > 0:
        #     cls_loss5.backward(retain_graph=True)
        # if cls_loss6> 0:
        #     cls_loss6.backward(retain_graph=True)
        # if cls_loss7 > 0:
        #     cls_loss7.backward(retain_graph=True)
        # if cls_loss8 > 0:
        #     cls_loss8.backward(retain_graph=True)

        lffd_optimer.step()
        lffd_lr_scheduler.step()

        if lffd_lr_scheduler._step_count % param_model_save_interval == 0:
            torch.save(net.state_dict(), "./ckpt/{}.pth".format(lffd_lr_scheduler._step_count))


if __name__ == '__main__':
    lffd_train()
