import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial
import cv2

class SOLOHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 strides=[8, 8, 16, 32, 32],
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
                 epsilon=0.2,
                 num_grids=[40, 36, 24, 16, 12],
                 cate_down_pos=0,
                 with_deform=False,
                 mask_loss_cfg=dict(weight=3),
                 cate_loss_cfg=dict(gamma=2,
                                alpha=0.25,
                                weight=1),
                 postprocess_cfg=dict(cate_thresh=0.2,
                                      ins_thresh=0.5,
                                      pre_NMS_num=50,
                                      keep_instance=5,
                                      IoU_thresh=0.5)):
        super(SOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.epsilon = epsilon
        self.cate_down_pos = cate_down_pos
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform

        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg
        # initialize the layers for cate and mask branch, and initialize the weights
        self._init_layers()
        self._init_weights()

        # check flag
        assert len(self.ins_head) == self.stacked_convs
        assert len(self.cate_head) == self.stacked_convs
        assert len(self.ins_out_list) == len(self.strides)
        pass

    # This function build network layer for cate and ins branch
    # it builds 4 self.var
        # self.cate_head is nn.ModuleList 7 inter-layers of conv2d
        # self.ins_head is nn.ModuleList 7 inter-layers of conv2d
        # self.cate_out is 1 out-layer of conv2d
        # self.ins_out_list is nn.ModuleList len(self.seg_num_grids) out-layers of conv2d, one for each fpn_feat
    def _init_layers(self):
        ## TODO initialize layers: stack intermediate layer and output layer
        # define groupnorm
        num_groups = 32
        # initial the two branch head modulelist
        self.cate_head = nn.ModuleList()
        self.ins_head = nn.ModuleList()

        # initialize each layer of the category head
        for i in range(self.stacked_convs):
            in_channels = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_head.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=self.seg_feat_channels, 
                              kernel_size=3, stride=1, padding=1,bias=False),
                    nn.GroupNorm(num_groups=num_groups, num_channels=self.seg_feat_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # category branch output layer
        self.cate_out = nn.Sequential(
            nn.Conv2d(in_channels=self.seg_feat_channels, out_channels=self.cate_out_channels,  
                kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

        # initialize each layer of the instance head
        for i in range(self.stacked_convs):
            in_channels = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.ins_head.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=self.seg_feat_channels,
                        kernel_size=3, stride=1, padding=1, bias=False),
                    nn.GroupNorm(num_groups=num_groups, num_channels=self.seg_feat_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # mask branch output layers (one for each FPN level)
        self.ins_out_list = nn.ModuleList()
        for seg_num_grid in self.seg_num_grids:
            self.ins_out_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.seg_feat_channels, out_channels=seg_num_grid ** 2, 
                        kernel_size=1, stride=1, padding=0, bias=True),
                    nn.Sigmoid()
                )
            )

    # This function initialize weights for head network
    def _init_weights(self):
        #initialize weights for category branch
        for m in self.cate_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize weights for the category output layer
        for m in self.cate_out.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # initialize weights for the mask branch
        for m in self.ins_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize weights for the mask output layers
        for ins_out in self.ins_out_list:
            for m in ins_out.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


    # Forward function should forward every levels in the FPN.
    # this is done by map function or for loop
    # Input:
        # fpn_feat_list: backout_list of resnet50-fpn
    # Output:
        # if eval = False
            # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred_list: list, len(fpn_level), each (bz,S,S,C-1) / after point_NMS
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, Ori_H, Ori_W) / after upsampling
    def forward(self,
                fpn_feat_list,
                eval=False):
        new_fpn_list = self.NewFPN(fpn_feat_list)  # stride[8,8,16,32,32]
        assert new_fpn_list[0].shape[1:] == (256,100,136)

        quart_shape = [new_fpn_list[0].shape[-2]*2, new_fpn_list[0].shape[-1]*2]  # stride: 4
        # TODO: use MultiApply to compute cate_pred_list, ins_pred_list. Parallel w.r.t. feature level.
        cate_pred_list, ins_pred_list = self.MultiApply(
            self.forward_single_level,
            new_fpn_list,
            list(range(len(new_fpn_list))),
            eval=eval,
            upsample_shape=quart_shape
        )

        assert len(new_fpn_list) == len(self.seg_num_grids)

        # assert cate_pred_list[1].shape[1] == self.cate_out_channels
        assert ins_pred_list[1].shape[1] == self.seg_num_grids[1]**2
        assert cate_pred_list[1].shape[2] == self.seg_num_grids[1]
        return cate_pred_list, ins_pred_list

    # This function upsample/downsample the fpn level for the network
    # In paper author change the original fpn level resolution
    # Input:
        # fpn_feat_list, list, len(FPN), stride[4,8,16,32,64]
    # Output:
    # new_fpn_list, list, len(FPN), stride[8,8,16,32,32]
    def NewFPN(self, fpn_feat_list):
        new_fpn_list = []

        fpn_p2 = fpn_feat_list[0]
        fpn_p2_downsampled = F.interpolate(fpn_p2, scale_factor=0.5, mode='bilinear', align_corners=False)
        new_fpn_list.append(fpn_p2_downsampled)

        new_fpn_list.extend(fpn_feat_list[1:-1])

        fpn_p6 = fpn_feat_list[-1]
        fpn_p6_upsampled = F.interpolate(fpn_p6, scale_factor=2, mode='bilinear', align_corners=False)
        new_fpn_list.append(fpn_p6_upsampled)

        return new_fpn_list

    # This function forward a single level of fpn_featmap through the network
    # Input:
        # fpn_feat: (bz, fpn_channels(256), H_feat, W_feat)
        # idx: indicate the fpn level idx, num_grids idx, the ins_out_layer idx
    # Output:
        # if eval==False
            # cate_pred: (bz,C-1,S,S)
            # ins_pred: (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred: (bz,S,S,C-1) / after point_NMS
            # ins_pred: (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling
    def forward_single_level(self, fpn_feat, idx, eval=False, upsample_shape=None):
        # upsample_shape is used in eval mode
        ## TODO: finish forward function for single level in FPN.
        ## Notice, we distinguish the training and inference.
 
        num_grid = self.seg_num_grids[idx]  # current level grid

        # category branch
        cate_feat = fpn_feat
        for layer in self.cate_head:
            cate_feat = layer(cate_feat)
        # resize to S x S
        cate_feat = F.interpolate(cate_feat, size=(num_grid, num_grid), mode='bilinear', align_corners=False)
        cate_pred = self.cate_out(cate_feat)  # shape: (bz, C-1, S, S)

        # mask branch
        batch_size, _, h, w = fpn_feat.size()
        # generate coordinate feature maps
        y_range = torch.linspace(-1, 1, steps=h, device=fpn_feat.device)
        x_range = torch.linspace(-1, 1, steps=w, device=fpn_feat.device)
        y = y_range.view(-1, 1).repeat(1, w)
        x = x_range.view(1, -1).repeat(h, 1)
        y = y.expand([batch_size, 1, -1, -1])
        x = x.expand([batch_size, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        # concatenate coordinate features
        ins_feat = torch.cat([fpn_feat, coord_feat], dim=1)  # shape: (bz, 258, H_feat, W_feat)
        # process through mask head
        for layer in self.ins_head:
            ins_feat = layer(ins_feat)
        # upsample feature map by factor of 2
        ins_feat = F.interpolate(ins_feat, scale_factor=2, mode='bilinear', align_corners=False)
        # output mask predictions
        ins_pred = self.ins_out_list[idx](ins_feat)  # shape: (bz, S^2, 2H_feat, 2W_feat)

        # in inference time, upsample the pred to (ori image size/4)
        if eval == True:
            ## TODO resize ins_pred
            ins_pred = F.interpolate(ins_pred, size=upsample_shape, mode='bilinear', align_corners=False)
            cate_pred = self.points_nms(cate_pred).permute(0,2,3,1) # shape: (bz, S, S, C-1)
        # check flag
        if eval == False:
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (num_grid**2, fpn_feat.shape[2]*2, fpn_feat.shape[3]*2)
        else:
            pass
        return cate_pred, ins_pred

    # Credit to SOLO Author's code
    # This function do a NMS on the heat map(cate_pred), grid-level
    # Input:
        # heat: (bz,C-1, S, S)
    # Output:
        # (bz,C-1, S, S)
    def points_nms(self, heat, kernel=2):
        # kernel must be 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep

    # This function compute loss for a batch of images
    # input:
        # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    # output:
        # cate_loss, mask_loss, total_loss
    def loss(self,
             cate_pred_list,
             ins_pred_list,
             ins_gts_list,
             ins_ind_gts_list,
             cate_gts_list):
        ## TODO: compute loss, vecterize this part will help a lot. To avoid potential ill-conditioning, 
        # if necessary, add a very small number to denominator for focalloss and diceloss computation.
        device = cate_pred_list[0].device
        num_classes = self.cate_out_channels

        # ----------------------------
        # Process mask predictions and ground truths
        # ----------------------------
        # Uniform the expression for ins_gts & ins_preds
        # ins_gts: list, len(fpn), (total_positive, 2H_feat, 2W_feat)
        # ins_preds: list, len(fpn), (total_positive, 2H_feat, 2W_feat)
        ins_gts = []
        ins_preds = []

        for level_idx in range(len(ins_pred_list)):
            ins_pred_level = ins_pred_list[level_idx]
            ins_ind_level = [ins_ind_gts_list[b][level_idx] for b in range(len(ins_ind_gts_list))]
            ins_gt_level = [ins_gts_list[b][level_idx] for b in range(len(ins_gts_list))]

            # Collect positive samples across the batch
            pos_inds = []
            pos_ins_preds = []
            pos_ins_gts = []

            for b in range(len(ins_ind_level)):
                pos_ind = ins_ind_level[b].nonzero(as_tuple=False).squeeze(1)
                if pos_ind.numel() == 0:
                    continue
                pos_inds.append(pos_ind)
                pos_ins_preds.append(ins_pred_level[b][pos_ind])
                pos_ins_gts.append(ins_gt_level[b][pos_ind])

            if len(pos_ins_preds) == 0:
                continue

            ins_preds.append(torch.cat(pos_ins_preds, dim=0))
            ins_gts.append(torch.cat(pos_ins_gts, dim=0))

        # Compute mask loss using DiceLoss
        mask_losses = []
        for pred_masks, gt_masks in zip(ins_preds, ins_gts):
            if pred_masks.size(0) == 0:
                continue
            for mask_pred, mask_gt in zip(pred_masks, gt_masks):
                loss = self.DiceLoss(mask_pred, mask_gt)
                mask_losses.append(loss)
        if len(mask_losses) > 0:
            mask_loss = torch.stack(mask_losses).mean()
        else:
            mask_loss = torch.tensor(0.0, device=device)

        # ----------------------------
        # Process category predictions and ground truths
        # ----------------------------
        # Uniform the expression for cate_gts & cate_preds
        # cate_gts: (total_entries,)
        # cate_preds: (total_entries, C-1)
        cate_gts = []
        cate_preds = []

        for level_idx in range(len(cate_pred_list)):
            cate_pred_level = cate_pred_list[level_idx]
            cate_pred_level = cate_pred_level.permute(0, 2, 3, 1).reshape(-1, num_classes)
            cate_preds.append(cate_pred_level)

            cate_gt_level = [cate_gts_list[b][level_idx].flatten() for b in range(len(cate_gts_list))]
            cate_gts.append(torch.cat(cate_gt_level))

        cate_preds = torch.cat(cate_preds, dim=0)
        cate_gts = torch.cat(cate_gts, dim=0).to(device)

        # Compute category loss using FocalLoss
        cate_loss = self.FocalLoss(cate_preds, cate_gts)

        # ----------------------------
        # Total loss
        # ----------------------------
        total_loss = cate_loss + 3 * mask_loss

        return cate_loss, mask_loss, total_loss



    # This function compute the DiceLoss
    # Input:
        # mask_pred: (2H_feat, 2W_feat)
        # mask_gt: (2H_feat, 2W_feat)
    # Output: dice_loss, scalar
    def DiceLoss(self, mask_pred, mask_gt):
        ## TODO: compute DiceLoss
        # Flatten the tensors
        mask_pred = mask_pred.contiguous().view(-1)
        mask_gt = mask_gt.contiguous().view(-1)

        # Compute Dice coefficient
        intersection = 2 * torch.sum(mask_pred * mask_gt)
        union = torch.sum(mask_pred ** 2) + torch.sum(mask_gt ** 2) + 1e-5  # Add epsilon to avoid division by zero

        dice_loss = 1 - (intersection / union)
        return dice_loss

    # This function compute the cate loss
    # Input:
        # cate_preds: (num_entry, C-1)
        # cate_gts: (num_entry,)
    # Output: focal_loss, scalar
    def FocalLoss(self, cate_preds, cate_gts):
        alpha = 0.25
        gamma = 2.0
        ## TODO: compute focalloss
        device = cate_preds.device
        num_classes = self.cate_out_channels

        # Prepare ground truth labels
        # cate_gts contains values in {0,1,2,3}, where 0 is background
        # Adjust labels to be in range {0, ..., C-1}
        valid_mask = (cate_gts >= 0)
        cate_preds = cate_preds[valid_mask]
        cate_gts = cate_gts[valid_mask]

        if cate_preds.numel() == 0:
            return torch.tensor(0.0, device=device)

        # Create one-hot encoding for labels (background class is 0)
        cate_labels = torch.zeros_like(cate_preds, device=device)
        fg_mask = (cate_gts > 0)
        if fg_mask.sum() > 0:
            idx = cate_gts[fg_mask] - 1  # Shift class indices to start from 0
            cate_labels[fg_mask, idx.long()] = 1

        # Compute sigmoid of predictions
        pred_sigmoid = cate_preds.sigmoid()

        # Compute pt and focal weight
        pt = torch.where(cate_labels == 1, pred_sigmoid, 1 - pred_sigmoid)
        alpha_t = torch.where(cate_labels == 1, alpha, 1 - alpha)
        focal_weight = alpha_t * ((1 - pt) ** gamma)

        # Compute binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(cate_preds, cate_labels, reduction='none')

        # Compute final loss
        loss = focal_weight * bce_loss
        loss = loss.sum() / cate_preds.size(0)

        return loss

    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)

        return tuple(map(list, zip(*map_results)))

    # This function build the ground truth tensor for each batch in the training
    # Input:
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # / ins_pred_list is only used to record feature map
        # bbox_list: list, len(batch_size), each (n_object, 4) (x1y1x2y2 system)
        # label_list: list, len(batch_size), each (n_object, )
        # mask_list: list, len(batch_size), each (n_object, 800, 1088)
    # Output:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    def target(self,
               ins_pred_list,
               bbox_list,
               label_list,
               mask_list):
        # TODO: use MultiApply to compute ins_gts_list, ins_ind_gts_list, cate_gts_list. Parallel w.r.t. img mini-batch
        # remember, you want to construct target of the same resolution as prediction output in training

        featmap_sizes = [ins_pred.shape[-2:] for ins_pred in ins_pred_list]

        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.MultiApply(
            self.target_single_img,
            bbox_list,
            label_list,
            mask_list,
            featmap_sizes=featmap_sizes
        )

        # check flag
        assert ins_gts_list[0][1].shape == (self.seg_num_grids[1]**2, 200, 272)
        assert ins_ind_gts_list[0][1].shape == (self.seg_num_grids[1]**2,)
        assert cate_gts_list[0][1].shape == (self.seg_num_grids[1], self.seg_num_grids[1])

        return ins_gts_list, ins_ind_gts_list, cate_gts_list
    # -----------------------------------
    ## process single image in one batch
    # -----------------------------------
    # input:
        # gt_bboxes_raw: n_obj, 4 (x1y1x2y2 system)
        # gt_labels_raw: n_obj,
        # gt_masks_raw: n_obj, H_ori, W_ori
        # featmap_sizes: list of shapes of featmap
    # output:
        # ins_label_list: list, len: len(FPN), (S^2, 2H_feat, 2W_feat)
        # cate_label_list: list, len: len(FPN), (S, S)
        # ins_ind_label_list: list, len: len(FPN), (S^2, )
    def target_single_img(self,
                          gt_bboxes_raw,
                          gt_labels_raw,
                          gt_masks_raw,
                          featmap_sizes=None):
        ## TODO: finish single image target build
        # compute the area of every object in this single image
        num_levels = len(self.seg_num_grids)

        # initial the output list, each entry for one featmap
        ins_label_list = []
        ins_ind_label_list = []
        cate_label_list = []

        num_instances = len(gt_labels_raw)

        img_h, img_w = gt_masks_raw[0].shape[-2:]

        #converting lists to tensors
        gt_bboxes = torch.tensor(gt_bboxes_raw, dtype=torch.float32)
        gt_labels = torch.tensor(gt_labels_raw, dtype=torch.float32)
        gt_masks = torch.stack(
            [torch.tensor(mask, dtype=torch.uint8) for mask in gt_masks_raw], 
            dim=0
        )

        x1 = gt_bboxes[:, 0]
        y1 = gt_bboxes[:, 1]
        x2 = gt_bboxes[:, 2]
        y2 = gt_bboxes[:, 3]
        w = x2 - x1
        h = y2 - y1
        areas = w * h
        instance_scales = torch.sqrt(areas)

        for level_idx in range(num_levels):
            num_grid = self.seg_num_grids[level_idx]
            featmap_size = featmap_sizes[level_idx]
            feat_h, feat_w = featmap_size
            mask_size = (feat_h, feat_w)
            min_scale, max_scale = self.scale_ranges[level_idx]

            cate_label = torch.zeros((num_grid, num_grid), dtype=torch.int64)
            ins_label = torch.zeros((num_grid ** 2, feat_h, feat_w), dtype=torch.uint8)
            ins_ind_label = torch.zeros((num_grid ** 2,), dtype=torch.uint8)

            # for each instance
            for idx in range(num_instances):
                instance_scale = instance_scales[idx]
                if instance_scale > max_scale or instance_scale < min_scale:
                    continue

                cur_mask = gt_masks[idx]
                ys, xs = torch.nonzero(cur_mask, as_tuple=True)

                if len(ys) == 0:
                    continue
                
                cx = xs.float().mean()
                cy = ys.float().mean()

                # norm center to [0,1]
                cx_norm = cx / img_w
                cy_norm = cy / img_h

                grid_x = min(int(torch.floor(cx_norm * num_grid)), num_grid - 1)
                grid_y = min(int(torch.floor(cy_norm * num_grid)), num_grid - 1)

                grid_x = max(0, grid_x)
                grid_y = max(0, grid_y)

                bbox_w = w[idx]
                bbox_h = h[idx]

                half_w = 0.5 * bbox_w * self.epsilon / img_w
                half_h = 0.5 * bbox_h * self.epsilon / img_h

                top_box = (cy_norm - half_h) * num_grid
                down_box = (cy_norm + half_h) * num_grid
                left_box = (cx_norm - half_w) * num_grid
                right_box = (cx_norm + half_w) * num_grid

                top = max(int(torch.floor(top_box)), 0)
                down = min(int(torch.ceil(down_box)), num_grid)
                left = max(int(torch.floor(left_box)), 0)
                right = min(int(torch.ceil(right_box)), num_grid)

                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        cate_label[i,j] = gt_labels[idx]
                        
                        k = i * num_grid + j
                        ins_ind_label[k] = 1

                        cur_mask_resized = F.interpolate(
                            cur_mask.unsqueeze(0).unsqueeze(0).float(),
                            size=mask_size,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)

                        ins_label[k] = cur_mask_resized

            ins_label_list.append(ins_label)
            ins_ind_label_list.append(ins_ind_label)
            cate_label_list.append(cate_label)


        # check flag
        assert ins_label_list[1].shape == (1296,200,272)
        assert ins_ind_label_list[1].shape == (1296,)
        assert cate_label_list[1].shape == (36, 36)
        return ins_label_list, ins_ind_label_list, cate_label_list
    

    # This function receive pred list from forward and post-process
    # Input:
        # ins_pred_list: list, len(fpn), (bz,S^2,Ori_H/4, Ori_W/4)
        # cate_pred_list: list, len(fpn), (bz,S,S,C-1)
        # ori_size: [ori_H, ori_W]
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)

    def PostProcess(self,
                    ins_pred_list,
                    cate_pred_list,
                    ori_size):

        ## TODO: finish PostProcess
        # Initialize output lists
        NMS_sorted_scores_list = []
        NMS_sorted_cate_label_list = []
        NMS_sorted_ins_list = []

        batch_size = ins_pred_list[0].size(0)

        for img_idx in range(batch_size):
            # Collect per-image predictions from all levels
            ins_pred_img_list = []
            cate_pred_img_list = []

            for level_idx in range(len(self.seg_num_grids)):
                ins_pred_level = ins_pred_list[level_idx]
                cate_pred_level = cate_pred_list[level_idx]

                ins_pred_img_level = ins_pred_level[img_idx]  # Shape: (S^2, ori_H/4, ori_W/4)
                cate_pred_img_level = cate_pred_level[img_idx]  # Shape: (S, S, C-1)

                # Reshape cate_pred_img_level to (S^2, C-1)
                S = cate_pred_img_level.size(0)
                cate_pred_img_level = cate_pred_img_level.view(-1, self.cate_out_channels)

                ins_pred_img_list.append(ins_pred_img_level)
                cate_pred_img_list.append(cate_pred_img_level)

            # Concatenate predictions from all levels
            ins_pred_img = torch.cat(ins_pred_img_list, dim=0)  # Shape: (sum(S^2), ori_H/4, ori_W/4)
            cate_pred_img = torch.cat(cate_pred_img_list, dim=0)  # Shape: (sum(S^2), C-1)

            # Process single image
            NMS_sorted_scores, NMS_sorted_cate_labels, NMS_sorted_ins = self.PostProcessImg(
                ins_pred_img, cate_pred_img, ori_size)

            NMS_sorted_scores_list.append(NMS_sorted_scores)
            NMS_sorted_cate_label_list.append(NMS_sorted_cate_labels)
            NMS_sorted_ins_list.append(NMS_sorted_ins)

        return NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list


    # This function Postprocess on single img
    # Input:
        # ins_pred_img: (all_level_S^2, ori_H/4, ori_W/4)
        # cate_pred_img: (all_level_S^2, C-1)
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
    def PostProcessImg(self,
                       ins_pred_img,
                       cate_pred_img,
                       ori_size):

        ## TODO: PostProcess on single image.
        # Get parameters from postprocess config
        cate_thresh = self.postprocess_cfg['cate_thresh']
        pre_NMS_num = self.postprocess_cfg['pre_NMS_num']
        keep_instance = self.postprocess_cfg['keep_instance']
        ins_thresh = self.postprocess_cfg['ins_thresh']

        ori_H, ori_W = ori_size

        # Compute the scores and labels
        # For each grid, get the max score and its category label
        scores, cate_labels = torch.max(cate_pred_img, dim=1)  # Shape: (sum(S^2),)

        # Filter out low-confidence predictions
        keep_inds = (scores > cate_thresh)
        scores = scores[keep_inds]
        cate_labels = cate_labels[keep_inds]
        ins_pred_img = ins_pred_img[keep_inds]  # Shape: (n_keep, ori_H/4, ori_W/4)

        if scores.numel() == 0:
            return [], [], []

        # If number of predictions is larger than pre_NMS_num, keep top pre_NMS_num
        if scores.numel() > pre_NMS_num:
            topk_inds = scores.topk(pre_NMS_num)[1]
            scores = scores[topk_inds]
            cate_labels = cate_labels[topk_inds]
            ins_pred_img = ins_pred_img[topk_inds]

        # Binarize the masks
        masks = (ins_pred_img > ins_thresh).float()

        # Compute mask areas
        mask_areas = masks.sum(dim=(1, 2))

        # Filter out masks with very small area
        keep_inds = mask_areas > 0
        scores = scores[keep_inds]
        cate_labels = cate_labels[keep_inds]
        masks = masks[keep_inds]

        if scores.numel() == 0:
            return [], [], []

        # Apply Matrix NMS
        decay_scores = self.MatrixNMS(masks, scores)

        # Multiply decay_scores with original scores
        scores = scores * decay_scores

        # Filter out masks with zero score
        keep_inds = scores > 0
        scores = scores[keep_inds]
        cate_labels = cate_labels[keep_inds]
        masks = masks[keep_inds]

        if scores.numel() == 0:
            return [], [], []

        # Sort the scores
        scores, sort_inds = torch.sort(scores, descending=True)
        masks = masks[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Keep top K instances
        if scores.numel() > keep_instance:
            scores = scores[:keep_instance]
            masks = masks[:keep_instance]
            cate_labels = cate_labels[:keep_instance]

        # Resize masks to original image size
        N = masks.size(0)
        masks = F.interpolate(masks.unsqueeze(1), size=(ori_H, ori_W), mode='bilinear', align_corners=False)
        masks = masks.squeeze(1)

        # Binarize masks again after interpolation
        masks = (masks > 0.5).float()

        return scores.tolist(), cate_labels.tolist(), masks.cpu().numpy()

    # This function perform matrix NMS
    # Input:
        # sorted_ins: (n_act, ori_H/4, ori_W/4)
        # sorted_scores: (n_act,)
    # Output:
        # decay_scores: (n_act,)
    def MatrixNMS(self, sorted_ins, sorted_scores, method='gauss', gauss_sigma=0.5):
        ## TODO: finish MatrixNMS
        n_samples = sorted_scores.size(0)
        if n_samples == 0:
            return []

        # Flatten masks
        masks = sorted_ins.view(n_samples, -1).float()  # Shape: (n_samples, h*w)

        # Compute pairwise IoU
        inter_matrix = torch.mm(masks, masks.t())  # Intersection areas
        areas = masks.sum(dim=1).unsqueeze(1)  # Areas of masks
        union_matrix = areas + areas.t() - inter_matrix + 1e-5  # Union areas
        iou_matrix = inter_matrix / union_matrix  # IoU matrix

        # Duplicate scores
        scores = sorted_scores.view(-1, 1)
        scores_t = scores.t()

        # Only consider lower scored masks
        # For each mask, compare with masks that have higher scores
        lower_mask = scores_t > scores

        # Initialize decay matrix
        decay_matrix = torch.ones_like(iou_matrix)

        # Compute decay factors
        if method == 'gauss':
            decay = torch.exp(-1 * (iou_matrix ** 2) / gauss_sigma)
        else:
            decay = (1 - iou_matrix) / (1 - iou_matrix.max(dim=0, keepdim=True)[0] + 1e-5)

        # Apply decay only for lower scored masks
        decay_matrix = torch.where(lower_mask, decay, decay_matrix)

        # For each mask, compute the cumulative decay
        decay_factors = decay_matrix.min(dim=0)[0]

        # Ensure decay factors are between 0 and 1
        decay_factors = decay_factors.clamp(min=0.0, max=1.0)

        return decay_factors

    # -----------------------------------
    ## The following code is for visualization
    # -----------------------------------
    # this function visualize the ground truth tensor
    # Input:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
        # color_list: list, len(C-1)
        # img: (bz,3,Ori_H, Ori_W)
        ## self.strides: [8,8,16,32,32]
    def PlotGT(self,
               ins_gts_list,
               ins_ind_gts_list,
               cate_gts_list,
               color_list,
               img, index):
        ## TODO: target image recover, for each image, recover their segmentation in 5 FPN levels.
        ## This is an important visual check flag.
        batch_size = len(ins_gts_list)
        num_levels = len(self.seg_num_grids)

        for img_idx in range(batch_size):
            img_np = img[img_idx].permute(1,2,0).cpu().numpy()

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean

            img_np = np.clip(img_np, 0, 1)
            
            fig, axes = plt.subplots(1, num_levels + 1, figsize=(20, 5))
            
            # Display the original image without any alterations
            axes[0].imshow(img_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            for level_idx in range(num_levels):
                num_grid = self.seg_num_grids[level_idx]
                cate_label = cate_gts_list[img_idx][level_idx].cpu().numpy()
                ins_label = ins_gts_list[img_idx][level_idx]
                ins_ind_label = ins_ind_gts_list[img_idx][level_idx]

                feat_h, feat_w = ins_label.shape[1], ins_label.shape[2]
                upsampled_mask = np.zeros((img_np.shape[0], img_np.shape[1]))

                for k in range(num_grid ** 2):
                    if ins_ind_label[k] == 0:
                        continue
                    mask = ins_label[k].cpu().numpy()
                    mask = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))
                    upsampled_mask = np.maximum(upsampled_mask, mask)

                cmap = plt.get_cmap(color_list[level_idx % len(color_list)])
                colored_mask = cmap(upsampled_mask)
                
                # Create an overlay of the mask on the original image
                overlay = img_np.copy()
                overlay[upsampled_mask > 0] = colored_mask[upsampled_mask > 0, :3]
                
                # Blend the overlay with the original image
                alpha = 0.5
                blended = (1 - alpha) * img_np + alpha * overlay

                axes[level_idx + 1].imshow(blended)
                axes[level_idx + 1].set_title(f'FPN Level {level_idx}')
                axes[level_idx + 1].axis('off')
                
            # if img_idx >= 100:
            #     break

            plt.tight_layout()
            
            plt.savefig(f'./fpn_imgs/fpn_img{index}.png')
            plt.close()


            # img_np = (img_np * 255).astype(np.uint8)
            # fig, axes = plt.subplots(1, num_levels + 1, figsize=(15, 5))
            # axes[0].imshow(img_np)
            # axes[0].set_title('Original Image')
            # axes[0].axis('off')

            # for level_idx in range(num_levels):
            #     num_grid = self.seg_num_grids[level_idx]
            #     # Get the category labels and instance masks
            #     cate_label = cate_gts_list[img_idx][level_idx].cpu().numpy()
            #     ins_label = ins_gts_list[img_idx][level_idx]
            #     ins_ind_label = ins_ind_gts_list[img_idx][level_idx]

            #     # Initialize an empty mask for visualization
            #     feat_h, feat_w = ins_label.shape[1], ins_label.shape[2]
            #     upsampled_mask = np.zeros((img_np.shape[0], img_np.shape[1]))

            #     # For each activated grid cell
            #     for k in range(num_grid ** 2):
            #         if ins_ind_label[k] == 0:
            #             continue
            #         # Get the resized mask
            #         mask = ins_label[k].cpu().numpy()
            #         # Upsample the mask to match the original image size
            #         mask = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))
            #         upsampled_mask = np.maximum(upsampled_mask, mask)

            #     # Apply colormap
            #     cmap = plt.get_cmap(color_list[level_idx % len(color_list)])
            #     colored_mask = cmap(upsampled_mask)
            #     # Overlay on the original image
                
            #     overlay = (0.5 * img_np + 0.5 * (colored_mask[:, :, :3] * 255)).astype(np.uint8)

            #     axes[level_idx + 1].imshow(overlay)
            #     axes[level_idx + 1].set_title(f'FPN Level {level_idx}')
            #     axes[level_idx + 1].axis('off')

            # plt.show()


    # This function plot the inference segmentation in img
    # Input:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
        # color_list: ["jet", "ocean", "Spectral"]
        # img: (bz, 3, ori_H, ori_W)
    def PlotInfer(self,
                  NMS_sorted_scores_list,
                  NMS_sorted_cate_label_list,
                  NMS_sorted_ins_list,
                  color_list,
                  img,
                  iter_ind):
        ## TODO: Plot predictions
        batch_size = len(NMS_sorted_scores_list)

        for img_idx in range(batch_size):
            img_np = img[img_idx].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, 3)
            ori_H, ori_W, _ = img_np.shape

            # Denormalize the image if necessary
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)

            # Get the masks, scores, and labels for this image
            scores = NMS_sorted_scores_list[img_idx]
            cate_labels = NMS_sorted_cate_label_list[img_idx]
            masks = NMS_sorted_ins_list[img_idx]  # Shape: (keep_instance, ori_H, ori_W)

            # Apply the hard threshold to the masks
            masks = (masks >= 0.5).astype(np.uint8)

            # Set a threshold for the NMS score to select instance segmentation
            score_thresh = 0.5
            keep_inds = [i for i, s in enumerate(scores) if s >= score_thresh]

            if len(keep_inds) == 0:
                print(f"No instances to display for image {img_idx} in iteration {iter_ind}")
                continue

            scores = [scores[i] for i in keep_inds]
            cate_labels = [cate_labels[i] for i in keep_inds]
            masks = masks[keep_inds]  # Shape: (n_instances, ori_H, ori_W)

            # Create a figure and axis
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(img_np)
            ax.axis('off')

            # Overlay masks
            img_overlay = img_np.copy()
            for i, mask in enumerate(masks):
                color = plt.get_cmap(color_list[cate_labels[i] % len(color_list)])(0.5)
                mask_bool = mask.astype(bool)

                # Create colored mask
                color_mask = np.zeros_like(img_np)
                color_mask[mask_bool] = color[:3]

                # Blend the mask with the image
                img_overlay[mask_bool] = img_overlay[mask_bool] * 0.5 + color_mask[mask_bool] * 0.5

                # Optionally, you can add boundaries or contours
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_overlay, contours, -1, color[:3], 2)

                # Add class label and score
                y_coords, x_coords = np.where(mask_bool)
                if y_coords.size > 0 and x_coords.size > 0:
                    y_min, x_min = y_coords.min(), x_coords.min()
                    ax.text(x_min, y_min - 5, f'Class {cate_labels[i]+1}: {scores[i]:.2f}', color='white',
                            fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

            # Show the final image
            ax.imshow(img_overlay)
            plt.tight_layout()
            plt.savefig(f'./infer_imgs/infer_img_{iter_ind}_{img_idx}.png')
            plt.close()

from backbone import *


def part_a_solo_head():
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
    
    
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()


    resnet50_fpn = Resnet50Backbone()
    solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.
    # loop the image
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        # fpn is a dict
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        # make the target


        ## demo
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                         bbox_list,
                                                                         label_list,
                                                                         mask_list)
        mask_color_list = ["jet", "ocean", "Spectral"]
        solo_head.PlotGT(ins_gts_list,ins_ind_gts_list,cate_gts_list,mask_color_list,img, iter)
        
        if iter >= 10:
            break


if __name__ == '__main__':
    part_a_solo_head()