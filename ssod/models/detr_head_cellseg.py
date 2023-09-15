# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils import preprocess_panoptic_gt

@HEADS.register_module()
class DETRHead_CellSeg(AnchorFreeHead):
    """Implements the DETR transformer head.

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_point=dict(type='L1Loss', loss_weight=5.0),
                 loss_mask=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    reduction='mean',
                    loss_weight=5.0),
                 loss_dice=dict(
                    type='DiceLoss',
                    use_sigmoid=True,
                    activate=True,
                    reduction='mean',
                    naive_dice=True,
                    eps=1.0,
                    loss_weight=5.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is DETRHead_CellSeg):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_point['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')

            self.sampler = build_sampler(sampler_cfg, context=self)
            mask_assigner = train_cfg["assigner_mask"]
            mask_sampler = train_cfg["sampler_mask"]
            self.mask_assigner = build_assigner(mask_assigner)
            self.mask_sampler = build_sampler(mask_sampler, context=self)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_point = build_loss(loss_point)
        self.class_weight = loss_cls.get('loss_weight', None)
        self.loss_mask_ = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.reg_ffn = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_reg_fcs,
            self.act_cfg,
            dropout=0.0,
            add_residual=False)
        self.fc_reg = Linear(self.embed_dims, 4)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is DETRHead_CellSeg:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single, feats, img_metas_list)

    def forward_single(self, x, img_metas):
        """"Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
                                       pos_embed)

        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(self.activate(
            self.reg_ffn(outs_dec))).sigmoid()
        return all_cls_scores, all_bbox_preds

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores_list,
             all_bbox_preds_list,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # NOTE defaultly only the outputs from the last feature scale is used.
        all_cls_scores = all_cls_scores_list[-1]
        all_bbox_preds = all_bbox_preds_list[-1]
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    cls_scores,
                    point_preds,
                    gt_points_list,
                    gt_labels_list,
                    img_metas,
                    gt_points_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        point_preds_list = [point_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, point_preds_list,
                                           gt_points_list, gt_labels_list,
                                           img_metas, gt_points_ignore_list)
        (labels_list, label_weights_list, point_targets_list, point_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        point_targets = torch.cat(point_targets_list, 0)
        point_weights = torch.cat(point_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, point_pred in zip(img_metas, point_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = point_pred.new_tensor([img_w, img_h]).unsqueeze(0).repeat(
                                               point_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        point_preds = point_preds.reshape(-1, 2)
        points = point_preds * factors
        points_gt = point_targets * factors

        # regression L1 loss
        loss_bbox = self.loss_point(
            point_preds, point_targets, point_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox

    def get_targets(self,
                    cls_scores_list,
                    point_preds_list,
                    gt_points_list,
                    gt_labels_list,
                    img_metas,
                    gt_points_ignore_list=None):
        assert gt_points_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_points_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, point_targets_list,
         point_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, point_preds_list,
             gt_points_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, point_targets_list,
                point_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           point_pred,
                           gt_points,
                           gt_labels,
                           img_meta,
                           gt_points_ignore=None):

        num_points = point_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(point_pred, cls_score, gt_points,
                                             gt_labels, img_meta,
                                             gt_points_ignore)
        sampling_result = self.sampler.sample(assign_result, point_pred,
                                              gt_points)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_points.new_full((num_points, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_points.new_ones(num_points)

        # bbox targets
        point_targets = torch.zeros_like(point_pred)
        point_weights = torch.zeros_like(point_pred)
        point_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # Edge Loss Up
        # point_weights[torch.where((img_h - gt_points[:, 0]) <= 50)] = 2.0
        # point_weights[torch.where((img_w - gt_points[:, 1]) <= 50)] = 2.0
        # point_weights[torch.where((gt_points[:, 0]) <= 50)] = 2.0
        # point_weights[torch.where((gt_points[:, 1]) <= 50)] = 2.0


        factor = point_pred.new_tensor([img_w, img_h]).unsqueeze(0)
        if sampling_result.pos_gt_bboxes.shape[0] == 0:
            pos_gt_points_normalized = sampling_result.pos_gt_bboxes[:,:2]
        else:
            pos_gt_points_normalized = sampling_result.pos_gt_bboxes / factor
        # pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        point_targets[pos_inds] = pos_gt_points_normalized
        return (labels, label_weights, point_targets, point_weights, pos_inds,
                neg_inds)

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      gt_points=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outputs_classes, outputs_coords, outputs_masks, \
        enc_outputs_class, \
        enc_outputs_coord = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = (outputs_classes, outputs_coords, gt_bboxes, img_metas)
        else:
            gt_labels, gt_masks = self.preprocess_gt(gt_labels, gt_masks,
                                                     None, img_metas)
            loss_inputs = (outputs_classes, outputs_coords, enc_outputs_class,
                           enc_outputs_coord, gt_bboxes, gt_labels, img_metas)
        losses = {}
        losses_centroid = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses_masks = self.loss_mask(outputs_classes, outputs_masks, gt_labels, gt_masks,
                           img_metas)
        losses.update(losses_centroid)
        losses.update(losses_masks)
        return losses

    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss_mask(self, all_cls_scores, all_mask_preds, gt_labels_list,
             gt_masks_list, img_metas):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice = multi_apply(
            self.loss_mask_single, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        return loss_dict

    def loss_mask_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (n, ). n is the sum of number of stuff
                types and number of instances in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single decoder\
                layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_mask_targets(cls_scores_list, mask_preds_list,
                                           gt_labels_list, gt_masks_list,
                                           img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        # class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=1)

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]
        target_shape = mask_targets.shape[-2:]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        # upsample to shape of target
        # shape (num_total_gts, h, w)
        mask_preds = F.interpolate(
            mask_preds.unsqueeze(1),
            target_shape,
            mode='bilinear',
            align_corners=False).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_preds, mask_targets, avg_factor=num_total_masks)

        # mask loss
        # FocalLoss support input of shape (n, num_class)
        h, w = mask_preds.shape[-2:]
        # shape (num_total_gts, h, w) -> (num_total_gts * h * w, 1)
        mask_preds = mask_preds.reshape(-1, 1)
        # shape (num_total_gts, h, w) -> (num_total_gts * h * w)
        mask_targets = mask_targets.reshape(-1)
        # target is (1 - mask_targets) !!!
        loss_mask = self.loss_mask_(
            mask_preds, 1 - mask_targets, avg_factor=num_total_masks * h * w)

        return loss_cls, loss_mask, loss_dice

    def get_mask_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - num_total_pos (int): Number of positive samples in\
                    all images.
                - num_total_neg (int): Number of negative samples in\
                    all images.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_mask_target_single, cls_scores_list,
                                      mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_mask_target_single(self, cls_score, mask_pred, gt_labels, gt_masks,
                           img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (n, ). n is the sum of number of stuff type and number
                of instance in a image.
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (n, h, w).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image.
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image.
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image.
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        target_shape = mask_pred.shape[-2:]
        if gt_masks.shape[0] > 0:
            gt_masks_downsampled = F.interpolate(
                gt_masks.unsqueeze(1).float(), target_shape,
                mode='nearest').squeeze(1).long()
        else:
            gt_masks_downsampled = gt_masks

        # assign and sample
        assign_result = self.mask_assigner.assign(cls_score, mask_pred, gt_labels,
                                             gt_masks_downsampled, img_metas)
        sampling_result = self.mask_sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_query, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones(self.num_query)

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_query, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    def preprocess_gt(self, gt_labels_list, gt_masks_list, gt_semantic_segs,
                      img_metas):
        """Preprocess the ground truth for all images.

        Args:
            gt_labels_list (list[Tensor]): Each is ground truth
                labels of each bbox, with shape (num_gts, ).
            gt_masks_list (list[BitmapMasks]): Each is ground truth
                masks of each instances of a image, shape
                (num_gts, h, w).
            gt_semantic_seg (Tensor | None): Ground truth of semantic
                segmentation with the shape (batch_size, n, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple: a tuple containing the following targets.
                - labels (list[Tensor]): Ground truth class indices\
                    for all images. Each with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in a image.
                - masks (list[Tensor]): Ground truth mask for each\
                    image, each with shape (n, h, w).
        """
        num_things_list = [self.num_classes] * len(gt_labels_list)
        num_stuff_list = [0] * len(gt_labels_list)
        if gt_semantic_segs is None:
            gt_semantic_segs = [None] * len(gt_labels_list)

        targets = multi_apply(preprocess_panoptic_gt, gt_labels_list,
                              gt_masks_list, gt_semantic_segs, num_things_list,
                              num_stuff_list, img_metas)
        labels, masks = targets
        return labels, masks

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores_list,
                   all_bbox_preds_list,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.
        cls_scores = all_cls_scores_list[-1][-1]
        bbox_preds = all_bbox_preds_list[-1][-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)

        return result_list

    def _get_points_single_pred(self,
                           cls_score,
                           point_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(point_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            point_index = indexes // self.num_classes
            point_pred = point_pred[point_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, point_index = scores.topk(max_per_img)
            point_pred = point_pred[point_index]
            det_labels = det_labels[point_index]

        det_points = point_pred
        det_points[:, 0] = det_points[:, 0] * img_shape[1]
        det_points[:, 1] = det_points[:, 1] * img_shape[0]
        det_points[:, 0].clamp_(min=0, max=img_shape[1])
        det_points[:, 1].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_points /= det_points.new_tensor(scale_factor)
        det_points = torch.cat((det_points, scores.unsqueeze(1)), -1)

        return det_points, det_labels

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)  # [cls, points]
        results_list = self.get_points_pred(*outs, img_metas, rescale=rescale)
        return results_list

    def simple_all_test_bboxes(self, feats, img_metas, rescale=False):
        outs = self.forward(feats, img_metas)  # [cls, points]
        results = self.get_all_points_pred(*outs, img_metas, rescale=rescale)

        return results

    def forward_onnx(self, feats, img_metas):
        """Forward function for exporting to ONNX.

        Over-write `forward` because: `masks` is directly created with
        zero (valid position tag) and has the same spatial size as `x`.
        Thus the construction of `masks` is different from that in `forward`.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single_onnx, feats, img_metas_list)

    def forward_single_onnx(self, x, img_metas):
        """"Forward function for a single feature level with ONNX exportation.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        # Note `img_shape` is not dynamically traceable to ONNX,
        # since the related augmentation was done with numpy under
        # CPU. Thus `masks` is directly created with zeros (valid tag)
        # and the same spatial shape as `x`.
        # The difference between torch and exported ONNX model may be
        # ignored, since the same performance is achieved (e.g.
        # 40.1 vs 40.1 for DETR)
        batch_size = x.size(0)
        h, w = x.size()[-2:]
        masks = x.new_zeros((batch_size, h, w))  # [B,h,w]

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        pos_embed = self.positional_encoding(masks)
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
                                       pos_embed)

        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(self.activate(
            self.reg_ffn(outs_dec))).sigmoid()
        return all_cls_scores, all_bbox_preds

    def onnx_export(self, all_cls_scores_list, all_bbox_preds_list, img_metas):
        """Transform network outputs into bbox predictions, with ONNX
        exportation.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        assert len(img_metas) == 1, \
            'Only support one input image while in exporting to ONNX'

        cls_scores = all_cls_scores_list[-1][-1]
        bbox_preds = all_bbox_preds_list[-1][-1]

        # Note `img_shape` is not dynamically traceable to ONNX,
        # here `img_shape_for_onnx` (padded shape of image tensor)
        # is used.
        img_shape = img_metas[0]['img_shape_for_onnx']
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        batch_size = cls_scores.size(0)
        # `batch_index_offset` is used for the gather of concatenated tensor
        batch_index_offset = torch.arange(batch_size).to(
            cls_scores.device) * max_per_img
        batch_index_offset = batch_index_offset.unsqueeze(1).expand(
            batch_size, max_per_img)

        # supports dynamical batch inference
        if self.loss_cls.use_sigmoid:
            cls_scores = cls_scores.sigmoid()
            scores, indexes = cls_scores.view(batch_size, -1).topk(
                max_per_img, dim=1)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_index = (bbox_index + batch_index_offset).view(-1)
            bbox_preds = bbox_preds.view(-1, 4)[bbox_index]
            bbox_preds = bbox_preds.view(batch_size, -1, 4)
        else:
            scores, det_labels = F.softmax(
                cls_scores, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img, dim=1)
            bbox_index = (bbox_index + batch_index_offset).view(-1)
            bbox_preds = bbox_preds.view(-1, 4)[bbox_index]
            det_labels = det_labels.view(-1)[bbox_index]
            bbox_preds = bbox_preds.view(batch_size, -1, 4)
            det_labels = det_labels.view(batch_size, -1)

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_preds)
        # use `img_shape_tensor` for dynamically exporting to ONNX
        img_shape_tensor = img_shape.flip(0).repeat(2)  # [w,h,w,h]
        img_shape_tensor = img_shape_tensor.unsqueeze(0).unsqueeze(0).expand(
            batch_size, det_bboxes.size(1), 4)
        det_bboxes = det_bboxes * img_shape_tensor
        # dynamically clip bboxes
        x1, y1, x2, y2 = det_bboxes.split((1, 1, 1, 1), dim=-1)
        from mmdet.core.export import dynamic_clip_for_onnx
        x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, img_shape)
        det_bboxes = torch.cat([x1, y1, x2, y2], dim=-1)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(-1)), -1)

        return det_bboxes, det_labels
