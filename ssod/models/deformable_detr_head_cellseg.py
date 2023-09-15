# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import HEADS
from .detr_head_cellseg import DETRHead_CellSeg
from ..utils.transformer_seg import MLP


@HEADS.register_module()
class DeformableDETRHead_CellSeg(DETRHead_CellSeg):
    """Head of DeformDETR: Deformable DETR: Deformable Transformers for End-to-
    End Object Detection.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 use_dab=False,
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.use_dab = use_dab
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        elif self.use_dab:
            transformer['use_dab'] = self.use_dab

        super(DeformableDETRHead_CellSeg, self).__init__(
            *args, transformer=transformer, **kwargs)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 2))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:

            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            if not self.use_dab:
                self.query_embedding = nn.Embedding(self.num_query,
                                                    self.embed_dims * 2)
            else:
                self.tgt_embed = nn.Embedding(self.num_query,
                                                    self.embed_dims)
                self.refpoint_embed = nn.Embedding(self.num_query, 2)

        self.mask_embed = MLP(self.embed_dims, self.embed_dims, self.embed_dims, 3)
        self.norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """

        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(# 从mask中采样同样的大小
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if self.use_dab:
            tgt_embed = self.tgt_embed.weight
            refanchor = self.refpoint_embed.weight
            query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
        elif not self.as_two_stage:
            query_embeds = self.query_embedding.weight


        hs, init_reference, inter_references, mask_features,\
            enc_outputs_class, enc_outputs_coord, enc_outputs_mask = self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None,  # noqa:E501
                    norm_branches=self.norm,
                    mask_embed_branches=self.mask_embed,
            )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_masks = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            mask_embed = self.mask_embed(self.norm(hs[lvl]))
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_masks.append(outputs_mask)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_masks = torch.stack(outputs_masks)
        if self.as_two_stage:
            return outputs_classes, outputs_coords, outputs_masks, \
                enc_outputs_class, \
                enc_outputs_coord.sigmoid()
        else:
            return outputs_classes, outputs_coords, outputs_masks, \
                None, None

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores,
             all_point_preds,
             enc_cls_scores,
             enc_point_preds,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
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
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)

        # trans bboxes to points
        gt_points_list = []
        for gt_bboxes_li in gt_bboxes_list:
            gt_pointes_li = torch.stack(((gt_bboxes_li[:, 0].clone().detach() + gt_bboxes_li[:,2].clone().detach()) / 2,
                                         (gt_bboxes_li[:, 1].clone().detach() + gt_bboxes_li[:,3].clone().detach()) / 2),
                                        1)
            gt_points_list.append(gt_pointes_li)

        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_points_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_points = multi_apply(
            self.loss_single, all_cls_scores, all_point_preds,
            all_gt_points_list, all_gt_labels_list, img_metas_list,
            all_gt_points_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_point = \
                self.loss_single(enc_cls_scores, enc_point_preds,
                                 gt_points_list, binary_labels_list,
                                 img_metas, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_point

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_points[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_point_i in zip(losses_cls[:-1],
                                                       losses_points[:-1],):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_point_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_points_pred(self,
                   all_cls_scores,
                   all_point_preds,
                   enc_cls_scores,
                   enc_bbox_preds,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
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
        cls_scores = all_cls_scores[-1]
        point_preds = all_point_preds[-1]  # [1, 1000, 2]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            point_pred = point_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor'][:2]
            proposals = self._get_points_single_pred(cls_score, point_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)
        return result_list

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_all_points_pred(self,
                        all_cls_scores,
                        all_point_preds,
                        enc_cls_scores,
                        enc_bbox_preds,
                        img_metas,
                        rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            dict
        """

        points_nb_dec = [[] for i in range(len(all_cls_scores))]
        labels_nb_dec = [[] for i in range(len(all_cls_scores))]

        for idx, (cls_scores, point_preds) in enumerate(zip(all_cls_scores, all_point_preds)):
            for img_id in range(len(img_metas)):
                cls_score = cls_scores[img_id]
                point_pred = point_preds[img_id]
                img_shape = img_metas[img_id]['img_shape']
                scale_factor = img_metas[img_id]['scale_factor'][:2]
                proposals = self._get_points_single_pred(cls_score, point_pred,
                                                         img_shape, scale_factor,
                                                         rescale)  # [[xy, score],cls]
                points_nb_dec[idx].append(proposals[0])
                labels_nb_dec[idx].append(proposals[1])

        return points_nb_dec, labels_nb_dec
