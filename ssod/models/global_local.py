import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector
import numpy as np
from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n
from collections.abc import Sequence
from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid


@DETECTORS.register_module()
class GlobalLocal(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(GlobalLocal, self).__init__(
            dict(globals=build_detector(model), locals=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("globals")
            self.global_weight = self.train_cfg.global_weight

    def loss(self,
             all_cls_scores,
             all_point_preds,
             enc_cls_scores,
             enc_point_preds,
             gt_points_list,
             gt_labels_list,
             img_metas,
             mode='local',
             gt_bboxes_ignore=None):

        num_dec_layers = len(all_cls_scores)
        loss_single = eval('self.{}s.bbox_head.loss_single'.format(mode))
        if mode == 'local':
            all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
            all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
            all_gt_points_ignore_list = [
                gt_bboxes_ignore for _ in range(num_dec_layers)
            ]
            img_metas_list = [img_metas for _ in range(num_dec_layers)]

        else:
            all_gt_points_list = gt_points_list
            all_gt_labels_list = gt_labels_list
            all_gt_points_ignore_list = [
                gt_bboxes_ignore for _ in range(num_dec_layers)
            ]
            img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_points = multi_apply(
            loss_single, all_cls_scores, all_point_preds,
            all_gt_points_list, all_gt_labels_list, img_metas_list,
            all_gt_points_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i]) if mode == "local" else torch.zeros_like(gt_labels_list[-1][i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_point = \
                loss_single(enc_cls_scores, enc_point_preds,
                            gt_points_list if mode == "local" else gt_points_list[-1],
                            binary_labels_list,
                            img_metas, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_point

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_points[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_point_i in zip(losses_cls[:-1],
                                            losses_points[:-1], ):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_point_i
            num_dec_layer += 1
        return loss_dict

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        # ! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        # ! it means that at least one sample for each group should be provided on each gpu.
        # ! In some situation, we can only put one image per gpu, we have to return the sum of loss
        # ! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "local" in data_groups:
            local_feat = self.locals.extract_feat(data_groups["local"]["img"])
            local_outs = self.locals.bbox_head.forward(local_feat, data_groups["local"]["img_metas"])
            local_loss_inputs = local_outs + (
                data_groups["local"]["gt_points"], data_groups["local"]["gt_labels"], data_groups["local"]["img_metas"])

            local_loss = self.loss(*local_loss_inputs, mode="local")
            local_loss = {"local_" + k: v for k, v in local_loss.items()}
            loss.update(**local_loss)
        if "global" in data_groups:
            global_loss = weighted_loss(
                self.foward_global_train(
                    data_groups["global"], data_groups["local"], local_outs
                ),
                weight=self.global_weight,
            )
            global_loss = {"global_" + k: v for k, v in global_loss.items()}
            loss.update(**global_loss)

        return loss

    def foward_global_train(self, global_data, local_data, local_outs):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in global_data["img_metas"]]
        snames = [meta["filename"] for meta in local_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            global_info = self.extract_global_info(
                global_data["img"][
                    torch.Tensor(tidx).to(global_data["img"].device).long()
                ],
                [global_data["img_metas"][idx] for idx in tidx],
                [global_data["proposals"][idx] for idx in tidx]
                if ("proposals" in global_data)
                   and (global_data["proposals"] is not None)
                else None,
            )

        offset = [torch.from_numpy(meta["bbox_offset"]).to(global_info["det_points"][0][0].device) for meta in local_data['img_metas']]
        crop_size = [torch.tensor(meta["crop_size"]).to(global_info["det_points"][0][0].device) for meta in local_data['img_metas']]
        pseudo_points_ = [[bat[:, :2].detach() for bat in lay] for lay in global_info["det_points"]]
        pseudo_labels_ = global_info["det_labels"]

        pseudo_points = [[] for i in range(len(pseudo_points_))]
        pseudo_labels = [[] for i in range(len(pseudo_points_))]
        for lay, (det_points, det_labels) in enumerate(zip(pseudo_points_, pseudo_labels_)):
            for det_bat, det_label_bat, offset_bat, crop_size_bat in zip(det_points, det_labels, offset, crop_size):
                points, labels = self.crop_valid(
                    det_bat,
                    det_label_bat,
                    offset_bat,
                    crop_size_bat
                )
                pseudo_points[lay].append(points)
                pseudo_labels[lay].append(labels)

        local_matrix = [torch.from_numpy(meta['transform_matrix']).to(global_info['transform_matrix'][0].device).float() for
                        meta in local_data['img_metas']]
        M = self._get_trans_mat(global_info['transform_matrix'], local_matrix)
        pseudo_points = [self._transform_points(
            det_points,
            M,
        ) for det_points in pseudo_points]
        global_input = local_outs + (pseudo_points, pseudo_labels, global_data["img_metas"])
        global_loss = self.loss(*global_input, mode="global")

        return global_loss

    def crop_valid(self, det_points, det_labels, offset, crop_size):
        valid_inds = (det_points[:, 0] >= offset[0]) & (det_points[:, 1] >= offset[1]) & (
                    det_points[:, 0] <= offset[0] + crop_size[1]) & (det_points[:, 1] <= offset[1] + crop_size[0])
        return det_points[valid_inds, :] - offset[:2], det_labels[valid_inds]

    @force_fp32(apply_to=["points", "trans_mat"])
    def _transform_points(self, points, trans_mat):
        points = Transform2D.transform_points(points, trans_mat)
        return points

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_global_info(self, img, img_metas, proposals=None, **kwargs):
        global_info = {}
        feat = self.globals.extract_feat(img)
        global_info["backbone_feature"] = feat
        global_info["proposals"] = proposals

        proposal_list, proposal_label_list = self.globals.bbox_head.simple_all_test_bboxes(
            feat, img_metas, rescale=False
        )

        proposal_list = [[p_b.to(feat[0].device) for p_b in p] for p in proposal_list]
        proposal_list = [
            [p_b if p_b.shape[0] > 0 else p_b.new_zeros(0, 3) for p_b in p] for p in proposal_list
        ]
        proposal_label_list = [[p_b.to(feat[0].device) for p_b in p] for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")

        for num_layer in range(len(proposal_list)):
            for batch_idx in range(len(proposal_list[num_layer])):
                valid = proposal_list[num_layer][batch_idx][:, -1] > thr
                proposal_list[num_layer][batch_idx] = proposal_list[num_layer][batch_idx][valid]
                proposal_label_list[num_layer][batch_idx] = proposal_label_list[num_layer][batch_idx][valid]

        det_points = proposal_list
        det_labels = proposal_label_list
        global_info["det_points"] = det_points
        global_info["det_labels"] = det_labels
        global_info["transform_matrix"] = [
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        global_info["img_metas"] = img_metas
        return global_info

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        if not any(["globals" in key or "locals" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"globals." + k: state_dict[k] for k in keys})
            state_dict.update({"locals." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

